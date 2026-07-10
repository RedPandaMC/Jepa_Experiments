r"""The lens bank — a turret of N soft-routed specialist lenses (v2).

At each deliberation step k a router reads $h_{k-1}$ and emits softmax
weights over N specialist lenses; the bank's update is the weighted sum of
the N lens deltas, mass-renormalized to $\|\overline{\Delta}_i\|$ (the mean
delta magnitude) and bounded by tanh:

  1. Each lens $i$ produces its additive-then-projected delta
     $\Delta_i = F_\theta^{(i)}(h_{k-1}) - h_{k-1}$.
  2. Router: $g = \mathrm{softmax}(\mathrm{router}(h_{k-1})) \in \mathbb{R}^N$.
  3. Mixed delta: $\Delta = \sum_i g_i\,\Delta_i$.
  4. Mass renorm: rescale $\Delta$ so $\|\Delta\|_2 \approx \|\overline{\Delta}_i\|_2$.
  5. Residual update: $h_k = h_{k-1} + \tanh(\Delta)$.

Each specialist lens keeps its own additive MLP and divergence projection
(per-lens `mlp_alpha` for specialization; the fixed Sobel kernels are
identical across lenses). The single `ViolationHead` scores the state
regardless of which lens produced it.

There is no action modality in the MOVi dataset — the lens bank refines a
purely visual latent toward a physically-grounded target.

When `n_lenses == 1` the router is not created and the single lens is
called directly, reproducing the original single-lens path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DivergenceProjection(nn.Module):
    r"""CFD-style projection step on a spatial latent.

    Computes the discrete divergence of the additive update via fixed Sobel
    kernels (depthwise convolution), measures per-channel density, learns a
    per-sample projection coefficient `alpha`, and applies

        h_proj = h_sp - alpha * div(h_sp)

    then rescales so `||h_proj||_2 ≈ ||h_sp||_2` (differentiable mass
    preservation). This is the incompressibility projection: density may
    redistribute but cannot be created or destroyed — the lens cannot
    "cheat" by zeroing the latent to avoid physical-violation loss.
    """

    def __init__(self, latent_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_channels = latent_channels

        # Fixed Sobel kernels (registered as buffers, not learned).
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        )
        sobel_y = sobel_x.t().contiguous()
        # [C, 1, 3, 3] depthwise conv weights
        sx = sobel_x.unsqueeze(0).unsqueeze(0).expand(latent_channels, 1, 3, 3)
        sy = sobel_y.unsqueeze(0).unsqueeze(0).expand(latent_channels, 1, 3, 3)
        self.register_buffer("sobel_x", sx.contiguous())
        self.register_buffer("sobel_y", sy.contiguous())

        # Per-sample projection coefficient from per-channel density.
        self.mlp_alpha = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_sp: torch.Tensor) -> torch.Tensor:
        """Apply the divergence projection to a spatial latent.

        Args:
            h_sp: [B, C, H, W] additive update (spatial).
        Returns:
            h_proj: [B, C, H, W] projected, mass-preserving update.
        """
        # Discrete divergence via depthwise Sobel convs (reflect padding so
        # the 4x4 field doesn't shrink).
        div_x = F.conv2d(h_sp, self.sobel_x, padding=1, groups=self.latent_channels)
        div_y = F.conv2d(h_sp, self.sobel_y, padding=1, groups=self.latent_channels)
        div = div_x + div_y  # [B, C, H, W]

        # Per-channel density (proxy for activation "mass").
        rho = h_sp.pow(2).mean(dim=(2, 3))  # [B, C]
        alpha = torch.sigmoid(self.mlp_alpha(rho))  # [B, 1] in (0, 1)
        alpha = alpha.view(-1, 1, 1, 1)  # broadcast over [C, H, W]

        # Projection: subtract a fraction of the divergence field.
        h_proj = h_sp - alpha * div

        # Mass (L2) renormalization so the projection redistributes rather
        # than destroys activation. Rescale per-sample to match the input
        # magnitude. Clamp the input norm away from zero to avoid div-by-0.
        in_norm = torch.norm(h_sp.flatten(1), p=2, dim=-1)  # [B]
        out_norm = torch.norm(h_proj.flatten(1), p=2, dim=-1)  # [B]
        scale = (in_norm / (out_norm + 1e-6)).view(-1, 1, 1, 1)
        h_proj = h_proj * scale

        return h_proj


class DeliberationStep(nn.Module):
    """One application of the lens (shared weights, reused K times)."""

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 512,
        latent_channels: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.spatial_side = 4  # 4x4 spatial latent
        assert latent_dim == latent_channels * self.spatial_side * self.spatial_side

        # Additive phase (advection): h_flat -> h_add_flat
        self.mlp_add = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Subtractive phase (projection): the divergence operator on the
        # reshaped spatial latent.
        self.projection = DivergenceProjection(
            latent_channels=latent_channels, hidden_dim=hidden_dim
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply one refinement step.

        Args:
            h: [B, latent_dim] previous latent (flat).
        Returns:
            h_next: [B, latent_dim] refined latent (residual update).
        """
        h_add = self.mlp_add(h)  # [B, d]
        # Reshape to spatial for the divergence/projection step.
        h_add_sp = h_add.view(-1, self.latent_channels, self.spatial_side, self.spatial_side)
        h_proj_sp = self.projection(h_add_sp)  # [B, C, 4, 4]
        h_proj = h_proj_sp.flatten(1)  # [B, d]
        delta = torch.tanh(h_proj)
        return h + delta


class ViolationHead(nn.Module):
    r"""$V_\psi$: predicts the physical-error (energy) of a latent state.

    A lightweight linear head producing a scalar per sample; used for the
    early-exit decision (spec §2.3). Trained to predict the residual latent
    error to the true target (self-supervised) AND a grounded collision-force
    regression target derived from MOVi's per-frame collision events.
    """

    def __init__(self, latent_dim: int = 1024, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] -> [B] scalar violation scores."""
        return self.net(h).squeeze(-1)


class LensBank(nn.Module):
    r"""A turret of N specialist lenses combined by a soft-routing router.

    Each lens is a `DeliberationStep` (additive MLP + divergence projection).
    A lightweight router reads $h_{k-1}$ and emits softmax weights over the
    N lenses; the bank's update is the weighted sum of the N lens deltas,
    mass-renormalized to the mean delta magnitude and bounded by tanh.

    When ``n_lenses == 1`` the router is not created and the single lens is
    called directly (exact single-lens path, ``gate`` is ``None``).
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 512,
        latent_channels: int = 64,
        n_lenses: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_lenses = n_lenses
        self.lenses = nn.ModuleList(
            [
                DeliberationStep(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    latent_channels=latent_channels,
                )
                for _ in range(n_lenses)
            ]
        )
        # Router only when there is a real choice to make.
        if n_lenses > 1:
            self.router: nn.Sequential | None = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_lenses),
            )
        else:
            self.router = None

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one bank refinement step.

        Args:
            h: [B, latent_dim] previous latent (flat).
        Returns:
            (h_next, gate): h_next is [B, latent_dim]; gate is [B, N] softmax
            weights, or None when n_lenses == 1.
        """
        if self.n_lenses == 1:
            return self.lenses[0](h), None

        # Per-lens deltas: F_i(h) - h  (the update, not h_next).
        deltas = torch.stack(
            [lens(h) - h for lens in self.lenses], dim=1
        )  # [B, N, d]

        logits = self.router(h)  # type: ignore[misc]  # [B, N]
        gate = torch.softmax(logits, dim=-1)  # [B, N]

        mixed = (gate.unsqueeze(-1) * deltas).sum(dim=1)  # [B, d]

        # Mass renormalization: rescale the mixed delta so its magnitude
        # matches the per-sample mean delta magnitude (preserve the
        # incompressibility spirit when soft-mixing lens outputs).
        mean_delta = deltas.mean(dim=1)  # [B, d]
        in_norm = torch.norm(mean_delta, p=2, dim=-1)  # [B]
        out_norm = torch.norm(mixed, p=2, dim=-1)  # [B]
        scale = (in_norm / (out_norm + 1e-6)).unsqueeze(-1)  # [B, d]
        mixed = mixed * scale

        delta = torch.tanh(mixed)
        return h + delta, gate
