r"""The shared refinement function $F_\theta$ — the "lens" (v2).

At each deliberation step k the lens produces a residual delta composed of
two fused phases, now framed as a fluid simulator:

  1. Additive (advection):    h_add_k = MLP_add(h_{k-1})
  2. Subtractive (projection): divergence-project h_add to enforce
     incompressibility (constant latent mass), via a learned per-sample
     projection scalar applied to the Sobel-diverggence field, followed by
     an L2 mass renormalization.

  3. Residual update:          h_k = h_{k-1} + tanh(h_projected)

The latent is always spatial: flat [B, d] in/out for the MLPs, reshaped to
[B, C, 4, 4] for the divergence operator. The additive phase is advection;
the subtractive phase is the CFD projection step that redistributes density
rather than zeroing overlap.

There is no action modality in the MOVi dataset — the lens refines a purely
visual latent toward a physically-grounded target, so the additive phase
consumes only the previous latent.
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
