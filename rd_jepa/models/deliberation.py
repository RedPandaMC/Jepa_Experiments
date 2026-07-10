r"""The kernel lens — depthwise conv kernels that mutate during test-time compute.

The lenses are a bank of N depthwise convolution kernels that operate on the
spatial latent ``[B, C, H, W]``. Unlike the previous MoE lens bank (N parallel
MLPs + a degenerate soft-routing router), these kernels are:

1. **Kernels on the latent space** — each lens is a ``[C, kH, kW]`` depthwise
   conv filter applied to the spatial latent via ``unfold`` + ``einsum``. This
   gives them genuine spatial inductive bias (they detect/produce local
   field patterns) rather than being unconstrained MLPs on a flat vector.

2. **Mutating** — the kernels are NOT static across the K deliberation steps.
   A mutator network reads the pooled latent and produces per-sample kernel
   deltas at each step. The kernel state ``[B, N, C, kH, kW]`` is carried
   across K steps, so each sample's kernels diverge from the learned base
   based on its own latent trajectory. This makes test-time compute
   meaningful: different inputs lead to different kernel evolutions.

3. **Attention-gated** — a lightweight gate reads the latent and produces
   ``[B, N]`` softmax weights over the N kernels. The gated sum of per-kernel
   spatial deltas is ``tanh``-bounded and added residually. No MoE
   load-balance / router-entropy losses are needed; a kernel diversity loss
   (on the base kernels) prevents collapse to identical filters.

Initialization: the base kernels are seeded with physics-inspired priors
(Sobel-x, Sobel-y, Laplacian, identity) so the bank starts with meaningful
spatial operators and then adapts via training + per-sample mutation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _sobel_x_kernel() -> torch.Tensor:
    return torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )


def _sobel_y_kernel() -> torch.Tensor:
    return _sobel_x_kernel().t().contiguous()


def _laplacian_kernel() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    )


def _identity_kernel() -> torch.Tensor:
    k = torch.zeros(3, 3)
    k[1, 1] = 1.0
    return k


class KernelLens(nn.Module):
    r"""Lenses as mutating depthwise conv kernels on the spatial latent.

    State carried across K steps:
      - ``h``: the flat latent ``[B, d]`` (reshaped to ``[B, C, H, W]`` internally)
      - ``kernel_state``: per-sample kernels ``[B, N, C, kH, kW]``

    Each step:
      1. Apply each kernel to ``h`` (per-sample depthwise conv) → deltas ``[B, N, C, H, W]``
      2. Gate: softmax attention over kernels from pooled latent → ``[B, N]``
      3. Combine: gated sum of deltas → ``[B, C, H, W]``; ``tanh``-bounded; residual add
      4. Mutate: mutator reads updated latent → per-sample kernel deltas → kernels evolve

    The kernel state is initialized from learned base kernels (seeded with
    Sobel / Laplacian / identity priors) and diverges per-sample.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        latent_channels: int = 64,
        spatial_side: int = 4,
        n_kernels: int = 4,
        kernel_size: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.spatial_side = spatial_side
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        assert latent_dim == latent_channels * spatial_side * spatial_side

        C = latent_channels
        k = kernel_size

        # Base kernels: [N, C, kH, kW]. Seeded with physics-inspired priors.
        base = self._init_base_kernels(n_kernels, C, k)
        self.base_kernels = nn.Parameter(base)

        # Gate: reads pooled latent [B, C] -> [B, N] attention weights.
        self.gate_norm = nn.LayerNorm(C)
        self.gate = nn.Sequential(
            nn.Linear(C, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_kernels),
        )
        self.gate_temperature = nn.Parameter(torch.tensor(1.0))

        # Mutator: reads pooled latent [B, C] -> [B, N*C*k*k] kernel deltas.
        self.mutator = nn.Sequential(
            nn.Linear(C, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_kernels * C * k * k),
        )

        # Learnable scales (bounded by tanh to keep updates stable).
        self.step_scale = nn.Parameter(torch.tensor(0.25))
        self.mutation_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _init_base_kernels(
        n_kernels: int, channels: int, kernel_size: int
    ) -> torch.Tensor:
        """Seed base kernels with physics-inspired spatial operators.

        For kernel_size=3: [sobel_x, sobel_y, laplacian, identity, ...random].
        Each channel starts with the same spatial pattern; training +
        per-sample mutation will differentiate them.
        """
        if kernel_size != 3:
            return torch.randn(n_kernels, channels, kernel_size, kernel_size) * 0.02

        priors = [_sobel_x_kernel(), _sobel_y_kernel(), _laplacian_kernel(), _identity_kernel()]
        base = torch.empty(n_kernels, channels, kernel_size, kernel_size)
        for i in range(n_kernels):
            pattern = priors[i % len(priors)]
            base[i] = pattern.unsqueeze(0).expand(channels, kernel_size, kernel_size)
        return base * 0.5  # scale down from the ±1/±2 prior range

    def init_kernels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize per-sample kernel state from the learned base kernels."""
        return (
            self.base_kernels.unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
            .contiguous()
            .to(device)
        )

    def forward(
        self, h: torch.Tensor, kernel_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One kernel-lens refinement step.

        Args:
            h: ``[B, latent_dim]`` flat latent.
            kernel_state: ``[B, N, C, kH, kW]`` current per-sample kernels.

        Returns:
            (h_next, gate, new_kernel_state)
            - h_next: ``[B, latent_dim]`` refined latent (residual update).
            - gate: ``[B, N]`` attention weights over kernels.
            - new_kernel_state: ``[B, N, C, kH, kW]`` mutated kernels.
        """
        B = h.shape[0]
        C = self.latent_channels
        H = W = self.spatial_side
        N = self.n_kernels
        k = self.kernel_size

        # Match kernel_state dtype to h (for autocast compatibility).
        if kernel_state.dtype != h.dtype:
            kernel_state = kernel_state.to(h.dtype)

        # Reshape flat latent to spatial for depthwise conv.
        h_sp = h.view(B, C, H, W)

        # Pool latent for gate and mutator inputs.
        h_pool = h_sp.mean(dim=(2, 3))  # [B, C]

        # Per-sample depthwise convolution with per-sample kernels.
        # Unfold spatial latent into sliding-window patches.
        patches = F.unfold(h_sp, k, padding=k // 2)  # [B, C*k*k, H*W]
        patches = patches.view(B, C, k * k, H * W)  # [B, C, k*k, L]

        # Reshape kernel state: [B, N, C, k*k]
        ks_flat = kernel_state.reshape(B, N, C, k * k)

        # Depthwise conv (per-channel): deltas[b,n,c,l] = sum_j ks[b,n,c,j] * patches[b,c,j,l]
        deltas = torch.einsum("bnck,bckl->bncl", ks_flat, patches)  # [B, N, C, L]

        # Attention gate: which kernels to activate for this latent state.
        gate_logits = self.gate(self.gate_norm(h_pool))  # [B, N]
        gate = F.softmax(
            gate_logits / self.gate_temperature.clamp(min=0.5), dim=-1
        )  # [B, N]

        # Weighted sum of per-kernel spatial deltas.
        gate_exp = gate.view(B, N, 1, 1)  # [B, N, 1, 1]
        delta_sp = (gate_exp * deltas).sum(dim=1)  # [B, C, L]
        delta_sp = delta_sp.view(B, C, H, W)

        # Flatten and bound the update.
        delta = delta_sp.flatten(1)  # [B, d]
        delta = torch.tanh(self.step_scale) * torch.tanh(delta)

        # Residual update.
        h_next = h + delta

        # Mutate kernels based on the updated latent state.
        # The kernels literally evolve — this is the test-time compute:
        # different inputs produce different kernel trajectories.
        h_next_sp = h_next.view(B, C, H, W)
        h_next_pool = h_next_sp.mean(dim=(2, 3))  # [B, C]
        kernel_delta = self.mutator(h_next_pool)  # [B, N*C*k*k]
        kernel_delta = kernel_delta.view(B, N, C, k, k)
        kernel_delta = torch.tanh(kernel_delta)  # bound each element to [-1, 1]
        new_kernel_state = kernel_state + torch.tanh(self.mutation_scale) * kernel_delta

        return h_next, gate, new_kernel_state


class ViolationHead(nn.Module):
    r"""$V_\psi$: predicts the physical-error (energy) of a latent state.

    A lightweight linear head producing a scalar per sample; used for the
    early-exit decision and grounded collision-force regression.
    """

    def __init__(self, latent_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] -> [B] scalar violation scores."""
        return self.net(self.norm(h)).squeeze(-1)
