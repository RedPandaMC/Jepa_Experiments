r"""Patch encoder for multivariate time-series.

Maps a context window [B, L, C] to a latent vector [B, d] via:
    1. 1D patching (Conv1d with stride = kernel = patch_len)
    2. Adaptive pooling to fixed n_patches
    3. LayerNorm + 2-layer MLP
"""
from __future__ import annotations

import torch
from torch import nn


class PatchEncoder(nn.Module):
    """Patch embedding + MLP for time-series → latent.

    Handles variable input lengths via adaptive pooling so the same
    encoder can process context windows (L=144) and target windows
    (H=72) — the EMA target encoder needs to process both.
    """

    def __init__(
        self,
        in_channels: int = 14,
        patch_len: int = 6,
        latent_dim: int = 256,
        n_patches: int = 24,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.latent_dim = latent_dim
        self.n_patches = n_patches

        self.patch_proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=latent_dim,
            kernel_size=patch_len,
            stride=patch_len,
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, latent_dim) * 0.02
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_patches)

        layers: list[nn.Module] = []
        d_in = latent_dim * n_patches
        for _ in range(n_layers):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.GELU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, latent_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, C] → [B, d]."""
        x = x.transpose(1, 2)  # [B, C, L]
        patches = self.patch_proj(x)  # [B, d, n_patches_raw]
        # Adaptive pool to fixed number of patches (handles variable L)
        patches = self.adaptive_pool(patches)  # [B, d, n_patches]
        patches = patches.transpose(1, 2)  # [B, n_patches, d]
        patches = patches + self.pos_embed
        flat = patches.reshape(patches.shape[0], -1)
        z = self.mlp(flat)
        z = self.norm(z)
        return z  # type: ignore[no-any-return]
