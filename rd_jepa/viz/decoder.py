r"""Viz-only pixel decoder.

The JEPA loss never decodes pixels (spec §3.2). But a POC needs to *show*
the latent focusing, so we attach a tiny auxiliary decoder trained
jointly ONLY to visualize latents — it is detached from the JEPA loss
and trained on a separate MSE to reproduce s_t from the encoder output.

It decodes a flat latent [B, d] -> [B, 1, 64, 64] via 4 conv-transpose
blocks. ~200K params, negligible VRAM.
"""
from __future__ import annotations

import torch
from torch import nn


class VizDecoder(nn.Module):
    """Tiny decoder for visualization only; not part of the JEPA loss."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # project latent -> [B, 256, 4, 4]
        self.proj = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, d] -> [B, 1, 64, 64] reconstructed frame in [0,1]."""
        x = self.proj(h).view(-1, 256, 4, 4)
        return self.net(x)

    def decoder_loss(self, h: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """MSE between the decoded frame and the original input frame.

        Trains the decoder to invert the encoder for visualization. This
        loss is applied ONLY to the decoder parameters; gradients do NOT
        flow into the encoder/lens via this path.
        """
        pred = self.forward(h.detach())  # detach: decoder must not influence JEPA
        return torch.nn.functional.mse_loss(pred, s_t)
