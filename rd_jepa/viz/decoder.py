r"""Asynchronous probing decoder (v2 core fix #1).

The JEPA loss never decodes pixels (spec §3.2). The decoder is a separate,
lightweight "viewport" trained on the **frozen** h_K latent: it learns to
invert the encoder for visualization without entangling its gradients with
the JEPA backward pass. Trained in its own optimizer + step cadence
(see rd_jepa/train.py:train_decoder_step).

It decodes a flat latent [B, d] -> [B, 3, 64, 64] RGB frame via 4
conv-transpose blocks. ~200K params, negligible VRAM.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import Config


class VizDecoder(nn.Module):
    """Tiny decoder for visualization only; not part of the JEPA loss."""

    def __init__(self, latent_dim: int = 1024, out_channels: int = 3):
        super().__init__()
        self.out_channels = out_channels
        # project latent -> [B, 256, 4, 4]
        self.proj = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, d] -> [B, C, 64, 64] reconstructed frame in [0,1]."""
        x = self.proj(h).view(-1, 256, 4, 4)
        return self.net(x)

    def decoder_loss(self, h: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """MSE between the decoded frame and the original input frame.

        Trains the decoder to invert the encoder for visualization. This
        loss is applied ONLY to the decoder parameters; gradients do NOT
        flow into the encoder/lens via this path (h is detached by the
        caller in train_decoder_step).
        """
        pred = self.forward(h)  # caller passes h.detach()
        return torch.nn.functional.mse_loss(pred, s_t)


def make_decoder_optimizer(
    decoder: VizDecoder, cfg: Config
) -> torch.optim.Optimizer:
    """Build the dedicated AdamW for the asynchronous probing decoder."""
    return torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg.decoder_lr,
        weight_decay=cfg.decoder_weight_decay,
    )

