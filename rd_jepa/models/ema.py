r"""EMA target encoder wrapper for the JEPA stop-gradient target.

Maintains an exponential moving average of the context encoder's weights
(like BYOL/I-JEPA). The target encoder is never updated by gradients;
only `update_ema()` is called each step. Stop-gradient is applied in the
loss, not here, but we expose `no_grad` forward for clarity.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class EMATargetEncoder(nn.Module):
    """Wraps a copy of the context encoder whose weights track it via EMA."""

    def __init__(self, encoder: nn.Module, decay: float = 0.996):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.encoder.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update_ema(self, encoder: nn.Module, step: int, warmup: int = 100) -> None:
        """Smoothly approach the context encoder's weights.

        During warmup we use a lower decay so the target catches up quickly;
        after warmup the configured (high) decay is used.
        """
        d = min(self.decay, 1.0 - 1.0 / max(step + 1, warmup))
        for p_ema, p in zip(self.encoder.parameters(), encoder.parameters(), strict=True):
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)
        for b_ema, b in zip(self.encoder.buffers(), encoder.buffers(), strict=True):
            b_ema.data.copy_(b.data)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
