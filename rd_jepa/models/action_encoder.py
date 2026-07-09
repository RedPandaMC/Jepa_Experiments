r"""Action encoder $A_\phi$ mapping a PhyRE action (x, y, r) into the latent space.

A 2-layer MLP that projects the 3-d action into the same latent space as
the encoder output, so it can be concatenated or added during deliberation.
"""
from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int = 3, latent_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """[B, 3] -> [B, latent_dim]."""
        return self.net(action)
