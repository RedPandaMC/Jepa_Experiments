r"""The shared refinement function $F_\theta$ — the "lens".

At each deliberation step k the lens produces a residual delta composed of
two fused phases (see spec §2.2):

  1. Additive (extrapolation): h_add_k = MLP_add(h_{k-1}, a_t)
  2. Subtractive (masking):    M_k = Gate(MLP_mask(h_add_k))
  3. Residual update:          h_k = h_{k-1} + tanh(M_k \odot h_add_k)

The gate is configurable (sigmoid vs sparsemax) to support the Decision-1
ablation. The action a_t can be injected every step or only at k=0
(Decision-2), controlled by the caller via the `inject_action` flag.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Sparsemax(nn.Module):
    """Sparsemax activation (Martins & Astudillo 2016) — projects to a
    probability simplex but can output exact zeros for hard pruning."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # operate over the last dim
        x_sorted, _ = torch.sort(x, dim=-1, descending=True)
        cumsum = torch.cumsum(x_sorted, dim=-1)
        rho = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        support = (x_sorted - (cumsum - 1) / rho) > 0
        k = support.sum(dim=-1).clamp(min=1)
        tau = (cumsum.gather(-1, (k - 1).unsqueeze(-1)) - 1) / k.unsqueeze(-1)
        return F.relu(x - tau)


class DeliberationStep(nn.Module):
    """One application of the lens (shared weights, reused K times)."""

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 512,
        gate: str = "sigmoid",
    ):
        super().__init__()
        self.gate_type = gate

        # Additive phase: (h, a) -> h_add
        self.mlp_add = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Subtractive phase: h_add -> mask (same dim as latent)
        self.mlp_mask = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        if gate == "sigmoid":
            self.gate = nn.Sigmoid()
        elif gate == "sparsemax":
            self.gate = Sparsemax()
        else:
            raise ValueError(f"unknown gate: {gate}")

    def forward(self, h: torch.Tensor, a: torch.Tensor | None) -> torch.Tensor:
        """Apply one refinement step.

        Args:
            h: [B, latent_dim] previous latent.
            a: [B, action_dim] encoded action, or None when action_inject='once'
                and k > 0.
        Returns:
            h_next: [B, latent_dim] refined latent (residual update).
        """
        if a is not None:
            x = torch.cat([h, a], dim=-1)
        else:
            x = torch.cat([h, torch.zeros_like(h)], dim=-1)
        h_add = self.mlp_add(x)
        mask = self.gate(self.mlp_mask(h_add))
        delta = torch.tanh(mask * h_add)
        return h + delta


class ViolationHead(nn.Module):
    r"""$V_\psi$: predicts the physical-error (energy) of a latent state.

    A lightweight linear head producing a scalar per sample; used for the
    early-exit decision (spec §2.3). Trained to predict the residual latent
    error to the true target so it needs no extra labels.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] -> [B] scalar violation scores."""
        return self.net(h).squeeze(-1)
