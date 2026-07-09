"""Evaluation utilities: linear probe for the violation target.

These connect the RD-JEPA latent space to a downstream task — predicting
the grounded collision-force violation target from the final latent h_K.
This validates that the world model is learning useful physical
representations. (The PhyRE-specific AUCCESS ranking metric has been
removed since MOVi has no action-solve ranking.)
"""
from __future__ import annotations

import torch
from torch import nn


class ViolationProbe(nn.Module):
    """Linear probe: predicts the grounded violation_gt from h_K.

    A simple downstream regression task that validates whether the learned
    representations capture collision/physics information. The probe is a
    single linear layer trained on frozen RD-JEPA latents; its MSE/R²
    against the collision-force target measures representation quality.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, d] -> [B] predicted violation."""
        return self.fc(h).squeeze(-1)

    def compute_metrics(
        self, h: torch.Tensor, violation_gt: torch.Tensor
    ) -> dict[str, float]:
        """Compute MSE and R² given latents and ground-truth targets."""
        with torch.no_grad():
            pred = self.forward(h)  # [B]
            target = violation_gt.float().clamp(0.0, 1.0)
            mse = torch.nn.functional.mse_loss(pred, target).item()
            # R² (coefficient of determination): 1 - SS_res/SS_tot
            ss_res = ((pred - target) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum().clamp(min=1e-8)
            r2 = (1.0 - ss_res / ss_tot).item()
            return {
                "probe/mse": mse,
                "probe/r2": r2,
                "probe/violation_mean": target.mean().item(),
                "probe/pred_mean": pred.mean().item(),
            }
