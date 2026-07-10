r"""Loss functions for RD-JEPA v4 (simplified).

  1. Latent prediction loss (JEPA): MSE between h_K and the stop-gradient
     EMA target encoder of s_{t+1}. Final-only (no discounted trajectory).
  2. VICReg variance + covariance regularization (collapse safety net).

All trajectory-based losses (energy, contrastive, divergence, violation aux,
violation supervision, violation grounded, kernel diversity) have been
removed — VICReg alone is a proven, sufficient collapse-prevention method,
and the reduced loss surface makes BPTT through K steps far cheaper and
more stable on consumer hardware.
"""
from __future__ import annotations

import torch

from .config import Config


def latent_prediction_loss(
    h_final: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between the final predicted latent and the stop-gradient target.

    Args:
        h_final: [B, d] predicted latent.
        target:  [B, d] stop-grad EMA target of s_{t+1}.
    """
    return torch.nn.functional.mse_loss(h_final, target)


def vicreg_variance_loss(
    z: torch.Tensor,  # [B, d]
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg variance loss: penalize low standard deviation per dimension."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(torch.relu(target_std - std))


def vicreg_covariance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg covariance loss: penalize off-diagonal covariance."""
    B, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)  # [d, d]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2)).sum() / d


def total_loss(
    h_final: torch.Tensor,
    target: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total RD-JEPA v4 loss + a dict of metric names for logging.

    Only three terms: JEPA MSE + VICReg variance + VICReg covariance.
    All loss weights are read from `cfg`.
    """
    l_jepa = latent_prediction_loss(h_final, target)
    l_var = vicreg_variance_loss(h_final, target_std=cfg.vicreg_target_std)
    l_cov = vicreg_covariance_loss(h_final)

    total = (
        l_jepa
        + cfg.vicreg_var_weight * l_var
        + cfg.vicreg_cov_weight * l_cov
    )
    metrics: dict[str, float] = {
        "loss/total": total.detach().float().item(),
        "loss/jepa": l_jepa.detach().float().item(),
        "loss/vicreg_variance": l_var.detach().float().item(),
        "loss/vicreg_covariance": l_cov.detach().float().item(),
        "repr/std_mean": h_final.std(dim=0).mean().detach().float().item(),
    }
    return total, metrics
