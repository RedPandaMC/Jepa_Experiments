r"""Loss functions for CK-JEPA (Resonant Decomposition JEPA).

Four terms:
    1. JEPA latent prediction loss (MSE between h_K and stop-grad EMA target).
    2. VICReg variance — keep each latent dimension's std above a target.
    3. VICReg covariance — decorrelate latent dimensions.
    4. Phase diversity — keep oscillator phases spread across the unit circle,
       preventing the N modes from collapsing to a single synchronized cluster.
"""
from __future__ import annotations

import torch

from .config import Config


def latent_prediction_loss(
    h_final: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted latent and stop-gradient EMA target."""
    return torch.nn.functional.mse_loss(h_final, target)


def vicreg_variance_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Penalize low per-dimension standard deviation."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(torch.relu(target_std - std))


def vicreg_covariance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Penalize off-diagonal covariance (decorrelate dimensions)."""
    B, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2)).sum() / d


def phase_diversity_loss(
    phases: torch.Tensor,  # [B, N]
    eps: float = 1e-4,
) -> torch.Tensor:
    r"""Prevent oscillator phase collapse.

    Measures the spread of phases on the unit circle via the magnitude of
    the mean resultant vector. When all phases are identical the mean
    vector has magnitude 1 (minimum diversity). When phases are uniformly
    spread the magnitude approaches 0 (maximum diversity).

    Loss = mean over batch of |mean_resultant_vector|
         = mean of |Σ_i exp(j φ_i)| / N

    Minimizing this pushes phases apart → diverse oscillator modes.
    """
    B, N = phases.shape
    # unit-circle vectors
    cos = torch.cos(phases)  # [B, N]
    sin = torch.sin(phases)
    # mean resultant vector
    rx = cos.mean(dim=1)  # [B]
    ry = sin.mean(dim=1)
    r_magnitude = torch.sqrt(rx**2 + ry**2 + eps)  # [B]
    return r_magnitude.mean()


def total_loss(
    h_final: torch.Tensor,
    target: torch.Tensor,
    phases: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss + metrics dict.

    Args:
        h_final: [B, d] predicted latent.
        target:  [B, d] stop-grad EMA target.
        phases:  [B, N] final oscillator phases (for diversity loss).
        cfg:     configuration with loss weights.
    """
    l_jepa = latent_prediction_loss(h_final, target)
    l_var = vicreg_variance_loss(h_final, target_std=cfg.vicreg_target_std)
    l_cov = vicreg_covariance_loss(h_final)
    l_phase = phase_diversity_loss(phases)

    total = (
        l_jepa
        + cfg.vicreg_var_weight * l_var
        + cfg.vicreg_cov_weight * l_cov
        + cfg.phase_div_weight * l_phase
    )
    metrics: dict[str, float] = {
        "loss/total": total.detach().float().item(),
        "loss/jepa": l_jepa.detach().float().item(),
        "loss/vicreg_variance": l_var.detach().float().item(),
        "loss/vicreg_covariance": l_cov.detach().float().item(),
        "loss/phase_diversity": l_phase.detach().float().item(),
        "repr/std_mean": h_final.std(dim=0).mean().detach().float().item(),
    }
    return total, metrics
