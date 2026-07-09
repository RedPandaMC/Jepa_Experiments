r"""Loss functions for RD-JEPA.

Implements spec §3.2:
  1. Latent reconstruction loss (JEPA): MSE between h_final and the
     stop-gradient EMA target encoder of s_{t+1}.
  2. Optional contrastive/violation loss: penalize non-monotonic
     violation trajectories (encourages the lens to focus faster).

Supports the Decision-3 ablation:
  - 'final'     : loss only on h_K.
  - 'discounted': discounted loss over all intermediate h_k with gamma.
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


def trajectory_loss(
    all_h: torch.Tensor,  # [K, B, d]
    target: torch.Tensor,  # [B, d]
    cfg: Config,
) -> torch.Tensor:
    """Loss over the K-step trajectory per the configured strategy.

    'final'      -> loss on the last step only.
    'discounted' -> sum_k gamma^{K-1-k} * loss(h_k) (later steps weighted more).
    """
    K = all_h.shape[0]
    if cfg.loss_trajectory.value == "final":
        return latent_prediction_loss(all_h[-1], target)
    # discounted over all steps
    losses = torch.stack(
        [latent_prediction_loss(all_h[k], target) for k in range(K)]
    )  # [K]
    weights = torch.tensor(
        [cfg.gamma ** (K - 1 - k) for k in range(K)],
        device=all_h.device,
        dtype=losses.dtype,
    )
    return (losses * weights).sum() / weights.sum()


def violation_aux_loss(
    violations: torch.Tensor,  # [K, B]
) -> torch.Tensor:
    """Encourage monotonically decreasing violation scores across the loop.

    Penalizes steps where the violation goes UP relative to the previous
    step. This pushes the lens to keep refining (focusing) rather than
    oscillating. A small auxiliary term.
    """
    if violations.shape[0] < 2:
        return torch.zeros((), device=violations.device, dtype=violations.dtype)
    diffs = violations[1:] - violations[:-1]  # [K-1, B]
    return torch.relu(diffs).mean()


def total_loss(
    all_h: torch.Tensor,
    h_final: torch.Tensor,
    target: torch.Tensor,
    violations: torch.Tensor,
    cfg: Config,
    violation_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total RD-JEPA loss + a dict of metric names for logging."""
    l_traj = trajectory_loss(all_h, target, cfg)
    l_viol = violation_aux_loss(violations)
    total = l_traj + violation_weight * l_viol
    metrics = {
        "loss/total": total.detach().float().item(),
        "loss/trajectory": l_traj.detach().float().item(),
        "loss/violation_aux": l_viol.detach().float().item(),
    }
    return total, metrics
