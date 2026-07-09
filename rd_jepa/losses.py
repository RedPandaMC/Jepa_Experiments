r"""Loss functions for RD-JEPA.

Implements spec §3.2:
  1. Latent reconstruction loss (JEPA): MSE between h_final and the
     stop-gradient EMA target encoder of s_{t+1}.
  2. Optional contrastive/violation loss: penalize non-monotonic
     violation trajectories (encourages the lens to focus faster).
  3. VICReg-style variance + covariance regularization to prevent
     representation collapse (safety net beyond EMA/stop-grad).
  4. Grounded violation supervision: regress V_psi toward MOVi's
     collision-force ground truth in the lookahead window.

Support the Decision-3 ablation:
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


def violation_supervision_loss(
    violations: torch.Tensor,  # [K, B]
    all_h: torch.Tensor,  # [K, B, d]
    target: torch.Tensor,  # [B, d]
) -> torch.Tensor:
    """Train V_psi to predict its own residual error to the true target.

    The supervision signal is the per-sample, per-step squared distance from
    h_k to the (stop-grad) target. This teaches the violation head to report
    how far the lens still is from being in focus, so the early-exit
    threshold tau is meaningful. No extra labels needed.
    """
    # [K, B] squared error per step
    with torch.no_grad():
        err = (all_h - target.unsqueeze(0)).pow(2).sum(dim=-1)  # [K, B]
        err = err / err.max().clamp(min=1e-6)  # normalize to ~[0,1]
    return torch.nn.functional.mse_loss(violations, err)


def violation_grounded_loss(
    violations: torch.Tensor,  # [K, B]
    violation_gt: torch.Tensor,  # [B] float in [0, 1]
) -> torch.Tensor:
    """Grounded supervision: regress V_psi toward MOVi collision-force target.

    The target is the normalized sum of collision force magnitudes occurring in
    the lookahead window (after s_t), derived from MOVi's per-frame collision
    events during cache conversion. A scene with high collision energy should
    report a high violation; a quiet scene should report near-zero violation.

    We supervise the *final-step* violation (h_K) since the lens has had the
    full deliberation budget by then. Using smooth-L1 keeps the regression
    robust to the heavy-tailed force distribution.
    """
    target = violation_gt.float().clamp(0.0, 1.0)
    final_v = violations[-1]  # [B]
    return torch.nn.functional.smooth_l1_loss(final_v, target)


def vicreg_variance_loss(
    z: torch.Tensor,  # [B, d]
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg variance loss: penalize low standard deviation per dimension.

    Encourages each dimension to have variance > 0 (prevents collapse
    where all samples map to the same point). The hinge loss encourages
    std to stay above target_std.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(torch.relu(target_std - std))


def vicreg_covariance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg covariance loss: penalize off-diagonal covariance.

    Decorrelates dimensions, encouraging diverse representations.
    Computes covariance matrix and penalizes squared off-diagonal entries.
    """
    B, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)  # [d, d]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2)).sum() / d


def total_loss(
    all_h: torch.Tensor,
    h_final: torch.Tensor,
    target: torch.Tensor,
    violations: torch.Tensor,
    cfg: Config,
    violation_gt: torch.Tensor | None = None,
    violation_weight: float = 0.01,
    violation_supervision_weight: float = 0.1,
    violation_grounded_weight: float = 0.1,
    vicreg_var_weight: float = 1.0,
    vicreg_cov_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total RD-JEPA loss + a dict of metric names for logging."""
    l_traj = trajectory_loss(all_h, target, cfg)
    l_viol = violation_aux_loss(violations)
    l_viol_sup = violation_supervision_loss(violations, all_h, target)

    # Grounded supervision using MOVi collision-force target (if provided)
    if violation_gt is not None:
        l_viol_ground = violation_grounded_loss(violations, violation_gt)
    else:
        l_viol_ground = torch.zeros((), device=violations.device, dtype=violations.dtype)

    # VICReg collapse-prevention (applied to final latent)
    l_var = vicreg_variance_loss(h_final, target_std=cfg.vicreg_target_std)
    l_cov = vicreg_covariance_loss(h_final)

    total = (
        l_traj
        + violation_weight * l_viol
        + violation_supervision_weight * l_viol_sup
        + violation_grounded_weight * l_viol_ground
        + vicreg_var_weight * l_var
        + vicreg_cov_weight * l_cov
    )
    metrics = {
        "loss/total": total.detach().float().item(),
        "loss/trajectory": l_traj.detach().float().item(),
        "loss/violation_aux": l_viol.detach().float().item(),
        "loss/violation_supervision": l_viol_sup.detach().float().item(),
        "loss/violation_grounded": l_viol_ground.detach().float().item(),
        "loss/vicreg_variance": l_var.detach().float().item(),
        "loss/vicreg_covariance": l_cov.detach().float().item(),
        "repr/std_mean": h_final.std(dim=0).mean().detach().float().item(),
    }
    return total, metrics
