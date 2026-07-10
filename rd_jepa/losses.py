r"""Loss functions for RD-JEPA v2.

Spec §3.2 plus three core fixes:
  1. Latent prediction loss (JEPA): MSE between h_K and the stop-gradient
     EMA target encoder of s_{t+1}. Final-only (no discounted trajectory).
  2. Violation losses: aux monotonicity + self-supervision + grounded
     collision-force regression.
  3. VICReg variance + covariance regularization (collapse safety net).
  4. Energy conservation: penalize latent magnitude drift across the K loop
     (physically forbids the subtractive phase from zeroing the state).
  5. Contrastive dynamics: margin loss penalizing stasis (h_K ≈ h_0) when a
     physical push (violation_gt > 0) was present.
  6. Divergence regularization: penalize per-step latent mass change across
     the K trajectory (constant-density / incompressibility proxy).
  7. Lens-bank routing aux: load-balance (uniform usage) + entropy bonus
     (non-degenerate routing). Only active when n_lenses > 1.
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


def energy_conservation_loss(all_h: torch.Tensor) -> torch.Tensor:
    r"""Latent energy conservation: $| \|h_K\|_2 - \|h_0\|_2 |^2$.

    Physically forbids the subtractive (divergence projection) phase from
    zeroing out the latent state. If it subtracts overlapping momentum, it
    must route that momentum somewhere else. Energy (magnitude) must be
    conserved across the deliberation loop. Averaged over the batch.
    """
    h0 = all_h[0]  # [B, d]
    hK = all_h[-1]  # [B, d]
    n0 = torch.norm(h0, p=2, dim=-1)  # [B]
    nK = torch.norm(hK, p=2, dim=-1)  # [B]
    return ((nK - n0) ** 2).mean()


def contrastive_dynamics_loss(
    all_h: torch.Tensor,
    violation_gt: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    r"""Margin loss penalizing stasis when a physical push was present.

    MOVi has no action modality, so "must make forward progress" is gated by
    the grounded collision signal: when `violation_gt > 0` (a physical push
    occurred in the lookahead window), the latent must move by at least
    `margin` in L2 norm from h_0 to h_K. Scenes with no collision energy
    are free to stay near h_0 (a quiet scene genuinely should not move).

    L = mean_b [ ReLU(margin - ||h_K - h_0||_2) * 1[violation_gt_b > 0] ]
    """
    h0 = all_h[0]  # [B, d]
    hK = all_h[-1]  # [B, d]
    delta_norm = torch.norm(hK - h0, p=2, dim=-1)  # [B]
    push = (violation_gt > 0.0).float()  # [B]
    return (torch.relu(margin - delta_norm) * push).mean()


def divergence_reg_loss(all_h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Constant-density / incompressibility proxy across the K trajectory.

    Penalizes change in latent mass (L2 norm) between consecutive steps.
    Combined with the divergence projection in the lens, this encourages the
    latent to behave like an incompressible fluid: mass may redistribute but
    not be created or destroyed. Averaged over the K-1 inter-step transitions
    and the batch.
    """
    if all_h.shape[0] < 2:
        return torch.zeros((), device=all_h.device, dtype=all_h.dtype)
    norms = torch.norm(all_h, p=2, dim=-1)  # [K, B]
    diffs = (norms[1:] - norms[:-1]).abs()  # [K-1, B]
    return diffs.mean()


def load_balance_loss(gates: torch.Tensor | None) -> torch.Tensor:
    r"""MoE load-balance loss: penalize non-uniform router usage.

    $N \cdot \sum_i \bar{g}_i^{\,2}$ where $\bar{g}_i$ is the mean usage of
    lens $i$ across the batch and K steps. Encourages all lenses to receive
    roughly equal routing mass on average. Returns 0 when ``gates`` is None
    (single-lens path, no router).
    """
    if gates is None:
        return torch.zeros((), device=torch.device("cpu"))
    # gates: [K, B, N]
    n = gates.shape[-1]
    usage = gates.mean(dim=(0, 1))  # [N]
    return n * (usage.pow(2)).sum()


def router_entropy_loss(
    gates: torch.Tensor | None, eps: float = 1e-8
) -> torch.Tensor:
    r"""Mean router entropy (to be maximized by the composer).

    Returns $-\mathrm{mean}_{k,b}\sum_i g_i \log g_i$ — the mean Shannon
    entropy of the per-step gate distribution. The composer subtracts a
    weight times this value from the total, so minimizing the total loss
    maximizes routing entropy (encouraging non-degenerate, spread routing).
    Returns 0 when ``gates`` is None (single-lens path).
    """
    if gates is None:
        return torch.zeros((), device=torch.device("cpu"))
    # gates: [K, B, N]
    ent = -(gates * (gates + eps).log()).sum(dim=-1)  # [K, B]
    return ent.mean()


def total_loss(
    all_h: torch.Tensor,
    h_final: torch.Tensor,
    target: torch.Tensor,
    violations: torch.Tensor,
    cfg: Config,
    violation_gt: torch.Tensor | None = None,
    gates: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total RD-JEPA v2 loss + a dict of metric names for logging.

    All loss weights are read from `cfg` (no hard-coded defaults here).
    """
    # JEPA core loss: final-only (no discounted trajectory in v2).
    l_jepa = latent_prediction_loss(h_final, target)

    # Violation losses.
    l_viol = violation_aux_loss(violations)
    l_viol_sup = violation_supervision_loss(violations, all_h, target)
    if violation_gt is not None:
        l_viol_ground = violation_grounded_loss(violations, violation_gt)
    else:
        l_viol_ground = torch.zeros(
            (), device=violations.device, dtype=violations.dtype
        )

    # VICReg collapse-prevention (applied to final latent).
    l_var = vicreg_variance_loss(h_final, target_std=cfg.vicreg_target_std)
    l_cov = vicreg_covariance_loss(h_final)

    # v2 core fixes: energy conservation + contrastive dynamics + divergence.
    l_energy = energy_conservation_loss(all_h)
    if violation_gt is not None:
        l_contrastive = contrastive_dynamics_loss(
            all_h, violation_gt, margin=cfg.contrastive_margin
        )
    else:
        l_contrastive = torch.zeros(
            (), device=all_h.device, dtype=all_h.dtype
        )
    l_div = divergence_reg_loss(all_h)

    # Lens-bank routing aux losses (only when n_lenses > 1).
    l_lb = load_balance_loss(gates)
    l_ent = router_entropy_loss(gates)

    total = (
        l_jepa
        + cfg.violation_weight * l_viol
        + cfg.violation_supervision_weight * l_viol_sup
        + cfg.violation_grounded_weight * l_viol_ground
        + cfg.vicreg_var_weight * l_var
        + cfg.vicreg_cov_weight * l_cov
        + cfg.energy_weight * l_energy
        + cfg.contrastive_weight * l_contrastive
        + cfg.divergence_reg_weight * l_div
        + cfg.load_balance_weight * l_lb
        - cfg.router_entropy_weight * l_ent
    )
    metrics: dict[str, float] = {
        "loss/total": total.detach().float().item(),
        "loss/jepa": l_jepa.detach().float().item(),
        "loss/violation_aux": l_viol.detach().float().item(),
        "loss/violation_supervision": l_viol_sup.detach().float().item(),
        "loss/violation_grounded": l_viol_ground.detach().float().item(),
        "loss/vicreg_variance": l_var.detach().float().item(),
        "loss/vicreg_covariance": l_cov.detach().float().item(),
        "loss/energy": l_energy.detach().float().item(),
        "loss/contrastive": l_contrastive.detach().float().item(),
        "loss/divergence_reg": l_div.detach().float().item(),
        "loss/load_balance": l_lb.detach().float().item(),
        "loss/router_entropy": l_ent.detach().float().item(),
        "repr/std_mean": h_final.std(dim=0).mean().detach().float().item(),
    }
    # Per-lens mean usage for monitoring specialization.
    if gates is not None:
        usage = gates.mean(dim=(0, 1)).detach().float()  # [N]
        for i in range(usage.shape[0]):
            metrics[f"router/lens_{i}_usage"] = usage[i].item()
    return total, metrics
