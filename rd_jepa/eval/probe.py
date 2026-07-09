"""Linear probe training and evaluation for the violation target."""
from __future__ import annotations

import torch
from torch import nn, optim

from .probe_module import ViolationProbe


def train_violation_probe(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_steps: int = 100,
    lr: float = 1e-3,
) -> ViolationProbe:
    """Train a linear probe on top of frozen RD-JEPA latents.

    The probe predicts the grounded collision-force violation_gt from h_K.
    This validates that the learned representations encode collision-relevant
    physical information.

    Args:
        model: Trained RD-JEPA (frozen, eval mode)
        dataloader: DataLoader yielding (context, target, violation_gt)
        device: torch device
        num_steps: Number of optimization steps
        lr: Learning rate for probe

    Returns:
        Trained ViolationProbe
    """
    model.eval()
    d = model.flat_dim
    probe = ViolationProbe(latent_dim=d).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)

    step = 0
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            s_context, _s_target, violation_gt = batch
            s_context = s_context.to(device)
            violation_gt = violation_gt.to(device)

            with torch.no_grad():
                out = model(s_context)
                h_final = out["h_K"]

            pred = probe(h_final)
            loss = torch.nn.functional.mse_loss(pred, violation_gt.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

    return probe


def evaluate_probe(
    model: nn.Module,
    probe: ViolationProbe,
    dataloader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate trained probe on a dataset.

    Returns:
        Dict with MSE, R², mean violation target, mean prediction.
    """
    model.eval()
    probe.eval()

    all_h = []
    all_gt = []

    with torch.no_grad():
        for batch in dataloader:
            s_context, _s_target, violation_gt = batch
            s_context = s_context.to(device)

            out = model(s_context)
            all_h.append(out["h_K"])
            all_gt.append(violation_gt.to(device))

    h = torch.cat(all_h, dim=0)
    violation_gt = torch.cat(all_gt, dim=0)

    return probe.compute_metrics(h, violation_gt)
