"""Linear probe training and evaluation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, optim

from .probe_module import SolvedProbe


def train_solved_probe(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_steps: int = 100,
    lr: float = 1e-3,
) -> SolvedProbe:
    """Train a linear probe on top of frozen RD-JEPA latents.

    The probe predicts solved/unsolved from h_K. This validates that
    the learned representations encode task-relevant information.

    Args:
        model: Trained RD-JEPA (frozen, eval mode)
        dataloader: DataLoader yielding (context, action, target, solved)
        device: torch device
        num_steps: Number of optimization steps
        lr: Learning rate for probe

    Returns:
        Trained SolvedProbe
    """
    model.eval()
    d = model.flat_dim
    probe = SolvedProbe(latent_dim=d).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)

    step = 0
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            s_context, action, s_target, solved = batch
            s_context = s_context.to(device)
            action = action.to(device)
            s_target = s_target.to(device)
            solved = solved.to(device)

            with torch.no_grad():
                out = model(s_context, action)
                h_final = out["h_K"]

            logits = probe(h_final)
            loss = F.binary_cross_entropy_with_logits(logits, solved.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

    return probe


def evaluate_probe(
    model: nn.Module,
    probe: SolvedProbe,
    dataloader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate trained probe on a dataset.

    Returns:
        Dict with accuracy, AUROC, solved fraction
    """
    model.eval()
    probe.eval()

    all_h = []
    all_solved = []

    with torch.no_grad():
        for batch in dataloader:
            s_context, action, s_target, solved = batch
            s_context = s_context.to(device)
            action = action.to(device)

            out = model(s_context, action)
            all_h.append(out["h_K"])
            all_solved.append(solved.to(device))

    h = torch.cat(all_h, dim=0)
    solved = torch.cat(all_solved, dim=0)

    return probe.compute_metrics(h, solved)
