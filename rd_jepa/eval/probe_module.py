"""Evaluation utilities: linear probe and AUCCESS metric.

These connect the RD-JEPA latent space to actual PhyRE task performance,
providing the ground-truth validation that the world model is learning
useful physical representations.
"""
from __future__ import annotations

import torch
from torch import nn


class SolvedProbe(nn.Module):
    """Linear probe: predicts solved/unsolved from the final latent h_K.

    This is a simple downstream task that validates whether the learned
    representations capture task-relevant physical information. If the
    probe can predict solve status, the world model is learning something
    meaningful about physical outcomes.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, d] -> [B] logit."""
        return self.fc(h).squeeze(-1)

    def compute_metrics(
        self, h: torch.Tensor, solved: torch.Tensor
    ) -> dict[str, float]:
        """Compute accuracy and AUROC given latents and ground-truth labels.

        Args:
            h: [B, d] final latent from RD-JEPA
            solved: [B] bool tensor, True if action solved the task

        Returns:
            Dict with accuracy, auroc, etc.
        """
        with torch.no_grad():
            logits = self.forward(h)
            probs = torch.sigmoid(logits)
            labels = solved.float()

            # Accuracy at threshold 0.5
            preds = (probs > 0.5).float()
            acc = (preds == labels).float().mean().item()

            # Simple AUROC approximation (Mann-Whitney U statistic)
            # Sort by predicted probability and compute rank correlation
            if labels.sum() > 0 and (1 - labels).sum() > 0:
                # Separate solved and unsolved scores
                solved_scores = probs[labels == 1]
                unsolved_scores = probs[labels == 0]

                # Compute AUROC via U statistic
                # For each solved-unsolved pair, count if solved_score > unsolved_score
                n_solved = solved_scores.shape[0]
                n_unsolved = unsolved_scores.shape[0]

                # Broadcast comparison
                comparisons = (
                    solved_scores.unsqueeze(1) > unsolved_scores.unsqueeze(0)
                ).float()
                # Add 0.5 for ties (not expected with float logits but safe)
                ties = (
                    solved_scores.unsqueeze(1) == unsolved_scores.unsqueeze(0)
                ).float() * 0.5
                u_stat = (comparisons + ties).sum()

                auroc = (u_stat / (n_solved * n_unsolved)).item()
            else:
                auroc = 0.5  # undefined if all same label

            return {
                "probe/accuracy": acc,
                "probe/auroc": auroc,
                "probe/solved_fraction": labels.mean().item(),
            }


def compute_auccess(
    scores: torch.Tensor,
    solved: torch.Tensor,
) -> float:
    """Compute PhyRE AUCCESS (Area Under Cumulative Success Curve).

    AUCCESS measures ranking quality: given candidate actions for a task,
    can we rank them so solved actions appear early? Higher scores indicate
    better physical reasoning (the model knows which actions will work).

    The metric is the area under the cumulative success curve:
    - Sort actions by predicted score (descending)
    - Compute cumulative success rate at each position
    - AUCCESS = mean(cumulative_success_rate)

    Perfect ranking (all solved first) -> AUCCESS = 1.0
    Random ranking -> AUCCESS ≈ solved_fraction
    Worst ranking (all unsolved first) -> AUCCESS = solved_fraction / N

    Args:
        scores: [N] predicted solve probabilities (higher = more likely to solve)
        solved: [N] bool tensor, True if action actually solves the task

    Returns:
        AUCCESS score in [0, 1]
    """
    with torch.no_grad():
        if solved.sum() == 0:
            return 0.0  # No solved actions to rank

        # Sort by score descending (best actions first)
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_solved = solved[sorted_indices].float()

        # Cumulative sum of solved actions
        cumulative = torch.cumsum(sorted_solved, dim=0)

        # Cumulative success rate at each position
        positions = torch.arange(1, len(solved) + 1, device=solved.device, dtype=torch.float32)
        cumulative_rate = cumulative / positions

        # AUCCESS is the area under this curve
        auccess = cumulative_rate.mean().item()

        return auccess


def compute_auccess_per_task(
    model: nn.Module,
    probe: SolvedProbe,
    task_actions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Compute AUCCESS for multiple tasks.

    Args:
        model: Trained RD-JEPA
        probe: Trained SolvedProbe
        task_actions: Dict mapping task_id -> (contexts [N,2,H,W], actions [N,3])
        device: torch device

    Returns:
        Dict with per-task AUCCESS and mean AUCCESS
    """
    model.eval()
    probe.eval()

    auccess_scores = {}

    with torch.no_grad():
        for task_id, (contexts, actions) in task_actions.items():
            contexts = contexts.to(device)
            actions = actions.to(device)

            # Get model predictions
            out = model(contexts, actions)
            h_final = out["h_K"]

            # Get solve probabilities from probe
            logits = probe(h_final)
            _ = torch.sigmoid(logits)  # Would be used with ground-truth solved flags

            # This is a simplified version - in practice, you'd simulate
            # the actions in PhyRE to get ground-truth solved flags
            # For now, assume solved is provided alongside contexts/actions
            # in the task_actions dict
            raise NotImplementedError(
                "Full AUCCESS requires PhyRE simulator to evaluate "
                "ground-truth solve status for each action. "
                "Pass solved flags alongside contexts/actions, or use "
                "compute_auccess() directly with pre-computed scores and solved flags."
            )

    return auccess_scores
