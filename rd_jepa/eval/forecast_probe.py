r"""Linear forecasting probe for evaluating learned representations.

A simple Linear(latent_dim, horizon * n_features) head trained on
*frozen* h_K representations to predict the future window. This is the
standard JEPA evaluation protocol: if the representation is good, even a
linear probe should forecast well.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW

from ..config import Config


class ForecastProbe(nn.Module):
    """Linear probe: latent → future values."""

    def __init__(self, latent_dim: int, horizon: int, n_features: int):
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features
        self.net = nn.Linear(latent_dim, horizon * n_features)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, d] → [B, H, C]."""
        out = self.net(h)
        return out.reshape(-1, self.horizon, self.n_features)  # type: ignore[no-any-return]

    def compute_loss(
        self, h: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """h: [B, d], target: [B, H, C] → (loss, metrics)."""
        pred = self.forward(h)
        loss = nn.functional.mse_loss(pred, target)
        with torch.no_grad():
            mae = (pred - target).abs().mean().item()
        return loss, {"probe/mse": loss.item(), "probe/mae": mae}


def train_forecast_probe(
    model: nn.Module,
    probe: ForecastProbe,
    train_loader,
    cfg: Config,
    device: torch.device,
) -> None:
    """Train linear probe on frozen h_K for ``cfg.probe_steps`` steps."""
    probe.to(device)
    probe.train()
    opt = AdamW(probe.parameters(), lr=cfg.probe_lr, weight_decay=0.0)

    step = 0
    for batch in train_loader:
        if step >= cfg.probe_steps:
            break
        context, target = batch
        context = context.to(device)
        target = target.to(device)

        with torch.no_grad():
            out = model(context, K_steps=cfg.K_steps)
            h_k = out["h_K"].detach()

        pred = probe(h_k)
        loss = nn.functional.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1


@torch.no_grad()
def evaluate_forecast_probe(
    model: nn.Module,
    probe: ForecastProbe,
    loader,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate probe on a dataset, returning MSE and MAE."""
    probe.to(device)
    probe.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        context, target = batch
        context = context.to(device)
        target = target.to(device)

        out = model(context, K_steps=cfg.K_steps)
        h_k = out["h_K"]
        pred = probe(h_k)

        total_mse += nn.functional.mse_loss(pred, target).item()
        total_mae += (pred - target).abs().mean().item()
        n_batches += 1

    return {
        "probe/mse": total_mse / max(n_batches, 1),
        "probe/mae": total_mae / max(n_batches, 1),
    }
