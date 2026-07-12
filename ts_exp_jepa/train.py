r"""Training loop for RD-JEPA (Resonant Decomposition JEPA).

Handles:
    - JEPA latent prediction loss + VICReg + phase diversity
    - EMA target encoder updates
    - AMP (bf16) for consumer GPU
    - Async linear forecasting probe evaluation
    - MLflow logging
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import Config
from .data.forecasting import build_dataloaders
from .eval.forecast_probe import (
    ForecastProbe,
    evaluate_forecast_probe,
    train_forecast_probe,
)
from .losses import total_loss
from .models.ts_exp_jepa import RDJEPA
from .viz.mlflow_logger import MLflowLogger


def _get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup → cosine decay."""
    warmup = cfg.lr_warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        if cfg.lr_cosine:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def _collapse_diagnostics(h: torch.Tensor) -> dict[str, float]:
    """Effective rank via SVD entropy and mean cosine similarity."""
    h_centered = h - h.mean(dim=0)
    s = torch.linalg.svdvals(h_centered)
    s_norm = s / (s.sum() + 1e-8)
    entropy = -(s_norm * (s_norm + 1e-8).log()).sum().item()
    eff_rank = torch.exp(torch.tensor(entropy)).item()
    h_norm = nn.functional.normalize(h_centered, dim=1)
    cos_sim = (h_norm @ h_norm.T).mean().item()
    return {
        "repr/effective_rank": eff_rank,
        "repr/mean_cosine_sim": cos_sim,
    }


def train_step(
    model: RDJEPA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    batch: tuple[torch.Tensor, torch.Tensor],
    cfg: Config,
    step: int,
) -> dict[str, float]:
    """Single training step. Returns metrics dict."""
    model.train()
    context, target_x = batch
    device = next(model.parameters()).device
    context = context.to(device)
    target_x = target_x.to(device)

    amp_enabled = cfg.amp_dtype == "bfloat16" and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled)

    with amp_ctx:
        out = model(context, K_steps=cfg.K_steps)
        h_k = out["h_K"]
        phases = out["phases"]

        with torch.no_grad():
            target = model.target(target_x)

        loss, metrics = total_loss(h_k, target, phases, cfg)

    optimizer.zero_grad()
    if scaler is not None and not amp_enabled:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    model.update_ema(step)
    return metrics


@torch.no_grad()
def eval_step(
    model: RDJEPA,
    batch: tuple[torch.Tensor, torch.Tensor],
    cfg: Config,
) -> dict[str, float]:
    """Single eval step. Returns metrics dict."""
    model.eval()
    context, target_x = batch
    device = next(model.parameters()).device
    context = context.to(device)
    target_x = target_x.to(device)

    out = model(context, K_steps=cfg.K_steps)
    h_k = out["h_K"]
    phases = out["phases"]

    target = model.target(target_x)
    _, metrics = total_loss(h_k, target, phases, cfg)
    metrics.update(_collapse_diagnostics(h_k))
    return metrics


def train(cfg: Config, logger: MLflowLogger | None = None) -> None:
    """Full training loop."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = RDJEPA(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.epochs * len(train_loader)
    scheduler = _get_lr_scheduler(optimizer, cfg, total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    probe = ForecastProbe(cfg.latent_dim, cfg.horizon, cfg.n_features)

    if logger is not None:
        logger.init_run()

    cfg.exp_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()

        for batch in train_loader:
            metrics = train_step(model, optimizer, scaler, batch, cfg, step)
            scheduler.step()

            if logger is not None and step % cfg.log_every_n_steps == 0:
                logger.log_metrics(metrics, step, context={"subset": "train"})
                logger.log_metrics(
                    {"train/lr": optimizer.param_groups[0]["lr"]}, step
                )

            step += 1

        epoch_time = time.time() - epoch_start

        val_metrics = _evaluate_loop(model, val_loader, cfg, device)
        if logger is not None:
            logger.log_metrics(val_metrics, epoch, context={"subset": "val"})
            logger.log_metrics({"time/epoch_seconds": epoch_time}, epoch)

        if (epoch + 1) % cfg.eval_every_n_epochs == 0 or epoch == cfg.epochs - 1:
            train_forecast_probe(model, probe, train_loader, cfg, device)
            probe_metrics = evaluate_forecast_probe(
                model, probe, val_loader, cfg, device
            )
            if logger is not None:
                logger.log_metrics(
                    probe_metrics, epoch, context={"subset": "probe_val"}
                )

            ckpt_path = cfg.exp_dir / "ckpt.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "probe": probe.state_dict(),
                    "cfg": cfg.to_dict(),
                    "epoch": epoch,
                    "step": step,
                },
                ckpt_path,
            )

    test_loader = loaders["test"]
    train_forecast_probe(model, probe, train_loader, cfg, device)
    test_probe = evaluate_forecast_probe(model, probe, test_loader, cfg, device)
    test_metrics = _evaluate_loop(model, test_loader, cfg, device)
    if logger is not None:
        logger.log_metrics(test_metrics, 0, context={"subset": "test"})
        logger.log_metrics(test_probe, 0, context={"subset": "probe_test"})
        logger.log_metrics({"probe/test_mse": test_probe["probe/mse"]}, step)

    print(f"Test probe MSE: {test_probe['probe/mse']:.6f}")
    print(f"Test probe MAE: {test_probe['probe/mae']:.6f}")


@torch.no_grad()
def _evaluate_loop(
    model: RDJEPA,
    loader,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run eval over all batches and average."""
    all_metrics: dict[str, list[float]] = {}
    for batch in loader:
        metrics = eval_step(model, batch, cfg)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in all_metrics.items()}
