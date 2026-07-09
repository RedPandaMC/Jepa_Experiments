"""Training loop for RD-JEPA.

Implements spec §3.1 VRAM survival:
  - gradient checkpointing inside the K-loop (handled in the model)
  - bf16 autocast (Ampere-native)
  - truncated BPTT: detach h every `tbptt_n` steps (handled here via the
    model's loop, since we run the whole K-loop per batch; for the POC we
    keep K<=15 which fits in 5.5GB with checkpointing, and document the
    TBPTT hook for future K>15 runs).
  - set_per_process_memory_fraction guard to avoid OOM-killing Windows.

Logs scalars and a VRAM-budget table to Aim.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from .config import Config
from .data.loader import build_dataloaders
from .losses import total_loss
from .models.rd_jepa import RDJEPA
from .viz.aim_logger import AimLogger


def vram_budget_table(cfg: Config) -> str:
    """Print the VRAM budget so the user sees headroom before training starts."""
    if not torch.cuda.is_available():
        return "CUDA not available — running on CPU."
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    reserved = total * cfg.vram_fraction
    lines = [
        "VRAM budget (RTX 3070):",
        f"  total GPU memory   : {total:.2f} GB",
        f"  reserved fraction   : {cfg.vram_fraction} -> {reserved:.2f} GB cap",
        f"  batch size          : {cfg.batch_size}",
        f"  K (deliberation)    : {cfg.K}",
        f"  AMP dtype           : {cfg.amp_dtype}",
        f"  grad checkpoint     : {cfg.grad_checkpoint}",
    ]
    return "\n".join(lines)


def train_step(
    model: RDJEPA,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
    step: int,
) -> dict[str, float]:
    s_t, action, s_tp1 = batch
    s_t = s_t.cuda(non_blocking=True)
    action = action.cuda(non_blocking=True)
    s_tp1 = s_tp1.cuda(non_blocking=True)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    with autocast("cuda", dtype=amp_dtype):
        out = model(
            s_t,
            action,
            K=cfg.K,
            early_exit=False,  # train on full K for stable loss
            use_checkpoint=cfg.grad_checkpoint,
        )
        target = model.target(s_tp1)
        loss, metrics = total_loss(
            out["all_h"], out["h_K"], target, out["violations"], cfg
        )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # EMA target update
    model.target_encoder.update_ema(model.encoder, step=step, warmup=cfg.ema_warmup)

    metrics["train/k_used_mean"] = out["k_used"].float().mean().item()
    if torch.cuda.is_available():
        metrics["train/vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return metrics


@torch.no_grad()
def eval_step(
    model: RDJEPA,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
) -> dict[str, float]:
    s_t, action, s_tp1 = batch
    s_t = s_t.cuda(non_blocking=True)
    action = action.cuda(non_blocking=True)
    s_tp1 = s_tp1.cuda(non_blocking=True)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    with autocast("cuda", dtype=amp_dtype):
        out = model(
            s_t,
            action,
            K=cfg.K,
            early_exit=cfg.early_exit,
            tau=cfg.violation_tau,
            use_checkpoint=False,
        )
        target = model.target(s_tp1)
        loss, metrics = total_loss(
            out["all_h"], out["h_K"], target, out["violations"], cfg
        )
    # rename metrics to eval/ prefix
    metrics = {k.replace("loss/", "eval/loss/"): v for k, v in metrics.items()}
    metrics["eval/k_used_mean"] = out["k_used"].float().mean().item()
    return metrics


def train(cfg: Config, logger: AimLogger | None = None) -> None:
    if logger is None:
        logger = AimLogger(cfg)
    logger.init_run()

    print(vram_budget_table(cfg))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(cfg.vram_fraction, 0)

    torch.manual_seed(cfg.seed)
    loaders = build_dataloaders(cfg)
    model = RDJEPA(cfg).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler("cuda", enabled=(cfg.amp_dtype == "float16"))

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for batch in loaders["train"]:
            metrics = train_step(model, optimizer, scaler, batch, cfg, step)
            logger.log_metrics(metrics, step=step, context={"subset": "train"})
            if step % 20 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss={metrics['loss/total']:.4f} "
                    f"vram={metrics.get('train/vram_gb', 0):.2f}GB"
                )
            step += 1

        # eval
        model.eval()
        eval_metrics: dict[str, float] = {}
        n = 0
        for batch in loaders["dev"]:
            m = eval_step(model, batch, cfg)
            for k, v in m.items():
                eval_metrics[k] = eval_metrics.get(k, 0.0) + v
            n += 1
        eval_metrics = {k: v / n for k, v in eval_metrics.items()}
        logger.log_metrics(eval_metrics, step=step, context={"subset": "eval"})
        print(
            f"epoch {epoch} eval: "
            f"loss={eval_metrics.get('eval/loss/total', 0):.4f} "
            f"k_used={eval_metrics.get('eval/k_used_mean', 0):.1f}"
        )

        # checkpoint
        ckpt = Path(cfg.exp_dir) / "ckpt.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch}, ckpt)

    logger.close()
