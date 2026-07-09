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
from .viz.decoder import VizDecoder


def _collapse_diagnostics(z: torch.Tensor) -> dict[str, float]:
    """Compute representation quality diagnostics to detect collapse.

    Args:
        z: [B, d] latent embeddings (already detached, on CPU or GPU).

    Returns:
        Dict with effective rank, mean pairwise cosine similarity, etc.
    """
    with torch.no_grad():
        B, d = z.shape
        # Center the embeddings
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Effective rank: measure of dimensionality actually used
        # Use SVD on covariance for numerical stability
        try:
            cov = (z_centered.T @ z_centered) / (B - 1)  # [d, d]
            eigvals = torch.linalg.eigvalsh(cov)  # sorted ascending
            # Normalize to probability distribution
            eigvals = eigvals.clamp(min=0)
            eig_sum = eigvals.sum()
            if eig_sum > 1e-8:
                p = eigvals / eig_sum
                # Shannon entropy of eigenvalue distribution
                entropy = -(p * (p + 1e-10).log()).sum()
                eff_rank = entropy.exp().item()
            else:
                eff_rank = 0.0
        except Exception:
            eff_rank = 0.0

        # Mean pairwise cosine similarity (high = collapsed)
        z_norm = torch.nn.functional.normalize(z, dim=1)
        cos_sim = z_norm @ z_norm.T  # [B, B]
        # Exclude diagonal (self-similarity = 1)
        mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        mean_cos_sim = cos_sim[mask].mean().item()

        return {
            "repr/effective_rank": eff_rank,
            "repr/mean_cosine_similarity": mean_cos_sim,
            "repr/std_mean": z.std(dim=0).mean().item(),
        }


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
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
    step: int,
    decoder: VizDecoder | None = None,
    decoder_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    # v2 batch: (context[2,H,W], action[3], target[2,H,W], solved[bool])
    s_context, action, s_target, solved = batch
    s_context = s_context.cuda(non_blocking=True)
    action = action.cuda(non_blocking=True)
    s_target = s_target.cuda(non_blocking=True)
    solved = solved.cuda(non_blocking=True)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    with autocast("cuda", dtype=amp_dtype):
        out = model(
            s_context,
            action,
            K=cfg.K,
            early_exit=False,  # train on full K for stable loss
            use_checkpoint=cfg.grad_checkpoint,
        )
        target = model.target(s_target)
        loss, metrics = total_loss(
            out["all_h"], out["h_K"], target, out["violations"], cfg, solved=solved
        )

        # Train viz decoder if provided (detached, doesn't affect JEPA)
        if decoder is not None and decoder_optimizer is not None:
            # Use first channel of context (s_t) as reconstruction target
            s_t = s_context[:, 0:1, :, :]  # [B, 1, H, W]
            dec_loss = decoder.decoder_loss(out["h_K"], s_t)
            metrics["loss/viz_decoder"] = dec_loss.detach().float().item()
            loss = loss + dec_loss  # Add to total loss for backward

    optimizer.zero_grad(set_to_none=True)
    if decoder_optimizer is not None:
        decoder_optimizer.zero_grad(set_to_none=True)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if decoder_optimizer is not None:
        scaler.unscale_(decoder_optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if decoder is not None:
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    scaler.step(optimizer)
    if decoder_optimizer is not None:
        scaler.step(decoder_optimizer)
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
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
) -> dict[str, float]:
    # v2 batch: (context[2,H,W], action[3], target[2,H,W], solved[bool])
    s_context, action, s_target, solved = batch
    s_context = s_context.cuda(non_blocking=True)
    action = action.cuda(non_blocking=True)
    s_target = s_target.cuda(non_blocking=True)
    solved = solved.cuda(non_blocking=True)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    with autocast("cuda", dtype=amp_dtype):
        out = model(
            s_context,
            action,
            K=cfg.K,
            early_exit=cfg.early_exit,
            tau=cfg.violation_tau,
            use_checkpoint=False,
        )
        target = model.target(s_target)
        loss, metrics = total_loss(
            out["all_h"], out["h_K"], target, out["violations"], cfg, solved=solved
        )
    # rename metrics to eval/ prefix
    metrics = {k.replace("loss/", "eval/loss/"): v for k, v in metrics.items()}
    metrics["eval/k_used_mean"] = out["k_used"].float().mean().item()

    # Collapse diagnostics on the final latent
    diag = _collapse_diagnostics(out["h_K"].detach())
    metrics.update({k.replace("repr/", "eval/repr/"): v for k, v in diag.items()})
    return metrics


def _get_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create LR scheduler with linear warmup + optional cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < cfg.lr_warmup_steps:
            # Linear warmup
            return (step + 1) / cfg.lr_warmup_steps
        if not cfg.lr_cosine:
            return 1.0
        # Cosine decay after warmup
        progress = (step - cfg.lr_warmup_steps) / max(1, total_steps - cfg.lr_warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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

    # Viz decoder for visualization (optional but recommended)
    decoder = VizDecoder(latent_dim=cfg.latent_total_dim).cuda()
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.lr)

    # Estimate total steps for LR scheduling
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = _get_lr_scheduler(optimizer, cfg, total_steps)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        decoder.train()
        for batch in loaders["train"]:
            metrics = train_step(
                model, optimizer, scaler, batch, cfg, step,
                decoder=decoder, decoder_optimizer=decoder_optimizer
            )
            scheduler.step()
            metrics["train/lr"] = optimizer.param_groups[0]["lr"]
            logger.log_metrics(metrics, step=step, context={"subset": "train"})
            if step % 20 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss={metrics['loss/total']:.4f} "
                    f"lr={metrics['train/lr']:.2e} "
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

        # Train and evaluate linear probe on solved/unsolved
        try:
            from .eval.probe import evaluate_probe, train_solved_probe
            probe = train_solved_probe(
                model, loaders["dev"], device=torch.device("cuda"), num_steps=50
            )
            probe_metrics = evaluate_probe(
                model, probe, loaders["dev"], device=torch.device("cuda")
            )
            eval_metrics.update({f"eval/{k}": v for k, v in probe_metrics.items()})
        except Exception as e:
            print(f"Probe eval skipped: {e}")

        logger.log_metrics(eval_metrics, step=step, context={"subset": "eval"})
        print(
            f"epoch {epoch} eval: "
            f"loss={eval_metrics.get('eval/loss/total', 0):.4f} "
            f"k_used={eval_metrics.get('eval/k_used_mean', 0):.1f} "
            f"probe_acc={eval_metrics.get('eval/probe/accuracy', 0):.3f}"
        )

        # checkpoint (model + decoder)
        ckpt = Path(cfg.exp_dir) / "ckpt.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "decoder": decoder.state_dict(),
            "cfg": cfg.to_dict(),
            "epoch": epoch
        }, ckpt)

    logger.close()
