"""Training loop for RD-JEPA v2.

Implements spec §3.1 VRAM survival:
  - gradient checkpointing inside the K-loop (handled in the model)
  - bf16 autocast (Ampere-native)
  - set_per_process_memory_fraction guard to avoid OOM-killing Windows.

v2 core fixes wired in here:
  - Curriculum K: per-epoch K_min -> K_max schedule (cfg.resolve_K(epoch)).
    Both train_step and eval_step use the epoch's K (K_epoch).
  - Asynchronous probing decoder: trained in its own optimizer + backward
    pass on a detached h_K, every cfg.decoder_interval JEPA steps. Zero
    gradient entanglement with the JEPA loop.

Logs scalars and a VRAM-budget table to Aim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast

from .config import Config
from .data.loader import build_dataloaders
from .losses import total_loss
from .models.rd_jepa import RDJEPA
from .viz.aim_logger import AimLogger
from .viz.decoder import VizDecoder, make_decoder_optimizer
from .viz.gif_writer import render_rollout_for_eval


def _should_render_gifs(epoch: int, total_epochs: int, every_n_epochs: int = 5) -> bool:
    """Render gifs only on a sparse cadence to keep training overhead low."""
    if every_n_epochs <= 0:
        return False
    return epoch % every_n_epochs == 0 or epoch == total_epochs - 1


def _should_eval(epoch: int, total_epochs: int) -> bool:
    """Validation cadence scaled by total epochs.

    Evaluates every epoch for short runs (<= 20 epochs). For longer runs,
    evaluates roughly every 5% of total epochs but always on the final
    epoch. This keeps eval time proportional: a 100-epoch run evaluates
    ~21 times, a 20-epoch run evaluates every epoch.
    """
    if epoch == total_epochs - 1:
        return True
    if total_epochs <= 20:
        return True
    interval = max(1, total_epochs // 20)
    return epoch % interval == 0


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


def vram_budget_table(cfg: Config, k_epoch: int) -> str:
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
        f"  K_max (deliberation): {cfg.K_max}  (this epoch K={k_epoch})",
        f"  n_lenses (lens bank): {cfg.n_lenses}",
        f"  AMP dtype           : {cfg.amp_dtype}",
        f"  grad checkpoint     : {cfg.grad_checkpoint}",
        f"  decoder interval    : every {cfg.decoder_interval} JEPA steps",
    ]
    return "\n".join(lines)


def train_step(
    model: RDJEPA,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
    step: int,
    k_epoch: int,
) -> dict[str, Any]:
    """One JEPA training step (no decoder — that runs asynchronously).

    Args:
        k_epoch: the curriculum-resolved K for this epoch.
    """
    # v3 batch: (context[2*C,H,W], target[2*C,H,W], violation_gt[1])
    s_context, s_target, violation_gt = batch
    s_context = s_context.cuda(non_blocking=True)
    s_target = s_target.cuda(non_blocking=True)
    violation_gt = violation_gt.cuda(non_blocking=True)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    with autocast("cuda", dtype=amp_dtype):
        out = model(
            s_context,
            K=k_epoch,
            early_exit=False,  # train on full K for stable loss
            use_checkpoint=cfg.grad_checkpoint,
        )
        target = model.target(s_target)
        loss, metrics = total_loss(
            out["all_h"],
            out["h_K"],
            target,
            out["violations"],
            cfg,
            violation_gt=violation_gt,
            gates=out["gates"],
        )
    # Widen to dict[str, Any] so we can stash transient tensors for the
    # asynchronous decoder step without violating the loss's float contract.
    m: dict[str, Any] = {**metrics}

    # JEPA-only backward. The decoder is trained separately in
    # train_decoder_step on a detached h_K with its own optimizer/backward.
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # EMA target update
    model.target_encoder.update_ema(model.encoder, step=step, warmup=cfg.ema_warmup)

    m["train/k_used_mean"] = out["k_used"].float().mean().item()
    m["train/K_epoch"] = float(k_epoch)
    if torch.cuda.is_available():
        m["train/vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    # Expose h_K (detached, fp32) for the async decoder step (caller drains it).
    m["_h_K_for_decoder"] = out["h_K"].detach().float()
    m["_s_context_for_decoder"] = s_context.detach()
    return m


def train_decoder_step(
    decoder: VizDecoder,
    decoder_optimizer: torch.optim.Optimizer,
    h_k: torch.Tensor,
    s_context: torch.Tensor,
    cfg: Config,
) -> dict[str, float]:
    """One asynchronous decoder training step on a detached h_K.

    Runs in its own backward pass with its own optimizer — zero gradient
    entanglement with the JEPA loop. The decoder learns to reconstruct s_t
    (the middle frame of the context stack) from the frozen latent.
    """
    # s_t is the second frame of the context stack: channels [C:2C].
    C = cfg.img_channels
    s_t = s_context[:, C : 2 * C, :, :].float()  # [B, C, H, W]

    # The VizDecoder always outputs 64x64 (4 conv-transpose blocks from 4x4).
    # Downsample the target to match so the MSE is well-defined regardless of
    # the training frame_size (64, 128, 256, ...).
    if s_t.shape[-1] != 64:
        s_t = torch.nn.functional.interpolate(
            s_t, size=(64, 64), mode="bilinear", align_corners=False
        )

    # Decoder runs in fp32 (conv-transpose + autocast dtype edge case).
    pred = decoder(h_k.detach())
    loss = torch.nn.functional.mse_loss(pred, s_t)

    decoder_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    decoder_optimizer.step()

    # PSNR for logging (higher = better reconstruction).
    with torch.no_grad():
        mse = loss.item()
        psnr = float("inf") if mse <= 1e-12 else 10.0 * torch.log10(
            torch.tensor(1.0) / torch.tensor(mse)
        ).item()
    return {
        "decoder/loss": loss.detach().float().item(),
        "decoder/psnr": psnr,
    }


@torch.no_grad()
def eval_step(
    model: RDJEPA,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
    k_epoch: int,
) -> dict[str, float]:
    """One eval step. Uses the curriculum-resolved K (same as training)."""
    # v3 batch: (context[2*C,H,W], target[2*C,H,W], violation_gt[1])
    s_context, s_target, violation_gt = batch
    s_context = s_context.cuda(non_blocking=True)
    s_target = s_target.cuda(non_blocking=True)
    violation_gt = violation_gt.cuda(non_blocking=True)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    with autocast("cuda", dtype=amp_dtype):
        out = model(
            s_context,
            K=k_epoch,
            early_exit=cfg.early_exit,
            tau=cfg.violation_tau,
            use_checkpoint=False,
        )
        target = model.target(s_target)
        loss, metrics = total_loss(
            out["all_h"],
            out["h_K"],
            target,
            out["violations"],
            cfg,
            violation_gt=violation_gt,
            gates=out["gates"],
        )
    # rename metrics to eval/ prefix
    metrics = {k.replace("loss/", "eval/loss/"): v for k, v in metrics.items()}
    metrics["eval/k_used_mean"] = out["k_used"].float().mean().item()
    metrics["eval/K_epoch"] = float(k_epoch)

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

    k_epoch = cfg.resolve_K(0)
    print(vram_budget_table(cfg, k_epoch))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(cfg.vram_fraction, 0)

    torch.manual_seed(cfg.seed)
    loaders = build_dataloaders(cfg)
    model = RDJEPA(cfg).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler("cuda", enabled=(cfg.amp_dtype == "float16"))

    # Asynchronous probing decoder: dedicated optimizer, trained in its own
    # backward pass on a detached h_K every cfg.decoder_interval JEPA steps.
    decoder = VizDecoder(
        latent_dim=cfg.latent_total_dim, out_channels=cfg.img_channels
    ).cuda()
    decoder_optimizer = make_decoder_optimizer(decoder, cfg)

    # Estimate total steps for LR scheduling
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = _get_lr_scheduler(optimizer, cfg, total_steps)

    step = 0
    for epoch in range(cfg.epochs):
        k_epoch = cfg.resolve_K(epoch)
        model.train()
        decoder.train()
        for batch in loaders["train"]:
            metrics = train_step(
                model, optimizer, scaler, batch, cfg, step, k_epoch=k_epoch
            )
            # Asynchronous decoder step: own backward pass, own cadence.
            if step % cfg.decoder_interval == 0:
                dec_metrics = train_decoder_step(
                    decoder,
                    decoder_optimizer,
                    metrics["_h_K_for_decoder"],
                    metrics["_s_context_for_decoder"],
                    cfg,
                )
                metrics.update(dec_metrics)
            # Drop the transient tensors from metrics before logging.
            metrics.pop("_h_K_for_decoder", None)
            metrics.pop("_s_context_for_decoder", None)

            scheduler.step()
            metrics["train/lr"] = optimizer.param_groups[0]["lr"]
            logger.log_metrics(metrics, step=step, context={"subset": "train"})
            if step % 20 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss={metrics['loss/total']:.4f} "
                    f"lr={metrics['train/lr']:.2e} "
                    f"K={metrics['train/K_epoch']:.0f} "
                    f"vram={metrics.get('train/vram_gb', 0):.2f}GB"
                )
            step += 1

        # eval (uses the same curriculum K_epoch as training).
        # Validation cadence scales with total epochs (see _should_eval).
        if not _should_eval(epoch, cfg.epochs):
            # Still checkpoint so training can resume.
            ckpt = Path(cfg.exp_dir) / "ckpt.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "cfg": cfg.to_dict(),
                "epoch": epoch,
            }, ckpt)
            step += 0  # step unchanged; no eval this epoch
            continue

        model.eval()
        eval_metrics: dict[str, float] = {}
        n = 0
        for batch in loaders["dev"]:
            m = eval_step(model, batch, cfg, k_epoch=k_epoch)
            for k, v in m.items():
                eval_metrics[k] = eval_metrics.get(k, 0.0) + v
            n += 1
        eval_metrics = {k: v / n for k, v in eval_metrics.items()}

        # Train and evaluate linear probe on the violation target
        try:
            from .eval.probe import evaluate_probe, train_violation_probe

            probe = train_violation_probe(
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
            f"K={eval_metrics.get('eval/K_epoch', 0):.0f} "
            f"k_used={eval_metrics.get('eval/k_used_mean', 0):.1f} "
            f"probe_r2={eval_metrics.get('eval/probe/r2', 0):.3f}"
        )

        # Render + log deliberation/rollout gifs to Aim. Dense early (every
        # epoch through the curriculum ramp), sparse once training plateaus.
        # Cadence scales with total epochs (see _should_render_gifs).
        if _should_render_gifs(epoch, cfg.epochs, every_n_epochs=cfg.viz_every_n_epochs):
            try:
                # Pull one dev batch for visualization. Reuse the first batch
                # by re-iterating; eval was in the same model.eval() context.
                eval_iter = iter(loaders["dev"])
                viz_batch = next(eval_iter)
                gif_dir = Path(cfg.exp_dir) / "gifs"
                gifs = render_rollout_for_eval(
                    model, decoder, viz_batch, cfg, gif_dir, sample_idx=0
                )
                logger.log_image(
                    "gif/deliberation",
                    gifs["deliberation_frames"],
                    step=step,
                    context={"subset": "gifs"},
                    caption=f"deliberation ep{epoch}",
                )
                logger.log_image(
                    "gif/rollout",
                    gifs["rollout_img"],
                    step=step,
                    context={"subset": "gifs"},
                    caption=f"rollout ep{epoch}",
                )
            except Exception as e:
                print(f"Gif render skipped: {e}")

        # checkpoint (model + decoder + decoder optimizer)
        ckpt = Path(cfg.exp_dir) / "ckpt.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "decoder": decoder.state_dict(),
            "decoder_optimizer": decoder_optimizer.state_dict(),
            "cfg": cfg.to_dict(),
            "epoch": epoch,
        }, ckpt)

    logger.close()
