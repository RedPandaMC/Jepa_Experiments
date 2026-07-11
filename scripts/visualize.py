#!/usr/bin/env python
r"""Generate rich animations of RD-JEPA forecasting and oscillator dynamics.

Produces two MP4 animations:
    1. ``oscillator_dynamics.mp4`` — N modes on the unit circle evolving
       through K resonance steps (phases + amplitudes), plus the coupling
       matrix heatmap and amplitude bar chart.
    2. ``forecasting.mp4`` — rolling forecast over the test split: context
       window, predicted future, and ground-truth future for a few key
       variables, animating forward through time.

Usage:
    uv run python scripts/visualize.py                     # uses best MLflow trial
    uv run python scripts/visualize.py --ckpt runs/default/ckpt.pt
    uv run python scripts/visualize.py --epochs 5           # retrain quickly
    uv run python scripts/visualize.py --no-anim            # static plots only
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch

from rd_jepa.config import Config
from rd_jepa.data.forecasting import JenaClimateDataset
from rd_jepa.eval.forecast_probe import ForecastProbe
from rd_jepa.models.rd_jepa import RDJEPA
from rd_jepa.train import train

COLUMN_NAMES = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)",
    "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m²)",
    "PAR (µmol/m²/s)", "max. PAR (µmol/m²/s)", "Tlog (degC)", "CO2 (ppm)",
]

OUTPUT_DIR = Path("runs/viz")


def _load_best_cfg_from_mlflow(db_path: str = "mlflow.db") -> Config | None:
    """Load the best trial's params from MLflow and build a Config."""
    if not Path(db_path).exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            "SELECT run_uuid, value FROM latest_metrics "
            "WHERE key='probe/mse' ORDER BY value ASC LIMIT 1"
        )
        row = c.fetchone()
        if row is None:
            return None
        run_uuid, _ = row
        c.execute(
            "SELECT key, value FROM params WHERE run_uuid=?", (run_uuid,)
        )
        params = dict(c.fetchall())
        conn.close()
        return _cfg_from_params(params)
    except Exception as e:
        print(f"Warning: could not read MLflow: {e}")
        return None


def _cfg_from_params(params: dict[str, str]) -> Config:
    """Build a Config from MLflow param strings."""
    overrides: dict[str, object] = {}
    type_hints: dict[str, type] = {
        "context_len": int, "horizon": int, "n_features": int,
        "patch_len": int, "latent_dim": int, "n_modes": int,
        "K_steps": int, "encoder_layers": int, "encoder_hidden": int,
        "batch_size": int, "epochs": int, "ema_warmup": int,
        "lr_warmup_steps": int, "num_workers": int,
        "probe_steps": int, "eval_every_n_epochs": int,
        "log_every_n_steps": int, "seed": int,
        "val_ratio": float, "test_ratio": float, "dt": float,
        "coupling_sparsity": float, "amp_init": float,
        "vicreg_var_weight": float, "vicreg_cov_weight": float,
        "vicreg_target_std": float, "phase_div_weight": float,
        "lr": float, "weight_decay": float, "ema_decay": float,
        "probe_lr": float, "optuna_n_trials": int,
        "optuna_timeout": int,
    }
    for key, val in params.items():
        if key in type_hints:
            cast = type_hints[key]
            try:
                overrides[key] = cast(val)
            except (ValueError, TypeError):
                pass
    return Config(**overrides)


def _load_checkpoint(path: Path, device: torch.device) -> tuple[RDJEPA, Config]:
    """Load model + probe from a checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    import dataclasses as dc
    valid_names = {f.name for f in dc.fields(Config)}
    cfg_kwargs = {}
    for k, v in ckpt["cfg"].items():
        if k not in valid_names:
            continue
        if k in ("data_dir", "runs_dir"):
            v = Path(v)
        elif isinstance(v, list) and k == "freq_init_range":
            v = tuple(v)
        cfg_kwargs[k] = v
    cfg = Config(**cfg_kwargs)
    model = RDJEPA(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _train_quick(cfg: Config, device: torch.device) -> tuple[RDJEPA, ForecastProbe]:
    """Train a quick model + probe for visualization."""
    print("Training a quick model for visualization...")
    train_cfg = Config(
        latent_dim=cfg.latent_dim,
        n_modes=cfg.n_modes,
        K_steps=cfg.K_steps,
        dt=cfg.dt,
        coupling_sparsity=cfg.coupling_sparsity,
        encoder_hidden=cfg.encoder_hidden,
        vicreg_var_weight=cfg.vicreg_var_weight,
        vicreg_cov_weight=cfg.vicreg_cov_weight,
        phase_div_weight=cfg.phase_div_weight,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        fast=False,
        exp_name="viz",
    )
    train(train_cfg)
    ckpt_path = train_cfg.exp_dir / "ckpt.pt"
    model, cfg2 = _load_checkpoint(ckpt_path, device)
    probe = ForecastProbe(cfg2.latent_dim, cfg2.horizon, cfg2.n_features)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "probe" in ckpt:
        probe.load_state_dict(ckpt["probe"])
    probe.to(device)
    probe.eval()
    return model, probe


def _get_model_and_probe(
    args, device: torch.device
) -> tuple[RDJEPA, ForecastProbe, Config]:
    """Resolve model + probe from checkpoint, MLflow, or quick retrain."""
    if args.ckpt and Path(args.ckpt).exists():
        print(f"Loading checkpoint: {args.ckpt}")
        model, cfg = _load_checkpoint(Path(args.ckpt), device)
        probe = ForecastProbe(cfg.latent_dim, cfg.horizon, cfg.n_features)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        if "probe" in ckpt:
            probe.load_state_dict(ckpt["probe"])
        probe.to(device)
        probe.eval()
        return model, probe, cfg

    if not args.retrain:
        cfg = _load_best_cfg_from_mlflow()
        if cfg is not None:
            print(f"Loaded best trial from MLflow (latent_dim={cfg.latent_dim}, "
                  f"n_modes={cfg.n_modes}, K={cfg.K_steps})")
            ckpt_path = cfg.exp_dir / "ckpt.pt"
            if ckpt_path.exists():
                model, cfg = _load_checkpoint(ckpt_path, device)
                probe = ForecastProbe(cfg.latent_dim, cfg.horizon, cfg.n_features)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                if "probe" in ckpt:
                    probe.load_state_dict(ckpt["probe"])
                probe.to(device)
                probe.eval()
                return model, probe, cfg

    # Retrain
    cfg = _load_best_cfg_from_mlflow() or Config()
    if args.epochs:
        cfg.epochs = args.epochs
    model, probe = _train_quick(cfg, device)
    return model, probe, cfg


@torch.no_grad()
def visualize_oscillator_dynamics(
    model: RDJEPA,
    cfg: Config,
    device: torch.device,
    save_path: Path,
) -> None:
    """Animate the K-step oscillator phase + amplitude evolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get one sample from the test set
    ds = JenaClimateDataset(
        cfg.data_dir / "jena_climate_2009_2016.csv",
        context_len=cfg.context_len,
        horizon=cfg.horizon,
        n_features=cfg.n_features,
        split="test",
    )
    context, _ = ds[0]
    context = context.unsqueeze(0).to(device)

    # Run model with large K to see the full trajectory
    K_viz = max(cfg.K_steps, 20)
    with torch.no_grad():
        out = model(context, K_steps=K_viz)
        all_phases = out["all_phases"].squeeze().cpu().numpy()  # [K, N]

    # Also get amplitudes at each step by re-running step-by-step
    with torch.no_grad():
        z_0 = model.encoder(context)
        r_init, phi_init = model.analytic(z_0)
        r_init = r_init.squeeze(0).cpu().numpy()
        phi_init = phi_init.squeeze(0).cpu().numpy()

        # Get oscillator params
        omega = model.resonator.freq_net(z_0) + model.resonator.freq_bias
        omega = (torch.tanh(omega) * 3.0).squeeze(0).cpu().numpy()
        coupling_flat = model.resonator.coupling_net(z_0)
        coupling = coupling_flat.reshape(cfg.n_modes, cfg.n_modes)
        coupling = coupling * model.resonator.coupling_mask
        coupling = torch.tanh(coupling).cpu().numpy()
        alpha = (torch.sigmoid(model.resonator.alpha_net(z_0)) * 0.5).squeeze(0).cpu().numpy()
        r_eq = torch.nn.functional.softplus(model.resonator.eq_net(z_0)).squeeze(0).cpu().numpy()

    # Simulate amplitudes step by step
    r_traj = np.zeros((K_viz, cfg.n_modes))
    r_k = r_init.copy()
    for k in range(K_viz):
        r_k = r_k + alpha * (r_eq - r_k) * cfg.dt
        r_traj[k] = r_k

    # Prepend initial state
    phases_full = np.vstack([phi_init, all_phases])  # [K+1, N]
    r_full = np.vstack([r_init, r_traj])  # [K+1, N]

    n_modes = cfg.n_modes
    n_show = min(n_modes, 32)  # cap for readability

    # ── Figure setup ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor="#0f0f1a")
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    ax_circle = fig.add_subplot(gs[:, 0], facecolor="#1a1a2e")
    ax_amp = fig.add_subplot(gs[0, 1], facecolor="#1a1a2e")
    ax_freq = fig.add_subplot(gs[1, 1], facecolor="#1a1a2e")
    ax_couple = fig.add_subplot(gs[:, 2], facecolor="#1a1a2e")

    fig.suptitle("RD-JEPA Oscillator Dynamics", color="white", fontsize=14, y=0.97)

    # Unit circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax_circle.plot(np.cos(theta_circle), np.sin(theta_circle), color="#333355", lw=1)
    ax_circle.set_xlim(-1.8, 1.8)
    ax_circle.set_ylim(-1.8, 1.8)
    ax_circle.set_aspect("equal")
    ax_circle.set_title("Phase Circle (amplitude × phase)", color="white", fontsize=10)
    ax_circle.tick_params(colors="#888")
    for spine in ax_circle.spines.values():
        spine.set_color("#333")

    # Amplitude bars
    ax_amp.set_xlim(0, n_show)
    ax_amp.set_ylim(0, max(r_full.max() * 1.2, 1.0))
    ax_amp.set_title("Mode Amplitudes", color="white", fontsize=10)
    ax_amp.tick_params(colors="#888")
    for spine in ax_amp.spines.values():
        spine.set_color("#333")
    amp_bars = ax_amp.bar(range(n_show), r_full[0, :n_show], color="#4ecdc4", alpha=0.8)

    # Frequencies
    ax_freq.set_xlim(0, n_show)
    ax_freq.set_ylim(omega[:n_show].min() * 1.2 - 0.1, omega[:n_show].max() * 1.2 + 0.1)
    ax_freq.set_title("Natural Frequencies ω", color="white", fontsize=10)
    ax_freq.tick_params(colors="#888")
    for spine in ax_freq.spines.values():
        spine.set_color("#333")
    ax_freq.bar(range(n_show), omega[:n_show], color="#ff6b6b", alpha=0.8)

    # Coupling matrix
    im = ax_couple.imshow(coupling, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_couple.set_title("Coupling Matrix K", color="white", fontsize=10)
    ax_couple.tick_params(colors="#888")
    for spine in ax_couple.spines.values():
        spine.set_color("#333")
    fig.colorbar(im, ax=ax_couple, fraction=0.046, pad=0.04)

    # Color palette for modes
    colors = plt.cm.hsv(np.linspace(0, 1, n_show))

    # Scatter points on circle
    scatter = ax_circle.scatter(
        r_full[0, :n_show] * np.cos(phases_full[0, :n_show]),
        r_full[0, :n_show] * np.sin(phases_full[0, :n_show]),
        c=colors, s=80, zorder=5, edgecolors="white", linewidths=0.5,
    )
    # Lines from origin
    lines = []
    for i in range(n_show):
        ln, = ax_circle.plot(
            [0, r_full[0, i] * np.cos(phases_full[0, i])],
            [0, r_full[0, i] * np.sin(phases_full[0, i])],
            color=colors[i], alpha=0.4, lw=1.5,
        )
        lines.append(ln)

    step_text = ax_circle.text(0.02, 0.97, "", transform=ax_circle.transAxes,
                               color="#4ecdc4", fontsize=11, va="top",
                               fontfamily="monospace")

    total_steps = len(phases_full)

    def update(frame: int):
        k = min(frame, total_steps - 1)
        xs = r_full[k, :n_show] * np.cos(phases_full[k, :n_show])
        ys = r_full[k, :n_show] * np.sin(phases_full[k, :n_show])

        scatter.set_offsets(np.c_[xs, ys])

        for i, ln in enumerate(lines):
            ln.set_data([0, xs[i]], [0, ys[i]])

        for i, bar in enumerate(amp_bars):
            bar.set_height(r_full[k, i])

        step_text.set_text(
            f"Step {k}/{total_steps-1}\n"
            f"dt={cfg.dt}  K={cfg.K_steps}\n"
            f"modes={n_modes}  d={cfg.latent_dim}"
        )
        return [scatter, step_text] + lines + list(amp_bars)

    anim = animation.FuncAnimation(
        fig, update, frames=total_steps, interval=150, blit=False, repeat=True
    )

    save_path = OUTPUT_DIR / "oscillator_dynamics.mp4"
    writer = animation.FFMpegWriter(fps=8, bitrate=2000)
    anim.save(str(save_path), writer=writer)
    plt.close(fig)
    print(f"Saved: {save_path}")


@torch.no_grad()
def visualize_forecasting(
    model: RDJEPA,
    probe: ForecastProbe,
    cfg: Config,
    device: torch.device,
    save_path: Path,
) -> None:
    """Animate rolling forecast over the test split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = JenaClimateDataset(
        cfg.data_dir / "jena_climate_2009_2016.csv",
        context_len=cfg.context_len,
        horizon=cfg.horizon,
        n_features=cfg.n_features,
        split="test",
    )

    # Pick a few interesting variables
    interesting = [1, 4, 5, 11]  # T, rh, VPmax, wv
    interesting = [i for i in interesting if i < cfg.n_features]
    n_vars = len(interesting)

    # Get normalization stats for denormalization
    mean = ds.mean
    std = ds.std

    # Run a rolling forecast
    n_rollouts = 20
    start_idx = 0
    step = 10  # stride between rollouts

    all_contexts = []
    all_preds = []
    all_targets = []

    for i in range(n_rollouts):
        idx = start_idx + i * step
        if idx + cfg.context_len + cfg.horizon > len(ds):
            break
        context, target = ds[idx]
        context = context.unsqueeze(0).to(device)
        out = model(context, K_steps=cfg.K_steps)
        h_k = out["h_K"]
        pred = probe(h_k).squeeze(0).cpu().numpy()  # [H, C]
        target = target.cpu().numpy()  # [H, C]

        # Denormalize
        pred_denorm = pred * std + mean
        target_denorm = target * std + mean
        context_denorm = context.squeeze(0).cpu().numpy() * std + mean

        all_contexts.append(context_denorm)
        all_preds.append(pred_denorm)
        all_targets.append(target_denorm)

    n_frames = len(all_preds)

    # Build figure
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2.5 * n_vars),
                             facecolor="#0f0f1a", sharex=True)
    if n_vars == 1:
        axes = [axes]

    fig.suptitle("RD-JEPA Rolling Forecast (test split)",
                 color="white", fontsize=14, y=0.98)

    L = cfg.context_len
    H = cfg.horizon
    x_total = L + H
    x_ctx = np.arange(L)
    x_pred = np.arange(L, L + H)

    lines_ctx = []
    lines_pred = []
    lines_target = []
    fills = []
    vlines = []

    for j, var_idx in enumerate(interesting):
        ax = axes[j]
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(0, x_total)
        name = COLUMN_NAMES[var_idx] if var_idx < len(COLUMN_NAMES) else f"var_{var_idx}"
        ax.set_ylabel(name, color="white", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")

        # Context line
        ctx_data = all_contexts[0][:, var_idx]
        ln_ctx, = ax.plot(x_ctx, ctx_data, color="#4ecdc4", lw=1.5, label="Context")
        lines_ctx.append(ln_ctx)

        # Prediction line
        pred_data = all_preds[0][:, var_idx]
        ln_pred, = ax.plot(x_pred, pred_data, color="#ff6b6b", lw=2, label="Forecast")
        lines_pred.append(ln_pred)

        # Target line
        tgt_data = all_targets[0][:, var_idx]
        ln_tgt, = ax.plot(x_pred, tgt_data, color="#888", lw=1, ls="--", label="Actual")
        lines_target.append(ln_tgt)

        # Fill between pred and target
        fill = ax.fill_between(x_pred, pred_data, tgt_data, alpha=0.15, color="#ff6b6b")
        fills.append(fill)

        # Vertical line at context/prediction boundary
        vline = ax.axvline(x=L, color="#555", ls=":", lw=1)
        vlines.append(vline)

        if j == 0:
            ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
                      edgecolor="#333", labelcolor="white")

    info_text = axes[0].text(
        0.02, 0.85, "", transform=axes[0].transAxes,
        color="#4ecdc4", fontsize=10, fontfamily="monospace", va="top",
    )

    def update(frame: int):
        k = min(frame, n_frames - 1)
        for j, var_idx in enumerate(interesting):
            ctx_data = all_contexts[k][:, var_idx]
            pred_data = all_preds[k][:, var_idx]
            tgt_data = all_targets[k][:, var_idx]

            lines_ctx[j].set_ydata(ctx_data)
            lines_pred[j].set_ydata(pred_data)
            lines_target[j].set_ydata(tgt_data)

            # Update y-limits
            all_vals = np.concatenate([ctx_data, pred_data, tgt_data])
            ymin, ymax = all_vals.min(), all_vals.max()
            margin = (ymax - ymin) * 0.1 + 0.01
            axes[j].set_ylim(ymin - margin, ymax + margin)

            # Remove and re-add fill
            if fills[j]:
                fills[j].remove()
            fills[j] = axes[j].fill_between(
                x_pred, pred_data, tgt_data, alpha=0.15, color="#ff6b6b"
            )

        # Compute MSE for this frame
        mse = np.mean((all_preds[k] - all_targets[k]) ** 2)
        mae = np.mean(np.abs(all_preds[k] - all_targets[k]))
        info_text.set_text(
            f"Rollout {k+1}/{n_frames}  "
            f"MSE={mse:.4f}  MAE={mae:.4f}  "
            f"K={cfg.K_steps}  N={cfg.n_modes}"
        )
        return lines_ctx + lines_pred + lines_target + fills + [info_text]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=300, blit=False, repeat=True
    )

    save_path = OUTPUT_DIR / "forecasting.mp4"
    writer = animation.FFMpegWriter(fps=4, bitrate=2000)
    anim.save(str(save_path), writer=writer)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RD-JEPA visualization")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--retrain", action="store_true", help="Retrain a quick model")
    parser.add_argument("--epochs", type=int, default=0, help="Epochs if retraining")
    parser.add_argument("--no-anim", action="store_true", help="Skip animations")
    parser.add_argument("--output-dir", type=str, default="runs/viz")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, probe, cfg = _get_model_and_probe(args, device)

    # Also train the probe if it's untrained
    if probe is not None:
        probe.to(device)
        probe.eval()

    print(f"Config: latent_dim={cfg.latent_dim}, n_modes={cfg.n_modes}, "
          f"K={cfg.K_steps}, dt={cfg.dt}")

    if not args.no_anim:
        try:
            visualize_oscillator_dynamics(model, cfg, device, OUTPUT_DIR)
        except Exception as e:
            print(f"Oscillator animation failed: {e}")

        if probe is not None:
            try:
                visualize_forecasting(model, probe, cfg, device, OUTPUT_DIR)
            except Exception as e:
                print(f"Forecasting animation failed: {e}")

    print(f"\nAnimations saved to: {OUTPUT_DIR}/")
    print("  - oscillator_dynamics.mp4")
    print("  - forecasting.mp4")


if __name__ == "__main__":
    main()
