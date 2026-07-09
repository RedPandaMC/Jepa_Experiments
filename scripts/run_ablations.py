#!/usr/bin/env python
"""Run the RD-JEPA ablation sweep over the 4 open research decisions.

Decisions (spec §5):
  D1 gate             : sigmoid vs sparsemax
  D2 action_inject    : once vs every
  D3 loss_trajectory  : final vs discounted
  D4 latent_shape     : flat vs spatial

Modes:
  --mode oat      : one-at-a-time (8 runs, others at default)   [default]
  --mode cartesian: full 2^4 = 16-run sweep

Each run is a short training on the 500-sample subset (--fast) for a few
epochs. Results are logged to Aim as separate experiments and a markdown
summary table is printed at the end.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from rd_jepa.config import ActionInject, Config, Gate, LatentShape, LossTrajectory


def oat_configs(defaults: dict) -> list[tuple[str, Config]]:
    """One-at-a-time: vary each decision while holding the rest at default."""
    base = dict(defaults)
    runs: list[tuple[str, Config]] = []
    variations = {
        "gate": [Gate.SIGMOID, Gate.SPARSEMAX],
        "action_inject": [ActionInject.ONCE, ActionInject.EVERY],
        "loss_trajectory": [LossTrajectory.FINAL, LossTrajectory.DISCOUNTED],
        "latent_shape": [LatentShape.FLAT, LatentShape.SPATIAL],
    }
    for dec, opts in variations.items():
        for opt in opts:
            name = f"{dec}={opt.value}"
            cfg_kwargs = dict(base)
            cfg_kwargs[dec] = opt
            cfg_kwargs["exp_name"] = f"abl_{dec}_{opt.value}"
            runs.append((name, Config(**cfg_kwargs)))
    return runs


def cartesian_configs(defaults: dict) -> list[tuple[str, Config]]:
    """Full 2^4 Cartesian sweep."""
    base = dict(defaults)
    gates = [Gate.SIGMOID, Gate.SPARSEMAX]
    injects = [ActionInject.ONCE, ActionInject.EVERY]
    losses = [LossTrajectory.FINAL, LossTrajectory.DISCOUNTED]
    shapes = [LatentShape.FLAT, LatentShape.SPATIAL]
    runs: list[tuple[str, Config]] = []
    for g, a, lt, s in itertools.product(gates, injects, losses, shapes):
        name = f"g={g.value}_a={a.value}_l={lt.value}_s={s.value}"
        cfg_kwargs = dict(base)
        cfg_kwargs.update(gate=g, action_inject=a, loss_trajectory=lt, latent_shape=s)
        cfg_kwargs["exp_name"] = f"abl_{name}"
        runs.append((name, Config(**cfg_kwargs)))
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description="RD-JEPA ablation sweep")
    p.add_argument("--mode", choices=["oat", "cartesian"], default="oat")
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", default="runs/ablations")
    args = p.parse_args()

    # import here so --help is fast and the phyre39 venv is not required
    from rd_jepa.train import train
    from rd_jepa.viz.aim_logger import AimLogger

    defaults = {
        "K": args.K,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "fast": True,
        "gate": Gate.SIGMOID,
        "action_inject": ActionInject.EVERY,
        "loss_trajectory": LossTrajectory.DISCOUNTED,
        "latent_shape": LatentShape.FLAT,
    }

    runs = oat_configs(defaults) if args.mode == "oat" else cartesian_configs(defaults)
    print(f"Running {len(runs)} ablation runs ({args.mode})...")
    for i, (name, cfg) in enumerate(runs):
        print(f"\n[{i + 1}/{len(runs)}] {name} -> exp={cfg.exp_name}")
        logger = AimLogger(cfg)
        try:
            train(cfg, logger=logger)
        except Exception as e:  # noqa: BLE001
            print(f"  RUN FAILED: {e}")
            continue

    # write a simple summary of what ran
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_index.md").write_text(
        f"# Ablation sweep ({args.mode})\n\n"
        f"Mode: {args.mode}\nRuns: {len(runs)}\n\n"
        "Each run logs to Aim as experiment `abl_*`. Open Aim UI to compare:\n"
        "```\n  aim up\n```\n\n"
        "## Experiments\n"
        + "\n".join(f"- `{cfg.exp_name}` — {name}" for name, cfg in runs)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary index written to {out / 'sweep_index.md'}")
    print("Launch the Aim UI to compare:  aim up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
