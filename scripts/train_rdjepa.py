#!/usr/bin/env python
"""Train RD-JEPA. Run with: uv run python scripts/train_rdjepa.py --K 15

Ablation flags (--gate, --action_inject, --loss_trajectory, --latent_shape)
override the Config defaults so the sweep script can call this directly.
"""
import argparse
import sys

from rd_jepa.config import (
    ActionInject,
    Config,
    Gate,
    LatentShape,
    LossTrajectory,
)
from rd_jepa.train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RD-JEPA")
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--exp_name", default="default")
    p.add_argument("--fast", action="store_true", help="500-sample subset for ablations")
    p.add_argument("--gate", choices=[g.value for g in Gate], default=Gate.SIGMOID.value)
    p.add_argument(
        "--action_inject",
        choices=[a.value for a in ActionInject],
        default=ActionInject.EVERY.value,
    )
    p.add_argument(
        "--loss_trajectory",
        choices=[x.value for x in LossTrajectory],
        default=LossTrajectory.DISCOUNTED.value,
    )
    p.add_argument(
        "--latent_shape",
        choices=[x.value for x in LatentShape],
        default=LatentShape.FLAT.value,
    )
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--early_exit", action="store_true")
    p.add_argument("--violation_tau", type=float, default=0.1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        K=args.K,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        exp_name=args.exp_name,
        fast=args.fast,
        gate=Gate(args.gate),
        action_inject=ActionInject(args.action_inject),
        loss_trajectory=LossTrajectory(args.loss_trajectory),
        latent_shape=LatentShape(args.latent_shape),
        gamma=args.gamma,
        early_exit=args.early_exit,
        violation_tau=args.violation_tau,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
