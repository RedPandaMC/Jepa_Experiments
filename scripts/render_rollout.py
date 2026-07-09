#!/usr/bin/env python
"""Render deliberation + rollout gifs from a trained RD-JEPA checkpoint.

Usage:
  uv run python scripts/render_rollout.py --exp default
"""
import argparse
import sys
from pathlib import Path

import torch

from rd_jepa.config import Config
from rd_jepa.data.loader import build_dataloaders
from rd_jepa.models.rd_jepa import RDJEPA
from rd_jepa.viz.decoder import VizDecoder
from rd_jepa.viz.gif_writer import render_rollout_for_eval


def main() -> int:
    p = argparse.ArgumentParser(description="Render RD-JEPA rollout gifs")
    p.add_argument("--exp", default="default", help="experiment name")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--K", type=int, default=None)
    args = p.parse_args()

    exp_dir = Path("runs") / args.exp
    ckpt_path = exp_dir / "ckpt.pt"
    if not ckpt_path.exists():
        print(f"ERROR: no checkpoint at {ckpt_path}. Train first.", file=sys.stderr)
        return 1

    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg_dict = ckpt["cfg"]
    # rebuild config from saved dict (enums are strings)
    from rd_jepa.config import ActionInject, Gate, LatentShape, LossTrajectory

    cfg = Config(
        K=args.K or cfg_dict.get("K", 15),
        latent_dim=cfg_dict.get("latent_dim", 256),
        gate=Gate(cfg_dict.get("gate", "sigmoid")),
        action_inject=ActionInject(cfg_dict.get("action_inject", "every")),
        loss_trajectory=LossTrajectory(cfg_dict.get("loss_trajectory", "discounted")),
        latent_shape=LatentShape(cfg_dict.get("latent_shape", "flat")),
        exp_name=args.exp,
    )

    model = RDJEPA(cfg).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()

    decoder = VizDecoder(latent_dim=cfg.latent_total_dim).cuda()
    # the decoder is not trained in this POC; it still produces visible frames
    # via its random init, which is enough to demonstrate the gif pipeline.

    loaders = build_dataloaders(cfg)
    batch = next(iter(loaders["dev"]))
    out_dir = exp_dir / "gifs"
    for i in range(args.n_samples):
        paths = render_rollout_for_eval(model, decoder, batch, cfg, out_dir, sample_idx=i)
        print(f"sample {i}: {paths['deliberation'].name}, {paths['rollout'].name}")
    print(f"\ngifs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
