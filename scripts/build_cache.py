#!/usr/bin/env python
"""Pre-roll PhyRE simulations into a compact .npz cache.

Runs under the dedicated PhyRE (Python 3.9) venv — see
rd_jepa/data/phyre_env.py. Produces per-fold shards containing state
transitions (s_t, action, s_{t+1}) suitable for JEPA training.

Each simulation produces `n_frames` evenly-spaced frames. We extract
consecutive-frame pairs as transitions, downsample to frame_size, and
store the raw uint8 scene-id maps (NOT RGB) — the encoder consumes the
scene-id channel directly, which preserves object identity.

Usage:
    <phyre39>/bin/python scripts/build_cache.py --tier ball_cross_template --fold 0
    <phyre39>/bin/python scripts/build_cache.py --help
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import phyre
from tqdm import tqdm

CACHE_VERSION = 1


def downsample(frames: np.ndarray, size: int) -> np.ndarray:
    """Downsample [T, H, W] uint8 scene-id maps via nearest-neighbour.

    Nearest keeps object-id semantics intact (averaging would blend ids
    into nonsense values).
    """
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    import cv2

    out = np.empty((frames.shape[0], size, size), dtype=np.uint8)
    for t in range(frames.shape[0]):
        out[t] = cv2.resize(frames[t], (size, size), interpolation=cv2.INTER_NEAREST)
    return out


def build_cache(
    tier: str = "ball_cross_template",
    fold: int = 0,
    n_actions: int = 100,
    n_frames: int = 8,
    frame_size: int = 64,
    out_dir: Path = Path("data/cache"),
    seed: int = 42,
    fast: bool = False,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, dev, test = phyre.get_fold(tier, fold)
    splits = {"train": train, "dev": dev, "test": test}
    if fast:
        splits = {k: v[:50] for k, v in splits.items()}

    action_mapper = phyre.action_mappers.get_action_mapper(tier.split("_")[0])

    total_pairs = 0
    for split_name, task_ids in splits.items():
        print(f"[{split_name}] {len(task_ids)} tasks, {n_actions} actions/task")
        sim = phyre.initialize_simulator(task_ids, "ball")

        s_t_list = []
        a_list = []
        s_tp1_list = []

        for task_i in tqdm(range(len(task_ids)), desc=split_name):
            for _ in range(n_actions):
                action = action_mapper.sample()
                res = sim.simulate_action(
                    task_i, action, need_images=True, stride=1
                )
                if res.images is None:
                    continue

                imgs = res.images  # [T, 256, 256] uint8 scene-ids
                if imgs.shape[0] < 2:
                    continue

                # pick n_frames evenly-spaced frames; form consecutive pairs
                idx = np.linspace(0, imgs.shape[0] - 1, n_frames).astype(int)
                frames = imgs[idx]
                frames_ds = downsample(frames, frame_size)  # [n_frames, H, W]

                for j in range(n_frames - 1):
                    s_t_list.append(frames_ds[j])
                    a_list.append(action.astype(np.float32))
                    s_tp1_list.append(frames_ds[j + 1])

        s_t = np.stack(s_t_list)  # [N, H, W] uint8
        a = np.stack(a_list)  # [N, 3] float32
        s_tp1 = np.stack(s_tp1_list)

        out_path = out_dir / f"{tier}_fold{fold}_{split_name}.npz"
        np.savez_compressed(
            out_path,
            s_t=s_t,
            action=a,
            s_tp1=s_tp1,
            task_ids=np.array(task_ids, dtype=object),
            frame_size=frame_size,
            version=CACHE_VERSION,
        )
        mb = out_path.stat().st_size / 1e6
        print(f"  wrote {out_path}: {s_t.shape[0]} pairs, {mb:.1f} MB")
        total_pairs += s_t.shape[0]

    print(f"\nDone: {total_pairs} transitions cached.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Build PhyRE simulation cache")
    p.add_argument("--tier", default="ball_cross_template")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n_actions", type=int, default=100)
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--frame_size", type=int, default=64)
    p.add_argument("--out_dir", default="data/cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true", help="50-task subset for ablations")
    args = p.parse_args()

    return build_cache(
        tier=args.tier,
        fold=args.fold,
        n_actions=args.n_actions,
        n_frames=args.n_frames,
        frame_size=args.frame_size,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        fast=args.fast,
    )


if __name__ == "__main__":
    sys.exit(main())
