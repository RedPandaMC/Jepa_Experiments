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
    stride: int = 10,
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

        # Re-initialize the simulator per shard to bound memory (phyre leaks
        # internal state per task; initializing all 1600 at once OOMs on 8GB).
        shard_size = 200

        s_t_list: list[np.ndarray] = []
        a_list: list[np.ndarray] = []
        s_tp1_list: list[np.ndarray] = []

        def flush_shard(shard_idx: int) -> int:
            nonlocal s_t_list, a_list, s_tp1_list
            if not s_t_list:
                return 0
            s_t = np.stack(s_t_list)
            a = np.stack(a_list)
            s_tp1 = np.stack(s_tp1_list)
            out_path = out_dir / f"{tier}_fold{fold}_{split_name}_shard{shard_idx}.npz"
            np.savez_compressed(
                out_path,
                s_t=s_t,
                action=a,
                s_tp1=s_tp1,
                frame_size=frame_size,
                version=CACHE_VERSION,
            )
            mb = out_path.stat().st_size / 1e6
            print(f"  shard {shard_idx}: {s_t.shape[0]} pairs, {mb:.1f} MB -> {out_path.name}")
            n = s_t.shape[0]
            s_t_list, a_list, s_tp1_list = [], [], []
            return n

        shard_idx = 0
        for start in tqdm(range(0, len(task_ids), shard_size), desc=split_name):
            end = min(start + shard_size, len(task_ids))
            shard_ids = task_ids[start:end]
            sim = phyre.initialize_simulator(shard_ids, "ball")
            for local_i in range(len(shard_ids)):
                for _ in range(n_actions):
                    action = action_mapper.sample()
                    try:
                        res = sim.simulate_action(
                            local_i, action, need_images=True, stride=stride
                        )
                    except Exception:
                        continue
                    if res.images is None:
                        continue

                    imgs = res.images
                    if imgs.shape[0] < 2:
                        continue

                    idx = np.linspace(0, imgs.shape[0] - 1, n_frames).astype(int)
                    frames = imgs[idx]
                    frames_ds = downsample(frames, frame_size)

                    for j in range(n_frames - 1):
                        s_t_list.append(frames_ds[j])
                        a_list.append(action.astype(np.float32))
                        s_tp1_list.append(frames_ds[j + 1])

            # flush after each shard and release the simulator
            total_pairs += flush_shard(shard_idx)
            shard_idx += 1
            del sim

        # write the task_ids manifest for this split
        manifest_path = out_dir / f"{tier}_fold{fold}_{split_name}_manifest.txt"
        manifest_path.write_text("\n".join(task_ids), encoding="utf-8")

    print(f"\nDone: {total_pairs} transitions cached (sharded).")
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
    p.add_argument("--stride", type=int, default=10, help="frame subsample stride")
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
        stride=args.stride,
    )


if __name__ == "__main__":
    sys.exit(main())
