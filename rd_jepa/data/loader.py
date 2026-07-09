"""PyTorch dataset/dataloader for the cached PhyRE transitions.

Reads the .npz shards produced by scripts/build_cache.py and yields
(s_t, action, s_tp1) batches on the training device. The scene-id maps
are normalized to float in [0,1] (divided by the max scene id seen in
the cache) so the encoder receives a clean continuous input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config


class PhyreTransitionDataset(Dataset):
    """Single-split dataset over cached (s_t, action, s_tp1) transitions.

    Reads the sharded .npz cache produced by scripts/build_cache.py and
    keeps shards in memory (no concatenation — concatenating 3GB of
    64x64 frames on the slow /mnt/c WSL filesystem hangs). Indexing is
    translated across shards via cumulative offsets.
    """

    def __init__(self, npz_path: str | Path):
        path = Path(npz_path)
        stem = path.stem  # e.g. "ball_cross_template_fold0_train"
        parent = path.parent
        shards = sorted(parent.glob(f"{stem}_shard*.npz"))
        if shards:
            self._s_t = []
            self._s_tp1 = []
            self._action = []
            self._offsets = [0]
            for sp in shards:
                d = np.load(sp, allow_pickle=True)
                self._s_t.append(d["s_t"])
                self._s_tp1.append(d["s_tp1"])
                self._action.append(d["action"])
                self._offsets.append(self._offsets[-1] + d["s_t"].shape[0])
            self.frame_size = int(np.load(shards[0])["frame_size"])
        elif path.exists():
            d = np.load(path, allow_pickle=True)
            self._s_t = [d["s_t"]]
            self._s_tp1 = [d["s_tp1"]]
            self._action = [d["action"]]
            self._offsets = [0, d["s_t"].shape[0]]
            self.frame_size = int(d["frame_size"])
        else:
            raise FileNotFoundError(
                f"No cache shards or single-file cache at {path}. Run build_cache.py first."
            )
        self.max_id = max(
            max(int(s.max()) for s in self._s_t),
            max(int(s.max()) for s in self._s_tp1),
        )

    def __len__(self) -> int:
        return self._offsets[-1]

    def _shard_for(self, idx: int) -> tuple[int, int]:
        import bisect

        shard_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[shard_idx]
        return shard_idx, local_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shard_idx, local_idx = self._shard_for(idx)
        s_t = (
            torch.from_numpy(self._s_t[shard_idx][local_idx])
            .float()
            .div(self.max_id)
            .unsqueeze(0)
        )
        s_tp1 = (
            torch.from_numpy(self._s_tp1[shard_idx][local_idx]).float().div(self.max_id).unsqueeze(0)
        )
        action = torch.from_numpy(self._action[shard_idx][local_idx])
        return s_t, action, s_tp1


def build_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """Build train/dev/test dataloaders from the configured cache dir."""
    base = Path(cfg.cache_dir)
    shard = f"{cfg.tier}_fold{cfg.fold}_{{split}}.npz"
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "dev", "test"):
        path = base / shard.format(split=split)
        ds = PhyreTransitionDataset(path)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=0,  # dataset is in-RAM; forking workers re-loads shards
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
