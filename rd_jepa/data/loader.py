"""PyTorch dataset/dataloader for the cached PhyRE transitions.

Reads the .npz shards produced by scripts/build_cache.py and yields
(context, action, target, solved) batches on the training device.

Cache v2 format: each transition provides s_{t-1}, s_t, s_{t+1} and a
solved flag. We stack (s_{t-1}, s_t) as the 2-channel context and
(s_t, s_{t+1}) as the 2-channel target — both encoders see velocity.
Scene-id maps are normalized to float in [0,1] (divided by the max
scene id seen in the cache) so the encoder receives a clean continuous
input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config


class PhyreTransitionDataset(Dataset):
    """Dataset over cached (s_{t-1}, s_t, action, s_{t+1}, solved) transitions.

    Reads the sharded .npz cache (v2) produced by scripts/build_cache.py and
    keeps shards in memory. Indexing is translated across shards via cumulative
    offsets.
    """

    def __init__(self, npz_path: str | Path):
        path = Path(npz_path)
        stem = path.stem  # e.g. "ball_cross_template_fold0_train"
        parent = path.parent
        shards = sorted(parent.glob(f"{stem}_shard*.npz"))
        if shards:
            self._s_tm1: list[np.ndarray] = []
            self._s_t: list[np.ndarray] = []
            self._s_tp1: list[np.ndarray] = []
            self._action: list[np.ndarray] = []
            self._solved: list[np.ndarray] = []
            self._offsets: list[int] = [0]
            for sp in shards:
                d = np.load(sp, allow_pickle=True)
                version = int(d.get("version", 1))
                if version < 2:
                    raise RuntimeError(
                        f"Cache v{version} found at {sp}; "
                        "rebuild with scripts/build_cache.py to get v2 (s_tm1, solved)"
                    )
                self._s_tm1.append(d["s_tm1"])
                self._s_t.append(d["s_t"])
                self._s_tp1.append(d["s_tp1"])
                self._action.append(d["action"])
                self._solved.append(d["solved"])
                self._offsets.append(self._offsets[-1] + d["s_t"].shape[0])
            self.frame_size = int(np.load(shards[0])["frame_size"])
        elif path.exists():
            d = np.load(path, allow_pickle=True)
            version = int(d.get("version", 1))
            if version < 2:
                raise RuntimeError(
                    f"Cache v{version} found at {path}; "
                    "rebuild with scripts/build_cache.py to get v2 (s_tm1, solved)"
                )
            self._s_tm1 = [d["s_tm1"]]
            self._s_t = [d["s_t"]]
            self._s_tp1 = [d["s_tp1"]]
            self._action = [d["action"]]
            self._solved = [d["solved"]]
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

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (context[2,H,W], action[3], target[2,H,W], solved[1])."""
        shard_idx, local_idx = self._shard_for(idx)
        # Normalize to [0,1]
        s_tm1 = (
            torch.from_numpy(self._s_tm1[shard_idx][local_idx])
            .float()
            .div(self.max_id)
        )
        s_t = (
            torch.from_numpy(self._s_t[shard_idx][local_idx])
            .float()
            .div(self.max_id)
        )
        s_tp1 = (
            torch.from_numpy(self._s_tp1[shard_idx][local_idx])
            .float()
            .div(self.max_id)
        )
        # Stack: context = (s_{t-1}, s_t), target = (s_t, s_{t+1})
        context = torch.stack([s_tm1, s_t], dim=0)  # [2, H, W]
        target = torch.stack([s_t, s_tp1], dim=0)  # [2, H, W]
        action = torch.from_numpy(self._action[shard_idx][local_idx])
        solved = torch.tensor(self._solved[shard_idx][local_idx], dtype=torch.bool)
        return context, action, target, solved


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
