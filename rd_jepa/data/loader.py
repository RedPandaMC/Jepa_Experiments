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
    """Single-split dataset over cached (s_t, action, s_tp1) transitions."""

    def __init__(self, npz_path: str | Path):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Cache shard not found: {path}. Run build_cache.py first.")
        d = np.load(path, allow_pickle=True)
        self.s_t = d["s_t"]  # [N, H, W] uint8
        self.s_tp1 = d["s_tp1"]
        self.action = d["action"]  # [N, 3] float32
        self.frame_size = int(d["frame_size"])
        self.max_id = max(int(self.s_t.max()), int(self.s_tp1.max()))

    def __len__(self) -> int:
        return self.s_t.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # normalize scene-ids to [0,1]; add channel dim -> [1, H, W]
        s_t = torch.from_numpy(self.s_t[idx]).float().div(self.max_id).unsqueeze(0)
        s_tp1 = torch.from_numpy(self.s_tp1[idx]).float().div(self.max_id).unsqueeze(0)
        action = torch.from_numpy(self.action[idx])
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
            num_workers=2,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
