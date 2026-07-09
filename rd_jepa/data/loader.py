"""PyTorch dataset/dataloader for the cached MOVi transitions.

Reads the .npz shards produced by scripts/convert_movi.py and yields
(context, target, violation_gt) batches on the training device.

Cache v3 format (MOVi): each transition provides RGB frames
s_{t-1}, s_t, s_{t+1} of shape [H, W, 3] uint8 and a float violation_gt
(the normalized collision-force sum in the lookahead window after s_t).
We stack (s_{t-1}, s_t) as the multi-channel context and (s_t, s_{t+1})
as the multi-channel target — both encoders see velocity. Frames are
normalized to float in [0,1] by dividing by 255 (standard RGB).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config


class MoviTransitionDataset(Dataset):
    """Dataset over cached (s_{t-1}, s_t, s_{t+1}, violation_gt) transitions.

    Reads the sharded .npz cache (v3) produced by scripts/convert_movi.py
    and keeps shards in memory. Indexing is translated across shards via
    cumulative offsets.
    """

    def __init__(self, npz_path: str | Path):
        path = Path(npz_path)
        stem = path.stem  # e.g. "movi_a_train"
        parent = path.parent
        shards = sorted(parent.glob(f"{stem}_shard*.npz"))
        if shards:
            self._s_tm1: list[np.ndarray] = []
            self._s_t: list[np.ndarray] = []
            self._s_tp1: list[np.ndarray] = []
            self._violation_gt: list[np.ndarray] = []
            self._offsets: list[int] = [0]
            for sp in shards:
                d = np.load(sp, allow_pickle=True)
                version = int(d.get("version", 1))
                if version < 3:
                    raise RuntimeError(
                        f"Cache v{version} found at {sp}; "
                        "rebuild with scripts/convert_movi.py to get v3 "
                        "(RGB frames, violation_gt)"
                    )
                self._s_tm1.append(d["s_tm1"])
                self._s_t.append(d["s_t"])
                self._s_tp1.append(d["s_tp1"])
                self._violation_gt.append(d["violation_gt"].astype(np.float32))
                self._offsets.append(self._offsets[-1] + d["s_t"].shape[0])
            self.frame_size = int(np.load(shards[0])["frame_size"])
            self.img_channels = int(np.load(shards[0])["img_channels"])
        elif path.exists():
            d = np.load(path, allow_pickle=True)
            version = int(d.get("version", 1))
            if version < 3:
                raise RuntimeError(
                    f"Cache v{version} found at {path}; "
                    "rebuild with scripts/convert_movi.py to get v3 "
                    "(RGB frames, violation_gt)"
                )
            self._s_tm1 = [d["s_tm1"]]
            self._s_t = [d["s_t"]]
            self._s_tp1 = [d["s_tp1"]]
            self._violation_gt = [d["violation_gt"].astype(np.float32)]
            self._offsets = [0, d["s_t"].shape[0]]
            self.frame_size = int(d["frame_size"])
            self.img_channels = int(d["img_channels"])
        else:
            raise FileNotFoundError(
                f"No cache shards or single-file cache at {path}. "
                "Run scripts/convert_movi.py first."
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (context[2*C,H,W], target[2*C,H,W], violation_gt[1]).

        The two stacked RGB frames are flattened into a single channel dim
        of size 2*C (frame-major: [s_tm1_R, s_tm1_G, s_tm1_B, s_t_R, s_t_G,
        s_t_B]) so the encoder's first Conv2d sees [2*C] channels.
        """
        shard_idx, local_idx = self._shard_for(idx)
        # Normalize RGB to [0,1]. Frames stored as [H, W, C] uint8.
        s_tm1 = torch.from_numpy(
            self._s_tm1[shard_idx][local_idx]
        ).float().div(255.0).permute(2, 0, 1)  # [C, H, W]
        s_t = torch.from_numpy(
            self._s_t[shard_idx][local_idx]
        ).float().div(255.0).permute(2, 0, 1)
        s_tp1 = torch.from_numpy(
            self._s_tp1[shard_idx][local_idx]
        ).float().div(255.0).permute(2, 0, 1)
        # Stack: context = (s_{t-1}, s_t), target = (s_t, s_{t+1}); flatten to
        # [2*C, H, W] (frame-major, channel-minor) for the encoder.
        context = torch.stack([s_tm1, s_t], dim=0).reshape(-1, s_tm1.shape[-2], s_tm1.shape[-1])
        target = torch.stack([s_t, s_tp1], dim=0).reshape(-1, s_t.shape[-2], s_t.shape[-1])
        violation_gt = torch.tensor(
            self._violation_gt[shard_idx][local_idx], dtype=torch.float32
        )
        return context, target, violation_gt


def build_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """Build train/dev/test dataloaders from the configured cache dir.

    The converter emits train/dev shards from MOVi's train+validation splits.
    'test' reuses 'dev' when a dedicated test shard is absent (MOVi-A only
    ships train+validation).
    """
    base = Path(cfg.cache_dir)
    shard = f"{cfg.movi_variant}_{{split}}.npz"

    def make_loader(split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        ds = MoviTransitionDataset(base / shard.format(split=split))
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,  # dataset is in-RAM; forking workers re-loads shards
            pin_memory=True,
            drop_last=drop_last,
        )

    loaders: dict[str, DataLoader] = {
        "train": make_loader("train", shuffle=True, drop_last=True),
        "dev": make_loader("dev", shuffle=False, drop_last=False),
    }
    try:
        loaders["test"] = make_loader("test", shuffle=False, drop_last=False)
    except FileNotFoundError:
        # MOVi-A only ships train+validation; reuse dev as test.
        loaders["test"] = loaders["dev"]
    return loaders
