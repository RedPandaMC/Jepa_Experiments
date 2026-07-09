"""PyTorch dataset/dataloader for the cached MOVi transitions.

Reads the .npz shards produced by scripts/build_data.py and yields
(context, target, violation_gt) batches on the training device.

Cache v3 format (MOVi): each transition provides RGB frames
s_{t-1}, s_t, s_{t+1} of shape [H, W, 3] uint8 and a float violation_gt
(the normalized collision-force sum in the lookahead window after s_t).
We stack (s_{t-1}, s_t) as the multi-channel context and (s_t, s_{t+1})
as the multi-channel target — both encoders see velocity. Frames are
normalized to float in [0,1] by dividing by 255 (standard RGB).

Shards are loaded **lazily** with an LRU cache: only a few shards live in
RAM at a time, so the dataset scales to large caches (256x256 frames)
without OOM. Shard metadata (offsets, frame_size) is scanned at init
without decompressing the frame arrays.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config


class _ShardCache:
    """LRU cache of decompressed shard arrays.

    Keeps at most ``max_cached`` shards in RAM; evicts the least-recently
    used when full. Each entry holds the four frame arrays + violation_gt.
    """

    def __init__(self, max_cached: int = 4):
        self.max_cached = max_cached
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def get(self, shard_idx: int, loader_fn) -> dict[str, np.ndarray]:
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]
        data = loader_fn()
        self._cache[shard_idx] = data
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return data


class MoviTransitionDataset(Dataset):
    """Dataset over cached (s_{t-1}, s_t, s_{t+1}, violation_gt) transitions.

    Reads the sharded .npz cache (v3) produced by scripts/build_data.py.
    Shards are loaded lazily via an LRU cache so only a handful live in RAM
    at any time — critical for 256x256 caches that would OOM if loaded
    eagerly. Indexing is translated across shards via cumulative offsets.
    """

    def __init__(self, npz_path: str | Path, max_cached_shards: int = 4):
        path = Path(npz_path)
        stem = path.stem  # e.g. "movi_a_train"
        parent = path.parent
        shards = sorted(parent.glob(f"{stem}_shard*.npz"))
        if shards:
            self._shard_paths: list[Path] = shards
            self._offsets: list[int] = [0]
            # Scan metadata only (don't decompress frame arrays).
            for sp in shards:
                d = np.load(sp, allow_pickle=True)
                version = int(d.get("version", 1))
                if version < 3:
                    raise RuntimeError(
                        f"Cache v{version} found at {sp}; "
                        "rebuild with scripts/build_data.py to get v3 "
                        "(RGB frames, violation_gt)"
                    )
                self._offsets.append(
                    self._offsets[-1] + d["s_t"].shape[0]
                )
                del d  # close the npz handle immediately
            self.frame_size = int(
                np.load(shards[0], allow_pickle=True)["frame_size"]
            )
            self.img_channels = int(
                np.load(shards[0], allow_pickle=True)["img_channels"]
            )
        elif path.exists():
            d = np.load(path, allow_pickle=True)
            version = int(d.get("version", 1))
            if version < 3:
                raise RuntimeError(
                    f"Cache v{version} found at {path}; "
                    "rebuild with scripts/build_data.py to get v3 "
                    "(RGB frames, violation_gt)"
                )
            self._shard_paths = [path]
            self._offsets = [0, d["s_t"].shape[0]]
            self.frame_size = int(d["frame_size"])
            self.img_channels = int(d["img_channels"])
        else:
            raise FileNotFoundError(
                f"No cache shards or single-file cache at {path}. "
                "Run scripts/build_data.py first."
            )

        self._cache = _ShardCache(max_cached=max_cached_shards)

    def _load_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        sp = self._shard_paths[shard_idx]

        def loader_fn() -> dict[str, np.ndarray]:
            d = np.load(sp, allow_pickle=True)
            return {
                "s_tm1": d["s_tm1"],
                "s_t": d["s_t"],
                "s_tp1": d["s_tp1"],
                "violation_gt": d["violation_gt"].astype(np.float32),
            }

        return self._cache.get(shard_idx, loader_fn)

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
        shard = self._load_shard(shard_idx)
        # Normalize RGB to [0,1]. Frames stored as [H, W, C] uint8.
        s_tm1 = torch.from_numpy(
            shard["s_tm1"][local_idx]
        ).float().div(255.0).permute(2, 0, 1)  # [C, H, W]
        s_t = torch.from_numpy(
            shard["s_t"][local_idx]
        ).float().div(255.0).permute(2, 0, 1)
        s_tp1 = torch.from_numpy(
            shard["s_tp1"][local_idx]
        ).float().div(255.0).permute(2, 0, 1)
        # Stack: context = (s_{t-1}, s_t), target = (s_t, s_{t+1}); flatten to
        # [2*C, H, W] (frame-major, channel-minor) for the encoder.
        context = torch.stack(
            [s_tm1, s_t], dim=0
        ).reshape(-1, s_tm1.shape[-2], s_tm1.shape[-1])
        target = torch.stack(
            [s_t, s_tp1], dim=0
        ).reshape(-1, s_t.shape[-2], s_t.shape[-1])
        violation_gt = torch.tensor(
            shard["violation_gt"][local_idx], dtype=torch.float32
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
            num_workers=0,  # lazy shard cache; forking would duplicate it
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
