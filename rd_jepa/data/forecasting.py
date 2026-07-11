r"""Jena Climate 2009-2016 dataset and data loaders.

The Jena climate dataset records 14 weather variables at 10-minute resolution
from the Max Planck Institute for Biogeochemistry in Jena, Germany.
CC-BY-4.0 license. Source: https://www.bgc-jena.mpg.de/wetter/

Cache format: a single CSV at ``data/jena_climate_2009_2016.csv`` with a
header row and ~420k rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config


class JenaClimateDataset(Dataset):
    """Sliding-window dataset for Jena Climate.

    Each item is a (context, target) pair where context is the past
    ``context_len`` timesteps and target is the next ``horizon`` timesteps.
    Both are float32 tensors of shape [L, C] / [H, C] with features
    normalized using training-split statistics.
    """

    def __init__(
        self,
        csv_path: Path | str,
        context_len: int = 144,
        horizon: int = 72,
        n_features: int = 14,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25,
        split: str = "train",
        normalize: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.context_len = context_len
        self.horizon = horizon
        self.n_features = n_features
        self.split = split

        data = np.loadtxt(
            str(self.csv_path),
            delimiter=",",
            skiprows=1,
            usecols=range(1, n_features + 1),
            dtype=np.float32,
        )
        n_total = data.shape[0]

        n_train = int(n_total * (1.0 - val_ratio - test_ratio))
        n_val = int(n_total * val_ratio)

        train_data = data[:n_train]

        self.mean = train_data.mean(axis=0)
        self.std = train_data.std(axis=0) + 1e-8

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = data[n_train : n_train + n_val]
        elif split == "test":
            self.data = data[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")

        if normalize:
            self.data = (self.data - self.mean) / self.std

        self.n_windows = len(self.data) - context_len - horizon + 1

    def __len__(self) -> int:
        return max(0, self.n_windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (context [L, C], target [H, C])."""
        start = idx
        ctx_end = start + self.context_len
        tgt_end = ctx_end + self.horizon

        context = torch.from_numpy(self.data[start:ctx_end].copy())
        target = torch.from_numpy(self.data[ctx_end:tgt_end].copy())

        return context, target


def build_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from cfg."""
    csv_path = cfg.data_dir / "jena_climate_2009_2016.csv"

    train_ds = JenaClimateDataset(
        csv_path,
        context_len=cfg.context_len,
        horizon=cfg.horizon,
        n_features=cfg.n_features,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        split="train",
    )
    val_ds = JenaClimateDataset(
        csv_path,
        context_len=cfg.context_len,
        horizon=cfg.horizon,
        n_features=cfg.n_features,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        split="val",
    )
    test_ds = JenaClimateDataset(
        csv_path,
        context_len=cfg.context_len,
        horizon=cfg.horizon,
        n_features=cfg.n_features,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        split="test",
    )

    if cfg.fast:
        n_train = cfg.context_len + cfg.horizon + 500
        n_eval = cfg.context_len + cfg.horizon + 200
        train_ds.data = train_ds.data[:n_train]
        train_ds.n_windows = 500
        val_ds.data = val_ds.data[:n_eval]
        val_ds.n_windows = 200
        test_ds.data = test_ds.data[:n_eval]
        test_ds.n_windows = 200

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
