"""Central configuration for RD-JEPA.

All hyperparameters live here so experiments can be swept by CLI overrides.
The ablation knobs (gate, action injection, loss trajectory, latent shape)
are first-class fields to keep the sweep script simple.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Gate(StrEnum):
    SIGMOID = "sigmoid"
    SPARSEMAX = "sparsemax"


class ActionInject(StrEnum):
    ONCE = "once"
    EVERY = "every"


class LossTrajectory(StrEnum):
    FINAL = "final"
    DISCOUNTED = "discounted"


class LatentShape(StrEnum):
    FLAT = "flat"
    SPATIAL = "spatial"


@dataclass
class Config:
    # --- data ---
    tier: str = "ball_cross_template"
    fold: int = 0
    frame_size: int = 64
    cache_dir: Path = Path("data/cache")
    runs_dir: Path = Path("runs")
    seed: int = 42

    # --- model ---
    latent_dim: int = 256
    latent_channels: int = 64  # only used when latent_shape == spatial -> [B, C, 4, 4]
    action_dim: int = 3  # (x, y, r) in [0,1]
    hidden_dim: int = 512
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    gate: Gate = Gate.SIGMOID
    action_inject: ActionInject = ActionInject.EVERY
    latent_shape: LatentShape = LatentShape.FLAT

    # --- deliberation loop ---
    K: int = 15
    early_exit: bool = True
    violation_tau: float = 0.1  # early-exit threshold on V_psi

    # --- training ---
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    amp_dtype: str = "bfloat16"  # bf16 on Ampere
    grad_checkpoint: bool = True
    tbptt_n: int = 5  # detach h every n steps
    ema_decay: float = 0.996
    ema_warmup: int = 100

    # --- loss ---
    loss_trajectory: LossTrajectory = LossTrajectory.DISCOUNTED
    gamma: float = 0.7  # discount for discounted loss

    # --- experiment ---
    exp_name: str = "default"
    fast: bool = False  # 500-sample subset for ablations
    vram_fraction: float = 0.7  # set_per_process_memory_fraction guard

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gate"] = self.gate.value
        d["action_inject"] = self.action_inject.value
        d["loss_trajectory"] = self.loss_trajectory.value
        d["latent_shape"] = self.latent_shape.value
        d["cache_dir"] = str(self.cache_dir)
        d["runs_dir"] = str(self.runs_dir)
        return d

    @property
    def exp_dir(self) -> Path:
        return self.runs_dir / self.exp_name

    @property
    def latent_total_dim(self) -> int:
        """Flat size of the latent used by the loop."""
        if self.latent_shape is LatentShape.SPATIAL:
            return self.latent_channels * 4 * 4
        return self.latent_dim
