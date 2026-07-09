"""Central configuration for RD-JEPA.

All hyperparameters live here so experiments can be swept by CLI overrides.
The ablation knobs (gate, loss trajectory, latent shape) are first-class
fields to keep the sweep script simple.

Dataset: Kubric MOVi-A (pre-rendered physics videos, passive — no action
modality). See scripts/convert_movi.py for cache generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Gate(StrEnum):
    SIGMOID = "sigmoid"
    SPARSEMAX = "sparsemax"


class LossTrajectory(StrEnum):
    FINAL = "final"
    DISCOUNTED = "discounted"


class LatentShape(StrEnum):
    FLAT = "flat"
    SPATIAL = "spatial"


@dataclass
class Config:
    # --- data ---
    movi_variant: str = "movi_a"  # one of movi_a/b/c/d/e
    movi_resolution: int = 128  # source resolution to download (128 or 256)
    split: str = "train"  # tfds split name (train/validation/test)
    frame_size: int = 64  # frames are downsampled to this during conversion
    img_channels: int = 3  # RGB (MOVi is RGB, not PhyRE scene-id maps)
    cache_dir: Path = Path("data/cache")
    runs_dir: Path = Path("runs")
    seed: int = 42

    # --- model ---
    latent_dim: int = 256
    latent_channels: int = 64  # only used when latent_shape == spatial -> [B, C, 4, 4]
    # encoder in_channels = img_channels * 2 (two stacked frames for velocity)
    hidden_dim: int = 512
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    gate: Gate = Gate.SIGMOID
    latent_shape: LatentShape = LatentShape.FLAT

    # --- deliberation loop ---
    K: int = 15
    early_exit: bool = True
    violation_tau: float = 0.1  # early-exit threshold on V_psi
    latent_layernorm: bool = True  # LayerNorm on encoder output before deliberation

    # --- violation grounding (collision-force regression target) ---
    violation_lookahead: int = 3  # frames ahead (after s_t) to sum collision force
    violation_force_scale: float = 50000.0  # MOVi collision forces are ~1e4-1e5

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

    # --- LR schedule ---
    lr_warmup_steps: int = 500  # linear warmup steps
    lr_cosine: bool = True  # use cosine decay after warmup

    # --- loss ---
    loss_trajectory: LossTrajectory = LossTrajectory.DISCOUNTED
    gamma: float = 0.7  # discount for discounted loss

    # --- VICReg collapse prevention ---
    vicreg_target_std: float = 1.0  # target std per dimension for variance loss

    # --- experiment ---
    exp_name: str = "default"
    fast: bool = False  # 500-sample subset for ablations
    vram_fraction: float = 0.7  # set_per_process_memory_fraction guard

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gate"] = self.gate.value
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

    @property
    def encoder_in_channels(self) -> int:
        """Number of input channels to the encoder (2 stacked RGB frames)."""
        return self.img_channels * 2
