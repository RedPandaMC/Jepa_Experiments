"""Central configuration for RD-JEPA v4 (simplified kernel lens).

The lens bank is a set of mutating depthwise conv kernels that operate
on the spatial latent and evolve per-sample during the K deliberation steps.

All hyperparameters live here so experiments can be swept by CLI overrides.

Dataset: Kubric MOVi-A (pre-rendered physics videos, passive — no action
modality). See scripts/build_data.py for cache generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # --- data ---
    movi_variant: str = "movi_a"  # one of movi_a/b/c/d/e
    movi_resolution: int = 128  # source resolution to download (128 or 256)
    split: str = "train"  # tfds split name (train/validation/test)
    frame_size: int = 128  # frames are downsampled to this during conversion
    img_channels: int = 3  # RGB (MOVi is RGB, not PhyRE scene-id maps)
    cache_dir: Path = Path("data/cache")
    runs_dir: Path = Path("runs")
    seed: int = 42

    # --- model (spatial latent, flat externally) ---
    # latent is [B, latent_channels, 4, 4] -> flat latent_dim = latent_channels * 16
    latent_channels: int = 64
    latent_dim: int = 1024  # = latent_channels * 4 * 4
    # encoder in_channels = img_channels * 2 (two stacked frames for velocity)
    hidden_dim: int = 64
    encoder_channels: tuple[int, ...] = (16, 32, 64, 128)

    # --- kernel lens ---
    # N depthwise conv kernels that mutate per-sample during K steps.
    n_kernels: int = 4
    kernel_size: int = 3

    # --- deliberation loop (curriculum K_min -> K_max) ---
    K_min: int = 1
    K_max: int = 3
    curriculum_warmup_epochs: int = 3  # linear ramp K_min -> K_max over these epochs
    latent_layernorm: bool = True  # LayerNorm on encoder output before deliberation

    # --- physics grounding (collision-force regression target) ---
    violation_lookahead: int = 3  # frames ahead (after s_t) to sum collision force
    violation_force_scale: float = 50000.0  # MOVi collision forces are ~1e4-1e5

    # --- training ---
    batch_size: int = 64  # fits 8GB VRAM at 128x128 with headroom
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 10
    amp_dtype: str = "bfloat16"  # bf16 on Ampere
    grad_checkpoint: bool = False  # model is small enough without it
    ema_decay: float = 0.996
    ema_warmup: int = 50

    # --- LR schedule ---
    lr_warmup_steps: int = 500  # linear warmup steps
    lr_cosine: bool = True  # use cosine decay after warmup

    # --- loss weights (simplified: JEPA + VICReg only) ---
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    vicreg_target_std: float = 1.0  # target std per dimension for variance loss

    # --- asynchronous probing decoder (separate step) ---
    decoder_lr: float = 3e-4
    decoder_interval: int = 4  # run decoder step every N JEPA steps (async cadence)
    decoder_weight_decay: float = 0.0

    # --- data loader ---
    num_workers: int = 4  # parallel shard decompression
    max_cached_shards: int = 2  # LRU cache size per worker

    # --- experiment ---
    exp_name: str = "default"
    fast: bool = False  # 500-sample subset for ablations
    vram_fraction: float = 0.90  # leave headroom on 8GB cards

    # --- visualization ---
    viz_every_n_epochs: int = 5  # render gifs every N epochs (default: sparse)
    viz_frame_stride: int = 2  # decode every Nth latent step for GIFs
    viz_max_frames: int = 4  # cap GIF length to keep render cost down
    viz_size: int = 192  # smaller output size for cheaper rendering

    # Removed v3 fields (MoE lens bank, trajectory losses, early exit) and
    # v2 fields are rejected so stale call sites fail loudly.
    def __post_init__(self) -> None:
        for forbidden in (
            "gate", "latent_shape", "loss_trajectory", "gamma",
            "tbptt_n", "K", "action_dim", "action_inject",
            "n_lenses", "load_balance_weight", "router_entropy_weight",
            "early_exit", "violation_tau", "violation_weight",
            "violation_supervision_weight", "violation_grounded_weight",
            "energy_weight", "contrastive_weight", "divergence_reg_weight",
            "contrastive_margin", "kernel_diversity_weight",
        ):
            if hasattr(self, forbidden):
                raise TypeError(
                    f"Config field '{forbidden}' is removed. The architecture "
                    "now uses JEPA + VICReg only (no trajectory losses, no "
                    "early exit). Use the simplified config as-is."
                )
        if self.n_kernels < 1:
            raise ValueError("n_kernels must be >= 1")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd (for symmetric padding)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        d["runs_dir"] = str(self.runs_dir)
        return d

    @property
    def exp_dir(self) -> Path:
        return self.runs_dir / self.exp_name

    @property
    def latent_total_dim(self) -> int:
        """Flat size of the spatial latent used by the deliberation MLPs."""
        return self.latent_channels * 4 * 4

    @property
    def encoder_in_channels(self) -> int:
        """Number of input channels to the encoder (2 stacked RGB frames)."""
        return self.img_channels * 2

    def resolve_K(self, epoch: int) -> int:
        """Linear curriculum schedule K_min -> K_max over warmup epochs."""
        if self.curriculum_warmup_epochs <= 0:
            return self.K_max
        progress = min(epoch / self.curriculum_warmup_epochs, 1.0)
        k = int(round(self.K_min + (self.K_max - self.K_min) * progress))
        return max(self.K_min, min(self.K_max, k))
