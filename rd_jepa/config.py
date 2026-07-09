"""Central configuration for RD-JEPA v2.

Single unified architecture — three core fixes (asynchronous probing
decoder, Navier-Stokes divergence masking, energy/contrastive/curriculum
anti-collapse) are intrinsic, not selectable. There are no ablation knobs
(spatial latent only, divergence mask only, final-only JEPA loss).

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
    frame_size: int = 256  # frames are downsampled to this during conversion
    img_channels: int = 3  # RGB (MOVi is RGB, not PhyRE scene-id maps)
    cache_dir: Path = Path("data/cache")
    runs_dir: Path = Path("runs")
    seed: int = 42

    # --- model (spatial latent is the only path) ---
    # latent is [B, latent_channels, 4, 4] -> flat latent_dim = latent_channels * 16
    latent_channels: int = 64
    latent_dim: int = 1024  # = latent_channels * 4 * 4 (kept explicit for clarity)
    # encoder in_channels = img_channels * 2 (two stacked frames for velocity)
    hidden_dim: int = 512
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)

    # --- deliberation loop (curriculum K_min -> K_max is the only path) ---
    K_min: int = 1
    K_max: int = 15
    curriculum_warmup_epochs: int = 5  # linear ramp K_min -> K_max over these epochs
    early_exit: bool = True
    violation_tau: float = 0.1  # early-exit threshold on V_psi
    latent_layernorm: bool = True  # LayerNorm on encoder output before deliberation

    # --- physics grounding (collision-force regression target) ---
    violation_lookahead: int = 3  # frames ahead (after s_t) to sum collision force
    violation_force_scale: float = 50000.0  # MOVi collision forces are ~1e4-1e5

    # --- training ---
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    amp_dtype: str = "bfloat16"  # bf16 on Ampere
    grad_checkpoint: bool = True
    ema_decay: float = 0.996
    ema_warmup: int = 100

    # --- LR schedule ---
    lr_warmup_steps: int = 500  # linear warmup steps
    lr_cosine: bool = True  # use cosine decay after warmup

    # --- loss weights (all always on; tunable) ---
    violation_weight: float = 0.01
    violation_supervision_weight: float = 0.1
    violation_grounded_weight: float = 0.1
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    energy_weight: float = 0.1
    contrastive_weight: float = 0.05
    divergence_reg_weight: float = 0.05
    contrastive_margin: float = 1.0
    vicreg_target_std: float = 1.0  # target std per dimension for variance loss

    # --- asynchronous probing decoder (separate step is the only path) ---
    decoder_lr: float = 3e-4
    decoder_interval: int = 4  # run decoder step every N JEPA steps (async cadence)
    decoder_weight_decay: float = 0.0

    # --- experiment ---
    exp_name: str = "default"
    fast: bool = False  # 500-sample subset for ablations
    vram_fraction: float = 0.7  # set_per_process_memory_fraction guard

    # Backward-compat alias: Config(K=15, ...) sets K_max. Removed fields
    # (gate, latent_shape, loss_trajectory, gamma, K, tbptt_n) are rejected
    # so stale call sites fail loudly rather than silently misconfiguring.
    def __post_init__(self) -> None:
        # No ablation fields may be re-introduced by accident.
        for forbidden in ("gate", "latent_shape", "loss_trajectory", "gamma",
                          "tbptt_n", "K", "action_dim", "action_inject"):
            if hasattr(self, forbidden):
                raise TypeError(
                    f"Config field '{forbidden}' is removed in v2 (no ablation "
                    "knobs). Use the unified architecture as-is."
                )

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
