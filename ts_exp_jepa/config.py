r"""Configuration for RD-JEPA v5 (Resonant Decomposition JEPA).

All fields are CLI-overridable via ``scripts/train.py`` (kebab-case).
Any field name from removed versions is rejected in ``__post_init__``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # ── data ──────────────────────────────────────────────────────────┐
    dataset_name: str = "jena_climate"
    data_dir: Path = Path("data")
    context_len: int = 144        # 1 day at 10-min resolution
    horizon: int = 72             # 12 hours ahead
    n_features: int = 21
    patch_len: int = 6            # 1-hour patches
    val_ratio: float = 0.25
    test_ratio: float = 0.25
    # ── model ─────────────────────────────────────────────────────────┤
    latent_dim: int = 512
    n_modes: int = 32             # coupled-oscillator modes
    K_steps: int = 6              # resonance steps (fixed, no curriculum)
    dt: float = 0.1               # Euler step size
    coupling_sparsity: float = 0.3
    freq_init_range: tuple[float, float] = (0.1, 2.0)
    amp_init: float = 1.0
    encoder_layers: int = 2
    encoder_hidden: int = 1024
    # ── collapse prevention ────────────────────────────────────────────┤
    vicreg_var_weight: float = 1.5
    vicreg_cov_weight: float = 1.0
    vicreg_target_std: float = 1.0
    phase_div_weight: float = 1.5
    # ── training ──────────────────────────────────────────────────────┤
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    amp_dtype: str = "bfloat16"
    ema_decay: float = 0.996
    ema_warmup: int = 50
    lr_warmup_steps: int = 500
    lr_cosine: bool = True
    num_workers: int = 2
    # ── probe ─────────────────────────────────────────────────────────┤
    probe_steps: int = 100
    probe_lr: float = 1e-3
    # ── logging / checkpoints ──────────────────────────────────────────┤
    runs_dir: Path = Path("runs")
    exp_name: str = "default"
    eval_every_n_epochs: int = 5
    log_every_n_steps: int = 50
    # ── misc ──────────────────────────────────────────────────────────┤
    seed: int = 42
    fast: bool = False
    # ── optuna ────────────────────────────────────────────────────────┤
    optuna_n_trials: int = 50
    optuna_timeout: int = 3600
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # ── rejected field names from prior versions ───────────────────────┐
    _REJECTED: set[str] = field(
        default_factory=lambda: {
            # general
            "K", "gate", "latent_shape", "loss_trajectory", "gamma",
            "tbptt_n", "action_dim", "action_inject", "n_lenses",
            "load_balance_weight", "router_entropy_weight",
            # v3 / v4
            "early_exit", "violation_tau", "violation_weight",
            "violation_supervision_weight", "violation_grounded_weight",
            "energy_weight", "contrastive_weight", "divergence_reg_weight",
            "contrastive_margin", "kernel_diversity_weight",
            "K_min", "K_max", "curriculum_warmup_epochs",
            "n_kernels", "kernel_size", "latent_channels",
            "hidden_dim", "encoder_channels",
            "movi_variant", "movi_resolution", "frame_size", "img_channels",
            "violation_lookahead", "violation_force_scale",
            "grad_checkpoint", "vram_fraction", "max_cached_shards",
            "decoder_lr", "decoder_interval", "decoder_weight_decay",
            "viz_every_n_epochs", "viz_frame_stride", "viz_max_frames",
            "viz_size",
        }
    )

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name.startswith("_"):
                continue
        # reject old field names that were passed as kwargs
        # (works because dataclass raises TypeError for unknown fields
        #  only when not using **kwargs; we handle **kwargs separately)
        if hasattr(self, "_extra_rejected"):
            bad = self._extra_rejected & self._REJECTED
            if bad:
                raise TypeError(
                    f"Config no longer accepts removed fields: {bad}"
                )

    @property
    def exp_dir(self) -> Path:
        return self.runs_dir / self.exp_name

    @property
    def n_patches(self) -> int:
        return self.context_len // self.patch_len

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, Path):
                val = str(val)
            elif isinstance(val, tuple):
                val = list(val)
            result[f.name] = val
        return result


def _check_rejected_kwargs(kwargs: dict[str, object]) -> set[str]:
    """Call from outside to detect rejected kwarg names."""
    rejected = {
        "K", "gate", "latent_shape", "loss_trajectory", "gamma",
        "tbptt_n", "action_dim", "action_inject", "n_lenses",
        "load_balance_weight", "router_entropy_weight",
        "early_exit", "violation_tau", "violation_weight",
        "violation_supervision_weight", "violation_grounded_weight",
        "energy_weight", "contrastive_weight", "divergence_reg_weight",
        "contrastive_margin", "kernel_diversity_weight",
        "K_min", "K_max", "curriculum_warmup_epochs",
        "n_kernels", "kernel_size", "latent_channels",
        "hidden_dim", "encoder_channels",
        "movi_variant", "movi_resolution", "frame_size", "img_channels",
        "violation_lookahead", "violation_force_scale",
        "grad_checkpoint", "vram_fraction", "max_cached_shards",
        "decoder_lr", "decoder_interval", "decoder_weight_decay",
        "viz_every_n_epochs", "viz_frame_stride", "viz_max_frames",
        "viz_size",
    }
    return set(kwargs.keys()) & rejected
