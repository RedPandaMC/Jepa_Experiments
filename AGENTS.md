# Commands for this project

All commands use `uv` (unified package and Python manager). There is a
**single Python environment** (Python 3.11) for everything.

## Environment Setup

```bash
uv sync                       # core deps (torch, numpy, tqdm, optuna, mlflow, optuna-dashboard)
uv sync --extra dev           # + pytest, ruff, mypy
uv sync --extra viz           # + matplotlib (animations)
```

## Development

```bash
# Lint
uv run ruff check .
uv run ruff check . --fix    # auto-fix issues

# Tests
uv run pytest

# Type check
uv run mypy experiments/ck_jepa/
```

## Data

Dataset: Jena Climate 2009–2016 (21 weather variables, 10-min resolution,
~420k rows). CC-BY-4.0 from Max Planck Institute for Biogeochemistry.

```bash
# Download + merge semesterly ZIPs into a single CSV
uv run python scripts/download_jena.py

# Verify
wc -l data/jena_climate_2009_2016.csv   # ~420k lines
```

## Training & Evaluation

```bash
# Easy way (recommended): all Config fields are CLI overrides.
uv run python scripts/train.py
uv run python scripts/train.py --exp-name big --epochs 100 --batch-size 512
uv run python scripts/train.py --fast    # 500-sample smoke test

# Direct (library entry point; same thing without argparse):
uv run python -c "from experiments.ck_jepa.config import Config; from experiments.ck_jepa.train import train; train(Config(exp_name='default'))"
```

## Hyperparameter Search (Optuna + MLflow)

```bash
# 50 trials, 20 epochs each, MLflow tracks every trial
uv run python scripts/optuna_search.py --n-trials 50 --epochs 20

# Quick test
uv run python scripts/optuna_search.py --n-trials 3 --epochs 5 --fast

# Custom MLflow URI
uv run python scripts/optuna_search.py --mlflow-uri sqlite:///mlflow.db --exp-name my_search
```

## Dashboards

Both `scripts/train.py` and `scripts/optuna_search.py` print the MLflow UI
and Optuna dashboard launch commands + URLs at the end of a run.

```bash
# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# → http://localhost:5000

# Optuna dashboard
optuna-dashboard sqlite:///optuna.db --port 8080
# → http://localhost:8080
```

## Key Implementation Details

- **Dataset**: Jena Climate 2009–2016 — 21 weather variables, 10-min
  resolution. Rich periodic structure (daily/seasonal cycles) that suits
  the coupled-oscillator inductive bias.
- **GPU**: Targets RTX 3070 8GB via AMP (bf16).
- **Architecture (Resonant Decomposition JEPA)**:
  - **PatchEncoder**: 1D conv patch embedding (patch_len=6, 1-hour patches)
    + adaptive pooling + 2-layer MLP → latent z ∈ R^d (d=512).
  - **AnalyticProjection**: decomposes z into N=32 amplitude-phase mode
    pairs (r_i, φ_i) via learned linear projection + atan2.
  - **ResonatorBank**: K=6 steps of Kuramoto-inspired coupled-oscillator
    dynamics. All parameters (natural frequencies ω, coupling matrix K,
    amplitude relaxation α, equilibria r_eq) are **input-conditioned** —
    predicted from z_0 via small MLPs. This makes test-time compute
    meaningful: different inputs → different oscillator trajectories.
  - **RecombineProjection**: N evolved modes → z_K via
    `Σ r_i * cos(φ_i) * W_i` + LayerNorm.
  - **EMA target encoder**: stop-gradient BYOL-style target.
- **Losses (4 terms)**: JEPA MSE (h_K vs EMA target) + VICReg variance +
  VICReg covariance + **phase diversity** (mean resultant vector
  magnitude — keeps oscillator phases spread on the unit circle).
- **No curriculum K**: training uses fixed K_steps (default 6).
- **Linear forecasting probe**: Linear(d, H*C) trained on frozen h_K
  for cfg.probe_steps, evaluated on val/test. Standard JEPA eval protocol.
- **Optuna + MLflow**: `scripts/optuna_search.py` samples hyperparameters
  (latent_dim, n_modes, K_steps, dt, coupling_sparsity, lr, loss weights,
  etc.) and tracks every trial in MLflow. Both `scripts/train.py` and
  `scripts/optuna_search.py` print the MLflow UI and Optuna dashboard
  launch commands + URLs at the end of a run.

## v5 changes (from v4)

- Config fields removed: everything related to kernels, video, MOVi,
  violations, curriculum K, decoders, viz. All rejected in `__post_init__`.
- Dataset changed: MOVi-A video → Jena Climate CSV.
- Model completely replaced: KernelLens → ResonatorBank (coupled oscillators).
- Losses: 3 terms → 4 terms (added phase diversity loss).
- No reconstruction decoder. Linear forecasting probe only.
- Dependencies dropped: tfrecord, opencv-python, imageio, imageio-ffmpeg,
  pillow, matplotlib. Added: optuna, mlflow, optuna-dashboard.
- Old checkpoints will not load.
