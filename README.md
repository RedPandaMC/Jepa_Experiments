# RD-JEPA

**Resonant Decomposition JEPA** — a coupled-oscillator latent world model for
multivariate time-series forecasting. Built for consumer GPUs (8GB VRAM,
tested on RTX 3070). Trained on Jena Climate 2009–2016 (21 weather
variables, 10-min resolution, ~420k rows).

---

## How it works

RD-JEPA decomposes a latent representation into N amplitude-phase mode pairs
and evolves them through K steps of Kuramoto-inspired coupled-oscillator
dynamics. The oscillator parameters (natural frequencies ω, coupling matrix K,
amplitude relaxation α, equilibria r_eq) are all **input-conditioned** —
predicted from the initial latent via small MLPs. This makes test-time compute
meaningful: different inputs produce different oscillator trajectories.

The evolved modes are recombined and compared (JEPA-style) to an EMA target
encoder's representation of the future window. A linear forecasting probe
evaluates representation quality on held-out data.

### Key ideas

- **Analytic projection**: latent z ∈ R^d → N complex mode pairs (r_i, φ_i)
  via learned linear projection + atan2
- **Coupled-oscillator dynamics**: K steps of Kuramoto phase synchronization
  + amplitude relaxation — emergent structure from periodic inductive bias
- **Input-conditioned parameters**: ω, K, α, r_eq all predicted from z_0
- **Phase diversity loss**: keeps oscillator phases spread on the unit circle
  (mean resultant vector magnitude), preventing mode collapse

## Architecture

**PatchEncoder**: 1D conv patch embedding (patch_len=6, 1-hour patches) +
adaptive pooling + 2-layer MLP → latent z ∈ R^d (d=256).

**AnalyticProjection**: decomposes z into N=32 amplitude-phase mode pairs
(r_i, φ_i) via learned linear projection + atan2.

**ResonatorBank**: K=6 steps of Kuramoto-inspired coupled-oscillator dynamics.
All parameters (natural frequencies ω, coupling matrix K, amplitude
relaxation α, equilibria r_eq) are input-conditioned — predicted from z_0
via small MLPs.

**RecombineProjection**: N evolved modes → z_K via
`Σ r_i * cos(φ_i) * W_i` + LayerNorm.

**EMA target encoder**: stop-gradient BYOL-style target.

## Losses

| Loss | Default weight | Purpose |
|---|---|---|
| JEPA MSE | 1.0 | predict EMA target latent |
| VICReg variance | 1.0 | prevent per-dimension collapse |
| VICReg covariance | 1.0 | decorrelate latent dimensions |
| Phase diversity | 0.5 | keep oscillator phases spread |

## Getting started

### Setup

```bash
uv sync                    # Python 3.11 (torch + numpy + tqdm)
uv sync --extra dev        # + pytest, ruff, mypy
uv sync --extra optuna     # + optuna, mlflow (hyperparameter search)
```

### Data

Download Jena Climate 2009–2016 (21 weather variables, 10-min resolution,
~420k rows, CC-BY-4.0):

```bash
uv run python scripts/download_jena.py
wc -l data/jena_climate_2009_2016.csv   # ~420k lines
```

### Train

```bash
uv run python scripts/train.py                         # defaults
uv run python scripts/train.py --exp-name big --epochs 100 --batch-size 512
uv run python scripts/train.py --fast                   # 500-sample smoke test
```

All Config fields are CLI overrides (kebab-case). Run `--help` to see options.

Metrics go to MLflow (when installed); the launch command + URL are printed at
the end of every run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# → http://localhost:5000
```

### Hyperparameter search

```bash
uv run python scripts/optuna_search.py --n-trials 50 --epochs 20
uv run python scripts/optuna_search.py --n-trials 3 --epochs 5 --fast   # quick
```

MLflow tracks every trial; the MLflow UI and Optuna dashboard launch
commands + URLs are printed at the end of every run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # → http://localhost:5000
optuna-dashboard sqlite:///optuna.db --port 8080                   # → http://localhost:8080
```

### Develop

```bash
uv run ruff check . --fix   # lint + fix
uv run pytest               # 36 tests, CPU-only
uv run mypy rd_jepa/        # type check
```

## Project structure

```
rd_jepa/
  config.py                Config dataclass (v5)
  losses.py                JEPA + VICReg + phase diversity
  train.py                 train_step / eval_step / train
  data/forecasting.py      Jena Climate dataset + loaders
  models/
    rd_jepa.py             RDJEPA: encoder → resonator → recombine
    patch_encoder.py       1D patch embedding + MLP
    resonator.py           AnalyticProjection + ResonatorBank + RecombineProjection
    ema.py                 EMA target encoder
  eval/forecast_probe.py   Linear forecasting probe
  viz/mlflow_logger.py     MLflow logging wrapper
  viz/dashboards.py        MLflow + Optuna dashboard launch helpers
scripts/
  download_jena.py         Jena Climate downloader
  train.py                 easy CLI entry point
  optuna_search.py         Optuna + MLflow hyperparameter search
tests/
  test_rd_jepa_pipeline.py  36 tests
data/                      CSV data
runs/                      checkpoints + MLflow logs
```

## License

See [LICENSE](LICENSE).
