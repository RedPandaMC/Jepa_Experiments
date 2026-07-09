# Commands for this project

All commands use `uv` (unified package and Python manager).

## Environment Setup

```bash
# Main training environment (Python 3.11, torch, etc.)
uv sync

# PhyRE data generation environment (Python 3.9, ABI-locked C++ bindings)
# This installs Python 3.9 via uv and creates .phyre39/
bash scripts/setup_phyre_venv.sh
```

## Development

```bash
# Lint
uv run ruff check .
uv run ruff check . --fix    # auto-fix issues

# Tests
uv run pytest

# Type check
uv run mypy rd_jepa/
```

## Data Generation (Python 3.9 env)

```bash
# Verify PhyRE works
.phyre39/bin/python scripts/smoke.py

# Build cache (sequential)
.phyre39/bin/python scripts/build_cache.py --tier ball_cross_template --fold 0

# Build cache (parallel - faster, 4 workers recommended for 8GB RAM)
rm -f data/cache/*.npz
for i in {0..3}; do
  .phyre39/bin/python scripts/build_cache.py --tier ball_cross_template --fold 0 \
    --worker-id $i --n-workers 4 &
done
wait
ls data/cache/*.npz | wc -l    # should show 3 shards (train/dev/test)
```

## Training & Evaluation

```bash
# Train
uv run python scripts/train_rdjepa.py --K 15 --epochs 20

# Ablations (sweeps 4 architectural decisions)
uv run python scripts/run_ablations.py --mode oat

# Cross-fold generalization eval
uv run python scripts/eval_cross_fold.py --exp default --folds 0 1 2

# Render rollout gif (now uses trained decoder)
uv run python scripts/render_rollout.py --exp default

# Metrics dashboard
uv run aim up
```

## Key Implementation Details

- **Two Python versions**: uv manages both 3.11 (training) and 3.9 (PhyRE data)
- **Cache format v2**: adds `s_tm1` (velocity context) and `solved` (grounded signal)
- **GPU**: Targets RTX 3070 8GB via AMP + gradient checkpointing
- **VICReg**: Variance/covariance regularization prevents representation collapse
- **Decoder**: Now trained jointly, produces actual visualizations (not random)
