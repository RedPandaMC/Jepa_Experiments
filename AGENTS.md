# Commands for this project

All commands use `uv` (unified package and Python manager). There is now a
**single Python environment** (Python 3.11) for both data conversion and
training — the previous dual-Python PhyRE setup has been removed.

## Environment Setup

```bash
uv sync
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

## Data Generation (single Python 3.11 env, no TensorFlow)

Dataset: Kubric MOVi-A, pre-rendered tfds shards on
`gs://kubric-public/tfds/movi_a/128x128/1.0.0/`, parsed with the pure-Python
`tfrecord` package + Pillow.

```bash
# Smoke test the parser: download 1 shard, parse 5 videos, emit a tiny npz shard.
uv run python scripts/convert_movi.py --smoke

# (Optional) estimate --force-scale from collision-force percentiles.
uv run python scripts/convert_movi.py --scan-scale --max-shards 20

# Build train cache (downsamples to 64x64). Bound work on an 8GB laptop:
uv run python scripts/convert_movi.py --split train --out-split train \
    --max-shards 50 --force-scale 1.0

# Build dev cache from MOVi's validation split.
uv run python scripts/convert_movi.py --tfds-split validation --out-split dev

ls data/cache/movi_a_*.npz | wc -l
```

## Training & Evaluation

```bash
# Train (library entry point; scripts/train_rdjepa.py was removed earlier).
uv run python -c "from rd_jepa.config import Config; from rd_jepa.train import train; train(Config(exp_name='default', K=15, epochs=20))"

# Render rollout gif (now uses the trained RGB decoder)
# (entry point consolidated into the library — see rd_jepa/viz/gif_writer.py)
uv run aim up
```

## Key Implementation Details

- **One Python version**: uv manages only 3.11 (training AND data generation)
- **No TensorFlow**: MOVi tfds shards parsed with `tfrecord` (pure Python)
- **Cache format v3**: RGB frames `[H,W,3]` uint8 + `violation_gt` float
  (collision-force regression target), replacing PhyRE v2's scene-id maps +
  `solved` bool
- **GPU**: Targets RTX 3070 8GB via AMP + gradient checkpointing
- **VICReg**: Variance/covariance regularization prevents representation collapse
- **Decoder**: RGB (3-channel) jointly trained, produces actual visualizations
- **Action modality removed**: MOVi is passive video; the Lens refines a purely
  visual latent
