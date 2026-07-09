# RD-JEPA

Recurrent Deliberation JEPA — a latent-space physics world model that uses
test-time compute to iteratively refine physical predictions. POC targeting
the Kubric MOVi-A dataset on a single RTX 3070 laptop (8GB VRAM).

## What's New (v3)

- **New dataset: Kubric MOVi-A**. Pre-rendered physics videos (rigid-body
  collisions of CLEVR-style shapes) downloaded from `gs://kubric-public/tfds`.
  This **removes the dual-Python-version problem**: the entire pipeline
  (download, convert, train) now runs in a single Python 3.11 env. No more
  `.phyre39/` venv, no ABI-locked C++ simulator bindings.
- **Pure-Python data ingestion**: MOVi's tfds shards are plain
  `tf.Example`/`.tfrecord` records parsed with the `tfrecord` PyPI package +
  Pillow. **No TensorFlow dependency anywhere.**
- **RGB input**: encoder now consumes stacked RGB frames `(s_{t-1}, s_t)` →
  `in_channels = 6` (was 2 for PhyRE scene-id maps).
- **Action modality removed**: MOVi is passive video (no ball-drop action),
  so the Action Encoder and action-injection ablation have been removed. The
  Lens now refines a purely visual latent.
- **Grounded violation supervision via collisions**: the Violation Head is now
  trained (regression, smooth-L1) against a target derived from MOVi's
  per-frame `events.collisions.force` — a genuine physics quantity instead of
  PhyRE's binary `solved` flag.
- **Linear probe** repurposed to predict the collision-force violation target
  (MSE/R²) instead of solved/unsolved classification. The PhyRE-only AUCCESS
  ranking metric has been removed.

## Quick start

```bash
# Single env for everything (Python 3.11 + torch + tfrecord)
uv sync

# Smoke test the data pipeline: download 1 MOVi-A shard, parse 5 videos,
# emit a tiny npz shard. (Validates parsing without a full download.)
uv run python scripts/convert_movi.py --smoke

# (Optional) estimate a good --force-scale before the full run:
uv run python scripts/convert_movi.py --scan-scale --max-shards 20

# Build the train cache (downsamples to 64x64). --max-shards N / --max-videos M
# to bound work on an 8GB laptop for a first pass.
uv run python scripts/convert_movi.py --split train --out-split train --force-scale 1.0

# Build the dev cache from MOVi's validation split.
uv run python scripts/convert_movi.py --tfds-split validation --out-split dev

# Train (library entry point)
uv run python -c "from rd_jepa.config import Config; from rd_jepa.train import train; train(Config(exp_name='default'))"

# Metrics dashboard
uv run aim up
```

## Unified uv Workflow

One environment, one Python version, one ML framework (PyTorch):

- **Main env** (`uv sync`): Python 3.11, PyTorch, `tfrecord`, Pillow. Used for
  data conversion AND training. No second venv, no TensorFlow.

```bash
uv sync
```

## Layout

```
rd_jepa/         library (config, data, models, losses, viz, eval)
scripts/         convert_movi.py (data cache builder)
tests/           pytest suite + collapse regression tests
data/cache/      MOVi transition cache (v3 format: RGB + violation_gt)
data/cache/_raw/ transient raw tfrecord shards (gitignored)
docs/            architecture diagrams
```

## Commands

```bash
# Development
uv run ruff check .          # lint
uv run ruff check . --fix    # auto-fix
uv run pytest                # tests
uv run mypy rd_jepa/         # type check

# Data
uv run python scripts/convert_movi.py --smoke
uv run python scripts/convert_movi.py --scan-scale --max-shards 20
uv run python scripts/convert_movi.py --split train --out-split train
uv run python scripts/convert_movi.py --tfds-split validation --out-split dev

# Training / Viz
uv run aim up                # metrics dashboard
```

## Architecture Highlights

- **Lens Paradigm**: Shared refinement function applied K times, constant VRAM
- **Early exit**: Dynamic deliberation depth via violation threshold
- **JEPA training**: Latent prediction with EMA target encoder, no pixel decode
- **Two-frame input**: Velocity-aware via frame stacking (now RGB)
- **Physics-grounded violation**: collision-force regression target

See `rd-jepa-technical-specification.md` for full details.
