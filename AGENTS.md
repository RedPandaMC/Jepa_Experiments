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
uv run python -c "from rd_jepa.config import Config; from rd_jepa.train import train; train(Config(exp_name='default', K_max=15, epochs=20))"

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
- **v2 architecture (three core fixes, all on by default)**:
  - **Spatial latent + Navier-Stokes masking**: the latent is always
    `[B, 64, 4, 4]` (flat 1024 for the MLPs). The lens subtractive phase is a
    `DivergenceProjection` (Sobel divergence + learned per-sample projection
    scalar + L2 mass renormalization) — the CFD incompressibility projection,
    redistributing density rather than zeroing overlap. No Sigmoid/Sparsemax
    gate, no FLAT latent path.
  - **Anti-collapse losses**: Latent Energy Conservation
    (`|‖h_K‖−‖h_0‖|²`), Contrastive Dynamics (margin loss penalizing stasis
    when `violation_gt > 0`), Divergence Regularization (per-step latent-mass
    stability). Final-only JEPA loss (no discounted trajectory).
  - **Curriculum K**: per-epoch linear ramp `K_min → K_max` over
    `curriculum_warmup_epochs`; both `train_step` and `eval_step` use the
    epoch's `K_epoch` (`cfg.resolve_K(epoch)`).
  - **Asynchronous probing decoder**: `VizDecoder` is trained in its own
    optimizer + backward pass on a **detached** `h_K`, every
    `cfg.decoder_interval` (default 4) JEPA steps — zero gradient
    entanglement with the thinking loop.
- **Action modality removed**: MOVi is passive video; the Lens refines a purely
  visual latent

## v2 breaking changes (from v1)

- `Config(K=15, ...)` no longer works — use `K_max=15`. The removed ablation
  fields (`gate`, `latent_shape`, `loss_trajectory`, `gamma`, `tbptt_n`, `K`)
  are rejected; the single unified architecture has no toggleable paths.
- Old `runs/*/ckpt.pt` files (FLAT latent + sigmoid gate) **will not load**
  into the v2 model — the `DeliberationStep` now contains `DivergenceProjection`
  params (Sobel buffers, `mlp_alpha`) and the encoder is spatial-only. Start
  fresh from a new run.
- `VizDecoder.decoder_loss` no longer detaches internally — the caller
  (`train_decoder_step`) detaches `h_K` explicitly. Direct callers of
  `decoder_loss` must pass `h.detach()`.
