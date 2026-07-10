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
# Easy way (recommended): builds train + dev with recommended defaults.
uv run python scripts/build_data.py
uv run python scripts/build_data.py --max-shards 10   # quick test
uv run python scripts/build_data.py --dev-only         # just dev split
uv run python scripts/build_data.py --scan-scale        # tune force-scale

# Direct (full control): see scripts/build_data.py --help
uv run python scripts/build_data.py --tfds-split train --out-split train \
    --max-shards 50 --force-scale 1.0
uv run python scripts/build_data.py --tfds-split validation --out-split dev

ls data/cache/movi_a_*.npz | wc -l
```

## Training & Evaluation

```bash
# Easy way (recommended): all Config fields are CLI overrides.
uv run python scripts/train.py
uv run python scripts/train.py --exp-name big --epochs 40 --batch-size 256
uv run python scripts/train.py --fast    # 500-sample smoke test

# Direct (library entry point; same thing without argparse):
uv run python -c "from rd_jepa.config import Config; from rd_jepa.train import train; train(Config(exp_name='default', K_max=15, epochs=100))"
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
  - **MoE Lens Bank**: `RDJEPA.lens` is a `LensBank` of N=4 specialist
    `DeliberationStep`s + a soft-routing router (MLP → softmax gate
    `[B,N]` per step). Each lens keeps its own `mlp_add` +
    `DivergenceProjection` (per-lens `mlp_alpha`); the fixed Sobel kernels
    are identical across lenses. The bank mixes the N lens deltas by the
    gate, mass-renormalizes to `‖mean_i delta_i‖`, and applies `tanh`.
    `load_balance_loss` (MoE uniform-usage, weight 0.01) +
    `router_entropy_loss` (entropy bonus, weight 0.005) prevent
    degenerate routing. `n_lenses=1` disables routing (single-lens path,
    `gate=None`, no router in the graph).
- **Action modality removed**: MOVi is passive video; the Lens Bank refines a
  purely visual latent

## v2 breaking changes (from v1)

- `Config(K=15, ...)` no longer works — use `K_max=15`. The removed ablation
  fields (`gate`, `latent_shape`, `loss_trajectory`, `gamma`, `tbptt_n`, `K`)
  are rejected; the single unified architecture has no toggleable paths.
- Old `runs/*/ckpt.pt` files (single `lens.*` keys) **will not load** into
  the lens-bank model — `RDJEPA.lens` is now a `LensBank` whose state-dict
  keys are `lens.lenses.{i}.*` + `lens.router.*`. Start fresh from a new run.
  (`n_lenses=1` gives the same single-lens behavior but still has the
  `lens.lenses.0.` prefix, so it's a key-path change even at N=1.)
- `VizDecoder.decoder_loss` no longer detaches internally — the caller
  (`train_decoder_step`) detaches `h_K` explicitly. Direct callers of
  `decoder_loss` must pass `h.detach()`.
