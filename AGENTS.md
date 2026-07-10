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
- **v3 architecture (mutating kernel lens)**:
  - **Spatial latent + kernel lens**: the latent is always
    `[B, 64, 4, 4]` (flat 1024 externally). The lens is a `KernelLens` — a
    bank of N=4 depthwise conv kernels `[N, C, 3, 3]` that operate on the
    spatial latent via `unfold` + `einsum` (per-sample depthwise conv with
    per-sample kernels). No MLP-based lenses, no `DivergenceProjection`.
  - **Mutating kernels (test-time compute)**: the kernel state
    `[B, N, C, kH, kW]` is initialized from learned base kernels (seeded
    with Sobel-x, Sobel-y, Laplacian, identity priors) and **evolves
    per-sample** during the K deliberation steps. A mutator network reads the
    pooled latent and produces per-sample kernel deltas at each step — the
    kernels literally transform based on what the latent looks like. This
    makes test-time compute meaningful: different inputs lead to different
    kernel trajectories. The `step_scale` and `mutation_scale` are
    learnable parameters bounded by `tanh`.
  - **Attention gate**: a lightweight gate (LayerNorm + MLP → softmax over N
    kernels) selects which kernels to activate at each step. No MoE router,
    no load-balance loss, no router-entropy loss. A `kernel_diversity_loss`
    (pairwise cosine similarity between base kernels, weight 0.01) prevents
    all kernels from collapsing to identical filters.
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
- **Action modality removed**: MOVi is passive video; the kernel lens refines a
  purely visual latent

## v3 breaking changes (from v2)

- `Config(n_lenses=4, ...)` no longer works — use `n_kernels=4`. The removed
  v2 fields (`n_lenses`, `load_balance_weight`, `router_entropy_weight`) are
  rejected. New fields: `n_kernels`, `kernel_size`, `kernel_diversity_weight`.
- Old `runs/*/ckpt.pt` files (MoE `LensBank` with `lens.lenses.{i}.*` +
  `lens.router.*` keys) **will not load** — `RDJEPA.lens` is now a
  `KernelLens` whose state-dict keys are `lens.base_kernels`,
  `lens.gate.*`, `lens.mutator.*`. Start fresh from a new run.
- `batch_size` reduced from 256 to 128; `grad_checkpoint` now defaults to
  `True` (laptop-friendly). `hidden_dim` reduced from 256 to 128.
- `latent_dim` fixed to 1024 (was incorrectly 512 in v2 despite the comment).
- `VizDecoder.decoder_loss` no longer detaches internally — the caller
  (`train_decoder_step`) detaches `h_K` explicitly. Direct callers of
  `decoder_loss` must pass `h.detach()`.
