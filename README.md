# RD-JEPA

Recurrent Deliberation JEPA — a latent-space world model that uses test-time
compute to iteratively refine physical predictions. POC targeting PhyRE on a
single RTX 3070 laptop (8GB VRAM).

## What's New (v2)

- **Two-frame input**: Model now sees velocity via stacked frames `(s_{t-1}, s_t)`, fixing the non-Markovian prediction target
- **Grounded supervision**: Violation head trained on PhyRE's `solved` flag, not just latent error
- **Collapse prevention**: VICReg variance/covariance regularization + diagnostics (effective rank, cosine similarity)
- **Linear probe eval**: Downstream `solved/unsolved` classification validates representation quality
- **AUCCESS metric**: Action ranking evaluation for physical reasoning benchmark
- **LR scheduling**: Warmup + cosine decay
- **LayerNorm**: On encoder output for training stability
- **Trained viz decoder**: Gifs now show actual reconstructions, not random noise

## Quick start

```bash
# Setup (both envs managed by uv)
uv sync                              # main training env (Python 3.11)
bash scripts/setup_phyre_venv.sh     # PhyRE data env (Python 3.9)
.phyre39/bin/python scripts/smoke.py # verify PhyRE works

# Build data cache (if needed)
rm -f data/cache/*.npz
for i in {0..3}; do
  .phyre39/bin/python scripts/build_cache.py --tier ball_cross_template --fold 0 \
    --worker-id $i --n-workers 4 &
done
wait

# Train
uv run python scripts/train_rdjepa.py --K 15

# Evaluate
uv run python scripts/eval_cross_fold.py --exp default --folds 0 1 2

# Visualize
uv run python scripts/render_rollout.py --exp default
uv run aim up
```

## Unified uv Workflow

Both environments are managed by uv:
- **Main env** (`uv sync`): Python 3.11, PyTorch, training code
- **PhyRE env** (`.phyre39/`): Python 3.9, PhyRE simulator, data generation

```bash
# Install Python 3.9 for PhyRE env
uv python install 3.9

# Recreate PhyRE env
bash scripts/setup_phyre_venv.sh
```

## Layout

```
rd_jepa/         library (config, data, models, losses, viz, eval)
scripts/         entry points (train, render, ablations, eval, cache)
tests/           pytest suite + collapse regression tests
data/cache/      PhyRE transition cache (v2 format with s_tm1, solved)
docs/            architecture diagrams
```

## Commands

```bash
# Development
uv run ruff check .          # lint
uv run ruff check . --fix    # auto-fix
uv run pytest                # tests
uv run mypy rd_jepa/         # type check

# Training
uv run python scripts/train_rdjepa.py --K 15 --epochs 20
uv run python scripts/run_ablations.py --mode oat

# Evaluation
uv run python scripts/eval_cross_fold.py --exp default --folds 0 1 2

# Visualization
uv run python scripts/render_rollout.py --exp default
uv run aim up                # metrics dashboard
```

## Architecture Highlights

- **Lens Paradigm**: Shared refinement function applied K times, constant VRAM
- **Early exit**: Dynamic deliberation depth via violation threshold
- **JEPA training**: Latent prediction with EMA target encoder, no pixel decode
- **Two-channel input**: Velocity-aware via frame stacking

See `rd-jepa-technical-specification.md` for full details.
