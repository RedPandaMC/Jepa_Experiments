# RD-JEPA

Recurrent Deliberation JEPA — a latent-space world model that uses test-time
compute to iteratively refine physical predictions. POC targeting PhyRE on a
single RTX 3070 laptop (8GB VRAM).

## Quick start

```bash
uv sync                              # install main deps (torch+cu121, aim, ...)
bash scripts/setup_phyre_venv.sh     # create the in-repo PhyRE 3.9 venv (.phyre39/)
.phyre39/bin/python scripts/smoke.py # render a PhyRE frame (verifies the venv)

# Data cache (1.28M transitions) is committed under data/cache/ — no rebuild
# needed. To rebuild from scratch:
#   .phyre39/bin/python scripts/build_cache.py --tier ball_cross_template --fold 0

uv run python scripts/train_rdjepa.py --K 15      # train
uv run python scripts/render_rollout.py --exp default  # render gifs
uv run aim up                                     # dashboard
```

## Why two Python environments?

PhyRE ships prebuilt wheels only for cp36–cp39 (its C++ simulator bindings
are ABI-locked). Training runs on Python 3.11 (the main uv venv with torch);
data generation runs under a separate in-repo Python 3.9 venv at `.phyre39/`
(gitignored, ~400MB). `scripts/setup_phyre_venv.sh` recreates it.

## Layout

```
rd_jepa/         library (config, data, models, losses, viz)
scripts/         entry points (setup venv, build cache, train, render, ablations)
tests/           pytest suite
data/cache/      committed PhyRE transition cache (1.28M pairs, ~105MB)
.phyre39/        PhyRE 3.9 venv (gitignored)
```

## Commands

- Lint: `uv run ruff check .`
- Tests: `uv run pytest`
- Train: `uv run python scripts/train_rdjepa.py --help`
- Ablations: `uv run python scripts/run_ablations.py --mode oat`

See `rd-jepa-technical-specification.md` for the full architecture spec.
