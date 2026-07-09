# RD-JEPA

Recurrent Deliberation JEPA — a latent-space world model that uses test-time
compute to iteratively refine physical predictions. POC targeting PhyRE on a
single RTX 3070 laptop (8GB VRAM).

## Quick start

```bash
uv sync                    # install deps
uv run python scripts/smoke.py        # render a PhyRE frame
uv run python scripts/build_cache.py  # pre-roll simulation cache
uv run python scripts/train_rdjepa.py --K 15
uv run python scripts/render_rollout.py --exp default
```

## Layout

```
rd_jepa/         library (config, data, models, losses, viz)
scripts/         entry points (build cache, train, render)
tests/           pytest suite
```

## Commands

- Lint: `uv run ruff check .`
- Tests: `uv run pytest`
- Train: `uv run python scripts/train_rdjepa.py --help`

See `rd-jepa-technical-specification.md` for the full architecture spec.
