# Commands for this project (run with `uv run ...`)
# Lint
ruff check .
# Tests
pytest
# Smoke test (PhyRE render)
python scripts/smoke.py
# Build simulation cache
python scripts/build_cache.py --tier ball_cross_template --fold 0
# Train
python scripts/train_rdjepa.py --K 15
# Render rollout gif
python scripts/render_rollout.py --exp default
