# Commands for this project (run with `uv run ...`)
# Lint
ruff check .
# Tests
pytest
# PhyRE venv (Python 3.9, in-repo at .phyre39/) for data generation:
#   bash scripts/setup_phyre_venv.sh   # create it once
#   .phyre39/bin/python scripts/smoke.py
# Data cache is committed under data/cache/ (1.28M transitions). To rebuild:
#   .phyre39/bin/python scripts/build_cache.py --tier ball_cross_template --fold 0
# Train
python scripts/train_rdjepa.py --K 15
# Render rollout gif
python scripts/render_rollout.py --exp default
