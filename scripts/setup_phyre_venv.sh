#!/usr/bin/env bash
# Recreate the in-repo PhyRE (Python 3.9) venv.
#
# PhyRE ships cp36-cp39 wheels only (its C++ simulator bindings are
# ABI-locked), so data generation runs under a separate Python 3.9 venv.
# This script creates it in-repo at .phyre39/ so it survives reboots
# and is not tied to a machine-specific path.
#
# Usage:  bash scripts/setup_phyre_venv.sh
set -euo pipefail

VENV_DIR=".phyre39"

if [ -x "$VENV_DIR/bin/python" ]; then
    echo "PhyRE venv already exists at $VENV_DIR/bin/python"
    "$VENV_DIR/bin/python" -c "import phyre; print('phyre OK')"
    exit 0
fi

echo "Creating PhyRE venv at $VENV_DIR ..."
uv venv --python 3.9 "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" phyre numpy imageio pillow \
    opencv-python-headless tqdm

echo
echo "Done. PhyRE venv ready at $VENV_DIR/bin/python"
echo "Verify with:"
echo "  $VENV_DIR/bin/python scripts/smoke.py"
