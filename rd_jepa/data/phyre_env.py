"""Locator for the dedicated PhyRE (Python 3.9) virtual environment.

PhyRE only ships prebuilt wheels for cp36-cp39 (its C++ simulator bindings
are ABI-locked). Rather than fighting a source build on Python 3.11, we keep
a separate Python 3.9 venv just for data generation:

    - build_cache.py / smoke.py  -> run under phyre39 (python 3.9 + phyre)
    - training / torch / tests   -> run under the main uv venv (python 3.11)

The phyre39 venv lives outside the project tree so it is not tied to the uv
project. We default to ``/tmp/phyre39`` but allow overriding via the
``PHYRE_VENV`` env var.
"""
import os
from pathlib import Path

DEFAULT_PHYRE_VENV = Path(os.environ.get("PHYRE_VENV", ".phyre39"))


def phyre_venv_python() -> Path:
    """Return the python executable for the phyre 3.9 venv, or error helpfully.

    The venv lives in-repo at ``.phyre39/`` (gitignored) so it survives
    reboots. Recreate it with ``scripts/setup_phyre_venv.sh`` if missing,
    or set PHYRE_VENV to point at an existing 3.9+phyre venv.
    """
    # resolve relative to the repo root (parent of rd_jepa/)
    if not DEFAULT_PHYRE_VENV.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        venv_dir = repo_root / DEFAULT_PHYRE_VENV
    else:
        venv_dir = DEFAULT_PHYRE_VENV
    py = venv_dir / "bin" / "python"
    if not py.exists():
        raise SystemExit(
            f"PhyRE venv not found at {py}.\n"
            f"Recreate it with:\n"
            f"  bash scripts/setup_phyre_venv.sh\n"
            f"Or set PHYRE_VENV to point at an existing 3.9+phyre venv."
        )
    return py
