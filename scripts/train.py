"""Easy training entry point for RD-JEPA v2.

Usage:
    uv run python scripts/train.py                      # defaults
    uv run python scripts/train.py --exp-name big --epochs 40 --batch-size 256
    uv run python scripts/train.py --fast                # 500-sample smoke test

All Config fields can be overridden via CLI flags (kebab-case). Run with
--help to see every option. The full config is printed before training
starts so you can verify what you're about to run.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from rd_jepa.config import Config
from rd_jepa.train import train


def _str2bool(s: str) -> bool:
    return s.lower() in ("true", "1", "yes")


def _str2tuple(s: str) -> tuple[int, ...]:
    """Parse '32,64,128,256' -> (32, 64, 128, 256)."""
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _is_tuple_type(ftype: str) -> bool:
    return "tuple" in ftype or "Sequence" in ftype or "list" in ftype


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for f in dataclasses.fields(Config):
        flag = f"--{f.name.replace('_', '-')}"
        ftype = str(f.type)
        if "bool" in ftype:
            p.add_argument(
                flag, type=_str2bool, default=f.default,
                help=f"Config.{f.name} (bool, default {f.default})",
            )
        elif _is_tuple_type(ftype):
            p.add_argument(
                flag, type=_str2tuple, default=f.default,
                help=f"Config.{f.name} (comma-sep ints, default {f.default})",
            )
        elif "int" in ftype:
            p.add_argument(
                flag, type=int, default=f.default,
                help=f"Config.{f.name} (int, default {f.default})",
            )
        elif "float" in ftype:
            p.add_argument(
                flag, type=float, default=f.default,
                help=f"Config.{f.name} (float, default {f.default})",
            )
        else:
            p.add_argument(
                flag, type=str, default=str(f.default),
                help=f"Config.{f.name} (default {f.default})",
            )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    overrides: dict = {}
    for f in dataclasses.fields(Config):
        raw = getattr(args, f.name)
        if raw is None:
            continue
        ftype = str(f.type)
        if "bool" in ftype:
            overrides[f.name] = bool(raw)
        elif _is_tuple_type(ftype):
            overrides[f.name] = raw  # already a tuple from _str2tuple
        elif "int" in ftype:
            overrides[f.name] = int(raw)
        elif "float" in ftype:
            overrides[f.name] = float(raw)
        elif "Path" in ftype:
            overrides[f.name] = Path(raw)
        else:
            overrides[f.name] = raw

    cfg = Config(**overrides)
    print("=" * 60)
    print("RD-JEPA v2 training run")
    print("=" * 60)
    for f in dataclasses.fields(Config):
        print(f"  {f.name:30s} = {getattr(cfg, f.name)}")
    print("=" * 60)
    train(cfg)


if __name__ == "__main__":
    main()
