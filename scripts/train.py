#!/usr/bin/env python
r"""Easy CLI entry point for RD-JEPA training.

All Config fields are CLI overrides (kebab-case).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields

from rd_jepa.config import Config, _check_rejected_kwargs
from rd_jepa.train import train
from rd_jepa.viz.dashboards import print_dashboards
from rd_jepa.viz.mlflow_logger import MLflowLogger

_TYPE_MAP: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


def _resolve_type(type_hint: object) -> type:
    """Resolve a (possibly stringified) type hint to a concrete type."""
    if isinstance(type_hint, type):
        return type_hint
    if isinstance(type_hint, str):
        # Handle "int", "float", "str", "bool", "Path", "tuple[...]"
        base = type_hint.split("[")[0].strip(" '\"")
        return _TYPE_MAP.get(base, str)
    return str


def _parse_value(val: str, ftype: type) -> object:
    if ftype is bool:
        return val.lower() in ("true", "1", "yes")
    if ftype is int:
        return int(val)
    if ftype is float:
        return float(val)
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description="RD-JEPA training")
    valid_fields = {f.name: f for f in fields(Config) if not f.name.startswith("_")}

    for name, f in valid_fields.items():
        if name == "fast":
            parser.add_argument("--fast", action="store_true", default=False)
            continue
        kw = name.replace("_", "-")
        parser.add_argument(f"--{kw}", type=str, default=None)

    args = parser.parse_args()

    overrides: dict[str, object] = {}
    for name, f in valid_fields.items():
        val = getattr(args, name)
        if val is None:
            continue
        if name == "fast":
            overrides["fast"] = val
        else:
            rtype = _resolve_type(f.type)
            overrides[name] = _parse_value(val, rtype)

    bad = _check_rejected_kwargs(overrides)
    if bad:
        print(f"Error: rejected removed fields: {bad}", file=sys.stderr)
        sys.exit(1)

    cfg = Config(**overrides)

    logger = MLflowLogger(cfg)
    train(cfg, logger=logger)
    logger.close()

    print_dashboards(mlflow_uri=cfg.mlflow_tracking_uri)


if __name__ == "__main__":
    main()
