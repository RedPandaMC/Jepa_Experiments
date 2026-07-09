"""Evaluation utilities for RD-JEPA."""
from .probe import evaluate_probe, train_violation_probe
from .probe_module import ViolationProbe

__all__ = ["ViolationProbe", "train_violation_probe", "evaluate_probe"]
