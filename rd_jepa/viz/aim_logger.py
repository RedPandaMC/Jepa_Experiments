r"""Thin Aim logging wrapper.

Aim is an optional experiment dashboard. We wrap it so the training code
stays backend-agnostic and we could swap to TensorBoard or MLflow later
by replacing this file only.

Aim repo lives at `.aim/` under the runs dir by default.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config


class AimLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run = None
        self._available = False
        try:
            from aim import Run  # type: ignore
        except ModuleNotFoundError:
            self._available = False
            self._run_cls = None
        else:
            self._available = True
            self._run_cls = Run

    @property
    def available(self) -> bool:
        return self._available

    def init_run(self) -> None:
        if not self._available or self._run_cls is None:
            self.run = None
            return
        self.run = self._run_cls(
            experiment=self.cfg.exp_name,
            system_tracking_interval=None,
        )
        self.run["config"] = self.cfg.to_dict()  # type: ignore[index]
        aim_dir = Path(self.cfg.runs_dir) / ".aim"
        aim_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(
        self, metrics: dict[str, float], step: int, context: dict | None = None
    ) -> None:
        if self.run is None:
            return
        ctx = context or {}
        for name, value in metrics.items():
            self.run.track(value, name=name, step=step, context=ctx)

    def close(self) -> None:
        if self.run is not None:
            self.run.close()
            self.run = None
