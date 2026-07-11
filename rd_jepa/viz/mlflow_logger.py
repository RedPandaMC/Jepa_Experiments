r"""Thin MLflow logging wrapper.

MLflow is an optional experiment dashboard. We wrap it so the training code
stays backend-agnostic and we could swap to another backend later
by replacing this file only.

MLflow runs are stored under the tracking URI (default ``sqlite:///mlflow.db``)
and viewed with ``mlflow ui``.
"""
from __future__ import annotations

from ..config import Config


class MLflowLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._available = False
        self._mlflow = None
        self._active = False
        try:
            import mlflow  # type: ignore
        except ModuleNotFoundError:
            self._available = False
            self._mlflow = None
        else:
            self._available = True
            self._mlflow = mlflow

    @property
    def available(self) -> bool:
        return self._available

    def init_run(self) -> None:
        if not self._available or self._mlflow is None:
            return
        self._mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        self._mlflow.set_experiment(self.cfg.exp_name)
        self._mlflow.start_run(run_name=self.cfg.exp_name)
        self._active = True
        self._mlflow.log_params(self.cfg.to_dict())

    def log_metrics(
        self, metrics: dict[str, float], step: int, context: dict | None = None
    ) -> None:
        if not self._active or self._mlflow is None:
            return
        ctx = context or {}
        prefix = "/".join(str(v) for v in ctx.values()) if ctx else ""
        items: dict[str, float] = {}
        for name, value in metrics.items():
            key = f"{prefix}/{name}" if prefix else name
            items[key] = value
        self._mlflow.log_metrics(items, step=step)

    def close(self) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.end_run()
            self._active = False
