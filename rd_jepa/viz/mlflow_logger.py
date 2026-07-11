r"""MLflow logging wrapper.

MLflow is the experiment dashboard for RD-JEPA. Runs are stored under
the tracking URI (default ``sqlite:///mlflow.db``) and viewed with
``mlflow ui``.
"""
from __future__ import annotations

import mlflow

from ..config import Config


class MLflowLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._active = False

    def init_run(self) -> None:
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.exp_name)
        mlflow.start_run(run_name=self.cfg.exp_name)
        self._active = True
        mlflow.log_params(self.cfg.to_dict())

    def log_metrics(
        self, metrics: dict[str, float], step: int, context: dict | None = None
    ) -> None:
        if not self._active:
            return
        ctx = context or {}
        prefix = "/".join(str(v) for v in ctx.values()) if ctx else ""
        items: dict[str, float] = {}
        for name, value in metrics.items():
            key = f"{prefix}/{name}" if prefix else name
            items[key] = value
        mlflow.log_metrics(items, step=step)

    def close(self) -> None:
        if self._active:
            mlflow.end_run()
            self._active = False
