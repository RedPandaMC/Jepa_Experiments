"""Thin Aim logging wrapper.

Aim is the chosen dashboard (better UI than TensorBoard, native
experiment comparison, image tracking for gifs). We wrap it so the
training code stays backend-agnostic and we could swap to TensorBoard
later by replacing this file only.

Aim repo lives at `.aim/` in the project root by default.
"""
from __future__ import annotations

from pathlib import Path

from aim import Run

from ..config import Config


class AimLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run: Run | None = None

    def init_run(self) -> None:
        self.run = Run(
            experiment=self.cfg.exp_name,
            system_tracking_interval=None,
        )
        self.run["config"] = self.cfg.to_dict()
        # point aim storage at the project runs dir for tidy cleanup
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

    def log_image(
        self,
        name: str,
        image,
        step: int,
        context: dict | None = None,
        caption: str = "",
    ) -> None:
        """Log an image (or animated sequence) to Aim.

        Aim requires its own ``aim.Image`` objects, not raw PIL images. A
        single PIL image is wrapped once; a list of PIL images is wrapped
        per-frame and tracked as a single "Images" sequence, which the Aim
        UI renders as an animated gif (used for the K-step deliberation
        rollout).
        """
        if self.run is None:
            return
        from aim import Image as AimImage

        wrapped: AimImage | list[AimImage]
        if isinstance(image, list):
            wrapped = [AimImage(frame, caption=caption) for frame in image]
        else:
            wrapped = AimImage(image, caption=caption)
        self.run.track(wrapped, name=name, step=step, context=context or {})

    def close(self) -> None:
        if self.run is not None:
            self.run.close()
            self.run = None
