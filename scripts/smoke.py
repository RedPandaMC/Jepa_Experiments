#!/usr/bin/env python
"""Render a single PhyRE frame to PNG to verify the data pipeline.

This script runs under the dedicated Python 3.9 + phyre venv (see
rd_jepa/data/phyre_env.py). It is intentionally import-light so it can be
run directly with: <phyre39>/bin/python scripts/smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import phyre
from PIL import Image


def main() -> int:
    train, dev, test = phyre.get_fold("ball_cross_template", 0)
    print(f"fold 0 ball_cross_template: train={len(train)} dev={len(dev)} test={len(test)}")
    print(f"first task: {train[0]}")

    sim = phyre.initialize_simulator(train[:1], "ball")
    rng = np.random.default_rng(42)
    saved = 0
    for i in range(10):
        action = rng.random(3)
        res = sim.simulate_action(0, action, need_images=True, stride=10)
        status = res.status
        img_shape = None if res.images is None else res.images.shape
        print(f"  action {i}: status={status.name} images={img_shape}")
        if res.images is not None:
            out_dir = Path("runs/smoke")
            out_dir.mkdir(parents=True, exist_ok=True)
            rgb0 = phyre.observations_to_uint8_rgb(res.images[0])
            rgb_last = phyre.observations_to_uint8_rgb(res.images[-1])
            Image.fromarray(rgb0).save(out_dir / "frame_first.png")
            Image.fromarray(rgb_last).save(out_dir / "frame_last.png")
            saved += 1
            if saved >= 1:
                break

    if saved == 0:
        print("ERROR: no valid simulations produced images", file=sys.stderr)
        return 1

    print("OK — wrote runs/smoke/frame_first.png and frame_last.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
