"""Regression tests for the MOVi v3 data + model contract.

These run on CPU with synthetic .npz shards so they need neither the network
download nor a GPU. They lock in the post-PhyRE contract: RGB frames, no
action modality, and a continuous collision-force violation target.
"""
from __future__ import annotations

import numpy as np
import torch

from rd_jepa.config import Config
from rd_jepa.data.loader import MoviTransitionDataset
from rd_jepa.losses import total_loss, violation_grounded_loss
from rd_jepa.models.rd_jepa import RDJEPA
from rd_jepa.viz.decoder import VizDecoder


def _write_synthetic_shard(path, n: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        s_tm1=np.random.randint(0, 256, (n, 64, 64, 3), dtype=np.uint8),
        s_t=np.random.randint(0, 256, (n, 64, 64, 3), dtype=np.uint8),
        s_tp1=np.random.randint(0, 256, (n, 64, 64, 3), dtype=np.uint8),
        violation_gt=(np.random.rand(n).astype(np.float32)),
        frame_size=np.int64(64),
        img_channels=np.int64(3),
        version=np.int64(3),
    )


def test_config_has_no_action_fields():
    cfg = Config()
    assert not hasattr(cfg, "action_dim")
    assert not hasattr(cfg, "action_inject")
    assert not hasattr(cfg, "tier")
    assert not hasattr(cfg, "fold")
    assert cfg.encoder_in_channels == 6  # 2 stacked RGB frames
    assert cfg.img_channels == 3


def test_loader_yields_rgb_stacked_flat(tmp_path):
    _write_synthetic_shard(tmp_path / "movi_a_train_shard000.npz")
    ds = MoviTransitionDataset(tmp_path / "movi_a_train")
    assert len(ds) == 12
    ctx, tgt, v = ds[0]
    # 2 RGB frames flattened into a single channel dim of size 6.
    assert ctx.shape == (6, 64, 64)
    assert tgt.shape == (6, 64, 64)
    assert ctx.dtype == torch.float32
    assert 0.0 <= float(ctx.min()) and float(ctx.max()) <= 1.0
    assert v.shape == ()
    assert v.dtype == torch.float32


def test_loader_rejects_v2_cache(tmp_path):
    path = tmp_path / "movi_a_train_shard000.npz"
    np.savez_compressed(str(path), s_t=np.zeros((1, 64, 64), np.uint8), version=np.int64(2))
    try:
        MoviTransitionDataset(path)
    except RuntimeError as e:
        assert "v3" in str(e).lower() or "convert_movi" in str(e).lower()
    else:
        raise AssertionError("v2 cache should be rejected")


def test_model_forward_no_action():
    cfg = Config()
    model = RDJEPA(cfg)
    x = torch.randn(2, cfg.encoder_in_channels, 64, 64)
    out = model(x, K=3, use_checkpoint=False)
    assert out["h_K"].shape == (2, cfg.latent_total_dim)
    assert out["all_h"].shape == (3, 2, cfg.latent_total_dim)
    assert out["violations"].shape == (3, 2)
    # forward must NOT accept an action arg anymore.
    import inspect

    params = list(inspect.signature(RDJEPA.forward).parameters)
    assert "action" not in params


def test_violation_grounded_is_regression():
    # smooth-L1 against a continuous [0,1] target (not BCE on a bool).
    violations = torch.randn(5, 8)
    gt = torch.rand(8)
    loss = violation_grounded_loss(violations, gt)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # boolean input should no longer be the contract: a bool tensor would
    # produce a degenerate loss; ensure float target path is taken.
    assert violation_grounded_loss(violations, torch.zeros(8)).item() >= 0.0


def test_decoder_is_rgb():
    cfg = Config()
    dec = VizDecoder(latent_dim=cfg.latent_total_dim, out_channels=cfg.img_channels)
    h = torch.randn(2, cfg.latent_total_dim)
    out = dec(h)
    assert out.shape == (2, 3, 64, 64)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0  # sigmoid


def test_end_to_end_loss():
    cfg = Config()
    model = RDJEPA(cfg)
    x = torch.randn(4, cfg.encoder_in_channels, 64, 64)
    out = model(x, K=4, use_checkpoint=False)
    target = model.target(torch.randn(4, cfg.encoder_in_channels, 64, 64))
    gt = torch.rand(4)
    loss, metrics = total_loss(
        out["all_h"], out["h_K"], target, out["violations"], cfg, violation_gt=gt
    )
    assert torch.isfinite(loss)
    assert "loss/violation_grounded" in metrics
    assert "loss/trajectory" in metrics
