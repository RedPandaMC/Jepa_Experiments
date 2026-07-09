"""Regression tests for the MOVi v3 data + v2 model contract.

These run on CPU with synthetic .npz shards so they need neither the network
download nor a GPU. They lock in the post-PhyRE contract: RGB frames, no
action modality, a continuous collision-force violation target, and the v2
unified architecture (spatial latent, divergence mask, energy/contrastive/
divergence losses, curriculum K, asynchronous probing decoder).
"""
from __future__ import annotations

import inspect

import numpy as np
import torch

from rd_jepa.config import Config
from rd_jepa.data.loader import MoviTransitionDataset
from rd_jepa.losses import (
    contrastive_dynamics_loss,
    divergence_reg_loss,
    energy_conservation_loss,
    total_loss,
    violation_grounded_loss,
)
from rd_jepa.models.deliberation import DivergenceProjection
from rd_jepa.models.rd_jepa import RDJEPA
from rd_jepa.viz.decoder import VizDecoder, make_decoder_optimizer


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


def test_config_v2_defaults():
    cfg = Config()
    # spatial latent only
    assert cfg.latent_channels == 64
    assert cfg.latent_dim == 1024
    assert cfg.latent_total_dim == 1024  # latent_channels * 4 * 4
    assert cfg.encoder_in_channels == 6  # 2 stacked RGB frames
    assert cfg.img_channels == 3
    # curriculum K
    assert cfg.K_min == 1
    assert cfg.K_max == 15
    assert cfg.curriculum_warmup_epochs == 5
    # no ablation knobs
    for forbidden in ("gate", "latent_shape", "loss_trajectory", "gamma",
                      "tbptt_n", "K", "action_dim", "action_inject"):
        assert not hasattr(cfg, forbidden)
    # decoder async config
    assert cfg.decoder_interval == 4


def test_config_rejects_removed_kwargs():
    # dataclasses reject unknown kwargs automatically, but verify.
    try:
        Config(K=15)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("Config should reject removed K= kwarg")


def test_resolve_K_linear():
    cfg = Config(K_min=1, K_max=10, curriculum_warmup_epochs=5)
    assert cfg.resolve_K(0) == 1  # start at K_min
    assert cfg.resolve_K(5) == 10  # reached K_max at warmup end
    assert cfg.resolve_K(100) == 10  # clamped at K_max
    # monotonic
    ks = [cfg.resolve_K(e) for e in range(6)]
    assert all(ks[i + 1] >= ks[i] for i in range(len(ks) - 1))


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
    params = list(inspect.signature(RDJEPA.forward).parameters)
    assert "action" not in params


def test_divergence_preserves_mass():
    """The divergence projection must conserve latent L2 mass (incompressibility)."""
    proj = DivergenceProjection(latent_channels=64)
    h_sp = torch.randn(4, 64, 4, 4)
    h_proj = proj(h_sp)
    in_norm = torch.norm(h_sp.flatten(1), dim=-1)
    out_norm = torch.norm(h_proj.flatten(1), dim=-1)
    assert torch.allclose(in_norm, out_norm, atol=1e-4)


def test_energy_loss_small_when_stable():
    """Energy conservation loss is ~0 when ||h_K|| ≈ ||h_0||."""
    h = torch.randn(8, 2, 1024)  # [K, B, d]
    h[-1] = h[0].clone()  # same magnitude -> zero energy loss
    loss = energy_conservation_loss(h)
    assert loss.item() < 1e-6


def test_contrastive_loss_penalizes_stasis_when_push():
    """When violation_gt > 0 and h_K == h_0, contrastive loss should be large."""
    h = torch.randn(8, 2, 1024)  # [K, B, d]
    h[-1] = h[0].clone()  # no movement
    gt = torch.ones(2)  # push present (B=2)
    loss_push = contrastive_dynamics_loss(h, gt, margin=1.0)
    gt_none = torch.zeros(2)  # no push
    loss_none = contrastive_dynamics_loss(h, gt_none, margin=1.0)
    assert loss_push.item() > 0.5  # margin penalty applies
    assert loss_none.item() < 1e-6  # no push -> no penalty


def test_divergence_reg_finite():
    h = torch.randn(5, 3, 1024)  # [K, B, d]
    loss = divergence_reg_loss(h)
    assert torch.isfinite(loss)
    # single-step trajectory -> zero
    assert divergence_reg_loss(torch.randn(1, 3, 1024)).item() == 0.0


def test_violation_grounded_is_regression():
    # smooth-L1 against a continuous [0,1] target (not BCE on a bool).
    violations = torch.randn(5, 8)
    gt = torch.rand(8)
    loss = violation_grounded_loss(violations, gt)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert violation_grounded_loss(violations, torch.zeros(8)).item() >= 0.0


def test_decoder_is_rgb_and_independent():
    cfg = Config()
    dec = VizDecoder(latent_dim=cfg.latent_total_dim, out_channels=cfg.img_channels)
    h = torch.randn(2, cfg.latent_total_dim, requires_grad=True)
    out = dec(h)
    assert out.shape == (2, 3, 64, 64)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0  # sigmoid

    # New v2 contract: the caller detaches h (see train_decoder_step).
    # decoder_loss no longer detaches internally. With a detached input no
    # graph connects to the latent, so its grad stays None.
    h2 = torch.randn(2, cfg.latent_total_dim, requires_grad=True)
    loss = dec.decoder_loss(h2.detach(), torch.rand(2, 3, 64, 64))
    loss.backward()
    assert h2.grad is None  # decoder is independent of the JEPA latent


def test_decoder_optimizer_is_dedicated():
    cfg = Config()
    dec = VizDecoder(latent_dim=cfg.latent_total_dim, out_channels=cfg.img_channels)
    opt = make_decoder_optimizer(dec, cfg)
    assert opt.param_groups[0]["lr"] == cfg.decoder_lr
    # decoder params only (not shared with any JEPA model)
    model = RDJEPA(cfg)
    dec_params = {id(p) for p in opt.param_groups[0]["params"]}
    model_params = {id(p) for p in model.parameters()}
    assert dec_params.isdisjoint(model_params)


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
    assert "loss/jepa" in metrics  # final-only (no 'loss/trajectory')
    assert "loss/trajectory" not in metrics
    assert "loss/energy" in metrics
    assert "loss/contrastive" in metrics
    assert "loss/divergence_reg" in metrics
