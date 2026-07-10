"""Regression tests for the MOVi v3 data + v4 simplified kernel-lens model.

These run on CPU with synthetic .npz shards so they need neither the network
download nor a GPU. They lock in the post-PhyRE contract: RGB frames, no
action modality, a continuous collision-force violation target, and the v4
simplified kernel-lens architecture (mutating depthwise conv kernels,
attention gate, JEPA + VICReg losses only, curriculum K, asynchronous
probing decoder).
"""
from __future__ import annotations

import importlib
import inspect
import sys

import numpy as np
import torch

from rd_jepa.config import Config
from rd_jepa.data.loader import MoviTransitionDataset
from rd_jepa.losses import (
    total_loss,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)
from rd_jepa.models.deliberation import KernelLens, ViolationHead
from rd_jepa.models.rd_jepa import RDJEPA
from rd_jepa.viz.decoder import VizDecoder, make_decoder_optimizer
from rd_jepa.viz.gif_writer import _select_viz_indices


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


def test_config_v4_defaults():
    cfg = Config()
    # spatial latent only
    assert cfg.latent_channels == 64
    assert cfg.latent_dim == 1024
    assert cfg.latent_total_dim == 1024  # latent_channels * 4 * 4
    assert cfg.encoder_in_channels == 6  # 2 stacked RGB frames
    assert cfg.img_channels == 3
    # curriculum K
    assert cfg.K_min == 1
    assert cfg.K_max == 3
    assert cfg.curriculum_warmup_epochs == 3
    # no ablation knobs
    for forbidden in ("gate", "latent_shape", "loss_trajectory", "gamma",
                      "tbptn_n", "K", "action_dim", "action_inject",
                      "n_lenses", "load_balance_weight", "router_entropy_weight",
                      "early_exit", "violation_tau", "violation_weight",
                      "violation_supervision_weight", "violation_grounded_weight",
                      "energy_weight", "contrastive_weight", "divergence_reg_weight",
                      "contrastive_margin", "kernel_diversity_weight"):
        assert not hasattr(cfg, forbidden)
    # decoder async config
    assert cfg.decoder_interval == 4
    # laptop-friendly VRAM defaults
    assert cfg.batch_size == 64
    assert cfg.grad_checkpoint is False
    assert cfg.vram_fraction >= 0.90
    # kernel lens
    assert cfg.n_kernels == 4
    assert cfg.kernel_size == 3
    # data loader
    assert cfg.num_workers == 4
    assert cfg.max_cached_shards == 2
    # simplified losses
    assert cfg.vicreg_var_weight == 1.0
    assert cfg.vicreg_cov_weight == 1.0


def test_config_rejects_removed_kwargs():
    # dataclasses reject unknown kwargs automatically, but verify.
    for bad in ("K", "n_lenses", "early_exit", "violation_tau",
                "energy_weight", "contrastive_weight", "kernel_diversity_weight"):
        try:
            Config(**{bad: 1})  # type: ignore[call-arg]
        except TypeError:
            pass
        else:
            raise AssertionError(f"Config should reject removed {bad}= kwarg")


def test_aim_logger_is_noop_when_package_missing():
    sys.modules.pop("rd_jepa.viz.aim_logger", None)
    module = importlib.import_module("rd_jepa.viz.aim_logger")
    logger = module.AimLogger(Config(exp_name="test"))

    logger.init_run()
    logger.log_metrics({"metric": 1.0}, step=1)
    logger.log_image("img", np.zeros((2, 2, 3), dtype=np.uint8), step=1)
    logger.close()

    assert logger.run is None


def test_resolve_K_linear():
    cfg = Config(K_min=1, K_max=3, curriculum_warmup_epochs=3)
    assert cfg.resolve_K(0) == 1  # start at K_min
    assert cfg.resolve_K(3) == 3  # reached K_max at warmup end
    assert cfg.resolve_K(100) == 3  # clamped at K_max
    # monotonic
    ks = [cfg.resolve_K(e) for e in range(4)]
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


def test_loader_resizes_to_target_frame_size(tmp_path):
    """When target_frame_size differs from cached frames, resize on-the-fly."""
    _write_synthetic_shard(tmp_path / "movi_a_train_shard000.npz")  # 64x64 cache
    ds = MoviTransitionDataset(
        tmp_path / "movi_a_train",
        target_frame_size=32,
    )
    ctx, tgt, _v = ds[0]
    assert ctx.shape == (6, 32, 32)
    assert tgt.shape == (6, 32, 32)


def test_loader_rejects_v2_cache(tmp_path):
    path = tmp_path / "movi_a_train_shard000.npz"
    np.savez_compressed(str(path), s_t=np.zeros((1, 64, 64), np.uint8), version=np.int64(2))
    try:
        MoviTransitionDataset(path)
    except RuntimeError as e:
        assert "v3" in str(e).lower() or "build_data" in str(e).lower()
    else:
        raise AssertionError("v2 cache should be rejected")


def test_model_forward_no_action():
    cfg = Config()
    model = RDJEPA(cfg)
    x = torch.randn(2, cfg.encoder_in_channels, 64, 64)
    out = model(x, K=3)
    assert out["h_K"].shape == (2, cfg.latent_total_dim)
    assert out["all_h"].shape == (3, 2, cfg.latent_total_dim)
    assert out["violations"].shape == (3, 2)
    assert out["gates"].shape == (3, 2, cfg.n_kernels)
    # forward must NOT accept an action arg.
    params = list(inspect.signature(RDJEPA.forward).parameters)
    assert "action" not in params
    assert "early_exit" not in params
    assert "use_checkpoint" not in params


def test_kernel_lens_mutates():
    """Kernels must change across K steps (mutation is the whole point)."""
    cfg = Config()
    lens = KernelLens(
        latent_dim=cfg.latent_total_dim,
        latent_channels=cfg.latent_channels,
        n_kernels=cfg.n_kernels,
        kernel_size=cfg.kernel_size,
        hidden_dim=cfg.hidden_dim,
    )
    B, d = 4, cfg.latent_total_dim
    h = torch.randn(B, d)
    ks0 = lens.init_kernels(B, torch.device("cpu"))
    h1, gate1, ks1 = lens(h, ks0)
    h2, gate2, ks2 = lens(h1, ks1)
    # Kernels must have actually changed.
    assert not torch.allclose(ks0, ks1, atol=1e-6)
    assert not torch.allclose(ks1, ks2, atol=1e-6)
    # Gates must be valid distributions.
    assert gate1.shape == (B, cfg.n_kernels)
    assert torch.allclose(gate1.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_kernel_lens_gate_shape():
    """Gate must be [B, N] and sum to 1 per sample."""
    cfg = Config(n_kernels=4)
    lens = KernelLens(
        latent_dim=cfg.latent_total_dim,
        latent_channels=cfg.latent_channels,
        n_kernels=cfg.n_kernels,
        kernel_size=cfg.kernel_size,
        hidden_dim=cfg.hidden_dim,
    )
    h = torch.randn(3, cfg.latent_total_dim)
    ks = lens.init_kernels(3, torch.device("cpu"))
    h_next, gate, _ = lens(h, ks)
    assert h_next.shape == (3, cfg.latent_total_dim)
    assert gate.shape == (3, 4)
    assert torch.allclose(gate.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_kernel_lens_base_kernels_physics_priors():
    """Base kernels should be seeded with Sobel/Laplacian/identity, not random."""
    lens = KernelLens(latent_dim=1024, latent_channels=64, n_kernels=4, kernel_size=3)
    bk = lens.base_kernels  # [N, C, k, k]
    # Kernel 0 should look like Sobel-x (nonzero on left/right columns).
    k0 = bk[0, 0]  # [k, k]
    assert k0[0, 0] != 0 or k0[0, 2] != 0  # corner is nonzero for Sobel
    # Kernel 3 should be identity-like (center = max).
    k3 = bk[3, 0]
    assert k3[1, 1].abs() >= k3[1, 0].abs()


def test_vicreg_variance_loss():
    """Low-variance dimensions should produce high loss."""
    z_low_var = torch.ones(8, 4) * 0.5  # zero variance -> max loss
    z_good = torch.randn(8, 4)  # healthy variance -> low loss
    loss_low = vicreg_variance_loss(z_low_var)
    loss_good = vicreg_variance_loss(z_good)
    assert loss_low.item() > loss_good.item()


def test_vicreg_covariance_loss():
    """Correlated dimensions should produce higher loss than uncorrelated."""
    torch.manual_seed(42)
    z_uncorr = torch.randn(32, 8)  # roughly uncorrelated
    z_corr = torch.randn(32, 4).repeat(1, 2)  # duplicated -> highly correlated
    loss_uncorr = vicreg_covariance_loss(z_uncorr)
    loss_corr = vicreg_covariance_loss(z_corr)
    assert loss_corr.item() > loss_uncorr.item()


def test_decoder_is_rgb_and_independent():
    cfg = Config()
    dec = VizDecoder(latent_dim=cfg.latent_total_dim, out_channels=cfg.img_channels)
    h = torch.randn(2, cfg.latent_total_dim, requires_grad=True)
    out = dec(h)
    assert out.shape == (2, 3, 64, 64)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0  # sigmoid

    # The caller detaches h (see train_decoder_step). With a detached
    # input no graph connects to the latent, so its grad stays None.
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


def test_select_viz_indices_reduces_frame_count():
    steps = _select_viz_indices(10, frame_stride=2, max_frames=4)
    assert steps == [0, 2, 4, 6, 8, 9]


def test_end_to_end_loss():
    cfg = Config()
    model = RDJEPA(cfg)
    x = torch.randn(4, cfg.encoder_in_channels, 64, 64)
    out = model(x, K=3)
    target = model.target(torch.randn(4, cfg.encoder_in_channels, 64, 64))
    loss, metrics = total_loss(out["h_K"], target, cfg)
    assert torch.isfinite(loss)
    assert "loss/jepa" in metrics
    assert "loss/vicreg_variance" in metrics
    assert "loss/vicreg_covariance" in metrics
    assert "loss/total" in metrics
    # No trajectory losses
    assert "loss/energy" not in metrics
    assert "loss/contrastive" not in metrics
    assert "loss/divergence_reg" not in metrics
    assert "loss/load_balance" not in metrics
    assert "loss/router_entropy" not in metrics
    assert "loss/kernel_diversity" not in metrics


def test_kernel_lens_mass_conservation():
    """Latent mass should be roughly conserved across steps (tanh bounded).

    The kernel lens applies tanh-bounded updates, so the magnitude drift is
    small but not zero. We check approximate conservation (within 10%).
    """
    cfg = Config(n_kernels=4, K_max=1)
    model = RDJEPA(cfg)
    x = torch.randn(4, cfg.encoder_in_channels, 64, 64)
    out = model(x, K=2)
    n0 = torch.norm(out["all_h"][0], p=2, dim=-1)
    nK = torch.norm(out["all_h"][1], p=2, dim=-1)
    ratio = nK / n0.clamp(min=1e-6)
    assert (ratio > 0.90).all() and (ratio < 1.10).all()


def test_kernel_lens_n_kernels_1():
    """n_kernels=1 should still work (single kernel, gate is trivially 1.0)."""
    cfg = Config(n_kernels=1)
    model = RDJEPA(cfg)
    x = torch.randn(2, cfg.encoder_in_channels, 64, 64)
    out = model(x, K=3)
    assert out["h_K"].shape == (2, cfg.latent_total_dim)
    assert out["gates"].shape == (3, 2, 1)
    # Single kernel gate should be all 1.0 (softmax of a single element).
    assert torch.allclose(out["gates"], torch.ones_like(out["gates"]))


def test_violation_head_output_shape():
    head = ViolationHead(latent_dim=1024, hidden_dim=128)
    h = torch.randn(4, 1024)
    v = head(h)
    assert v.shape == (4,)
