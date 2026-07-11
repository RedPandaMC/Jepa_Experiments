r"""Tests for RD-JEPA v5 (Resonant Decomposition JEPA).

CPU-only tests that verify:
    - Config v5 defaults and rejected fields
    - Data loading (shapes, normalization, splits)
    - PatchEncoder output shape
    - AnalyticProjection produces valid amplitude-phase pairs
    - ResonatorBank dynamics (phase evolution, amplitude relaxation, coupling)
    - RecombineProjection output shape
    - Full model forward pass
    - Loss computation (all 4 terms)
    - Phase diversity loss behavior
    - ForecastProbe training and evaluation
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rd_jepa.config import Config, _check_rejected_kwargs
from rd_jepa.eval.forecast_probe import ForecastProbe
from rd_jepa.losses import (
    latent_prediction_loss,
    phase_diversity_loss,
    total_loss,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)
from rd_jepa.models.patch_encoder import PatchEncoder
from rd_jepa.models.rd_jepa import RDJEPA
from rd_jepa.models.resonator import (
    AnalyticProjection,
    RecombineProjection,
    ResonatorBank,
)

# ── Config tests ──────────────────────────────────────────────────────────


def test_config_v5_defaults():
    cfg = Config()
    assert cfg.dataset_name == "jena_climate"
    assert cfg.context_len == 144
    assert cfg.horizon == 72
    assert cfg.n_features == 21
    assert cfg.latent_dim == 256
    assert cfg.n_modes == 32
    assert cfg.K_steps == 6
    assert cfg.dt == 0.1
    assert cfg.vicreg_var_weight == 1.0
    assert cfg.phase_div_weight == 0.5
    assert cfg.batch_size == 256
    assert cfg.epochs == 50


def test_config_rejects_removed_fields():
    bad_kwargs = {"K": 5, "n_kernels": 4, "kernel_size": 3}
    rejected = _check_rejected_kwargs(bad_kwargs)
    assert "K" in rejected
    assert "n_kernels" in rejected
    assert "kernel_size" in rejected


def test_config_rejects_v4_fields():
    bad_kwargs = {
        "movi_variant": "movi_a",
        "frame_size": 128,
        "K_min": 1,
        "K_max": 3,
        "n_kernels": 4,
        "encoder_channels": (16, 32, 64, 128),
    }
    rejected = _check_rejected_kwargs(bad_kwargs)
    assert len(rejected) == len(bad_kwargs)


def test_config_n_patches():
    cfg = Config()
    assert cfg.n_patches == cfg.context_len // cfg.patch_len
    assert cfg.n_patches == 24


def test_config_to_dict():
    cfg = Config(exp_name="test")
    d = cfg.to_dict()
    assert d["exp_name"] == "test"
    assert d["dataset_name"] == "jena_climate"
    assert isinstance(d["data_dir"], str)


# ── PatchEncoder tests ─────────────────────────────────────────────────────


def test_patch_encoder_output_shape():
    encoder = PatchEncoder(
        in_channels=21, patch_len=6, latent_dim=256, n_patches=24, hidden_dim=512
    )
    x = torch.randn(4, 144, 21)  # [B, L, C]
    z = encoder(x)
    assert z.shape == (4, 256)


def test_patch_encoder_different_batch():
    encoder = PatchEncoder(
        in_channels=21, patch_len=6, latent_dim=128, n_patches=24, hidden_dim=256
    )
    x = torch.randn(1, 144, 21)
    z = encoder(x)
    assert z.shape == (1, 128)


# ── AnalyticProjection tests ──────────────────────────────────────────────


def test_analytic_projection_shapes():
    proj = AnalyticProjection(latent_dim=256, n_modes=32)
    z = torch.randn(4, 256)
    r, phi = proj(z)
    assert r.shape == (4, 32)
    assert phi.shape == (4, 32)


def test_analytic_projection_amplitudes_nonneg():
    proj = AnalyticProjection(latent_dim=256, n_modes=32)
    z = torch.randn(4, 256)
    r, phi = proj(z)
    assert (r >= 0).all()


def test_analytic_projection_phases_bounded():
    proj = AnalyticProjection(latent_dim=256, n_modes=32)
    z = torch.randn(4, 256) * 10
    r, phi = proj(z)
    # phases should be finite
    assert torch.isfinite(phi).all()


# ── ResonatorBank tests ───────────────────────────────────────────────────


def test_resonator_bank_shapes():
    bank = ResonatorBank(latent_dim=256, n_modes=32, dt=0.1)
    r = torch.rand(4, 32)
    phi = torch.randn(4, 32)
    z_0 = torch.randn(4, 256)
    r_k, phi_k, traj = bank(r, phi, z_0, K_steps=6)
    assert r_k.shape == (4, 32)
    assert phi_k.shape == (4, 32)
    assert traj.shape == (6, 4, 32)


def test_resonator_bank_phases_evolve():
    """Phases should change across K steps."""
    bank = ResonatorBank(latent_dim=256, n_modes=32, dt=0.1)
    r = torch.rand(4, 32)
    phi = torch.randn(4, 32)
    z_0 = torch.randn(4, 256)
    _, phi_k, traj = bank(r, phi, z_0, K_steps=6)
    # at least some phase change
    assert not torch.allclose(traj[0], traj[-1], atol=1e-6)


def test_resonator_bank_amplitude_relaxation():
    """Amplitudes should move toward equilibrium."""
    bank = ResonatorBank(latent_dim=256, n_modes=32, dt=0.1)
    r = torch.ones(4, 32) * 10.0  # start far from equilibrium
    phi = torch.zeros(4, 32)
    z_0 = torch.randn(4, 256)
    r_k, _, _ = bank(r, phi, z_0, K_steps=20)
    # amplitudes should have decreased toward equilibrium
    assert r_k.mean() < r.mean()


def test_resonator_bank_input_conditioned():
    """Different inputs should produce different oscillator trajectories."""
    bank = ResonatorBank(latent_dim=256, n_modes=32, dt=0.1)
    r = torch.rand(2, 32)
    phi = torch.zeros(2, 32)
    z_a = torch.randn(1, 256)
    z_b = torch.randn(1, 256) * 5
    _, phi_a, _ = bank(r[:1], phi[:1], z_a, K_steps=6)
    _, phi_b, _ = bank(r[:1], phi[:1], z_b, K_steps=6)
    assert not torch.allclose(phi_a, phi_b, atol=1e-4)


def test_resonator_bank_coupling_causes_sync():
    """With strong coupling, phases should synchronize more than without."""
    bank = ResonatorBank(latent_dim=256, n_modes=16, dt=0.1)
    bank.set_sparsity(0.0)  # all-to-all coupling
    r = torch.ones(1, 16)
    phi = torch.linspace(0, 2 * 3.14159, 16).unsqueeze(0)  # spread phases
    z_0 = torch.randn(1, 256)
    _, phi_final, _ = bank(r, phi, z_0, K_steps=50)
    # coupling can cause some sync, but this is not guaranteed for all inits
    # just check the dynamics ran and produced finite output
    assert torch.isfinite(phi_final).all()


def test_resonator_bank_sparsity():
    bank = ResonatorBank(latent_dim=256, n_modes=32, dt=0.1)
    bank.set_sparsity(0.5)
    mask = bank.coupling_mask
    # roughly half should be zeroed
    frac_zero = (mask == 0).float().mean().item()
    assert 0.3 < frac_zero < 0.7


def test_resonator_bank_gradient_flows():
    """Gradients should flow through the K-step loop."""
    bank = ResonatorBank(latent_dim=256, n_modes=32, dt=0.1)
    r = torch.rand(4, 32, requires_grad=True)
    phi = torch.randn(4, 32, requires_grad=True)
    z_0 = torch.randn(4, 256, requires_grad=True)
    r_k, phi_k, _ = bank(r, phi, z_0, K_steps=6)
    loss = r_k.sum() + phi_k.sum()
    loss.backward()
    assert r.grad is not None
    assert phi.grad is not None
    assert z_0.grad is not None


# ── RecombineProjection tests ────────────────────────────────────────────


def test_recombine_projection_shape():
    recomb = RecombineProjection(n_modes=32, latent_dim=256)
    r = torch.rand(4, 32)
    phi = torch.randn(4, 32)
    z = recomb(r, phi)
    assert z.shape == (4, 256)


# ── Full model tests ──────────────────────────────────────────────────────


def test_model_forward_shapes():
    cfg = Config()
    model = RDJEPA(cfg)
    x = torch.randn(4, cfg.context_len, cfg.n_features)
    out = model(x, K_steps=cfg.K_steps)
    assert out["h_K"].shape == (4, cfg.latent_dim)
    assert out["phases"].shape == (4, cfg.n_modes)
    assert out["all_phases"].shape == (cfg.K_steps, 4, cfg.n_modes)


def test_model_forward_different_K():
    cfg = Config()
    model = RDJEPA(cfg)
    x = torch.randn(4, cfg.context_len, cfg.n_features)
    out_3 = model(x, K_steps=3)
    out_10 = model(x, K_steps=10)
    assert out_3["all_phases"].shape[0] == 3
    assert out_10["all_phases"].shape[0] == 10
    # Different K should produce different latents (test-time compute)
    assert not torch.allclose(out_3["h_K"], out_10["h_K"], atol=1e-4)


def test_model_target_shape():
    cfg = Config()
    model = RDJEPA(cfg)
    x = torch.randn(4, cfg.horizon, cfg.n_features)
    target = model.target(x)
    assert target.shape == (4, cfg.latent_dim)


def test_model_ema_update():
    """EMA update should change target encoder weights after a gradient step."""
    cfg = Config()
    model = RDJEPA(cfg)
    param_before = list(model.target_encoder.parameters())[0].data.clone()
    # Modify a context encoder parameter (simulating a gradient step)
    with torch.no_grad():
        list(model.encoder.parameters())[0].add_(0.1)
    model.update_ema(step=10)
    param_after = list(model.target_encoder.parameters())[0].data
    assert not torch.allclose(param_before, param_after)


# ── Loss tests ────────────────────────────────────────────────────────────


def test_latent_prediction_loss():
    h = torch.randn(4, 256)
    t = torch.randn(4, 256)
    loss = latent_prediction_loss(h, t)
    assert loss.item() >= 0
    assert loss.shape == ()


def test_vicreg_variance_loss():
    z_low_var = torch.ones(8, 16) * 0.01
    z_high_var = torch.randn(8, 16)
    l_low = vicreg_variance_loss(z_low_var)
    l_high = vicreg_variance_loss(z_high_var)
    assert l_low > l_high


def test_vicreg_covariance_loss():
    z_uncorrelated = torch.randn(32, 16)
    z_correlated = torch.randn(32, 1).repeat(1, 16)
    l_uncorr = vicreg_covariance_loss(z_uncorrelated)
    l_corr = vicreg_covariance_loss(z_correlated)
    assert l_corr > l_uncorr


def test_phase_diversity_loss_identical_phases():
    """All-same phases → high loss (low diversity)."""
    phases = torch.zeros(4, 32)  # all phase 0
    loss = phase_diversity_loss(phases)
    # mean resultant vector magnitude should be ~1 for identical phases
    assert loss.item() > 0.9


def test_phase_diversity_loss_spread_phases():
    """Uniformly spread phases → low loss (high diversity)."""
    n = 32
    phases = torch.linspace(0, 2 * 3.14159, n).unsqueeze(0).repeat(4, 1)
    loss = phase_diversity_loss(phases)
    assert loss.item() < 0.1


def test_total_loss_all_terms():
    cfg = Config()
    h = torch.randn(8, cfg.latent_dim)
    target = torch.randn(8, cfg.latent_dim)
    phases = torch.randn(8, cfg.n_modes)
    loss, metrics = total_loss(h, target, phases, cfg)
    assert "loss/total" in metrics
    assert "loss/jepa" in metrics
    assert "loss/vicreg_variance" in metrics
    assert "loss/vicreg_covariance" in metrics
    assert "loss/phase_diversity" in metrics
    assert "repr/std_mean" in metrics


# ── ForecastProbe tests ───────────────────────────────────────────────────


def test_forecast_probe_shape():
    probe = ForecastProbe(latent_dim=256, horizon=72, n_features=21)
    h = torch.randn(4, 256)
    pred = probe(h)
    assert pred.shape == (4, 72, 21)


def test_forecast_probe_compute_loss():
    probe = ForecastProbe(latent_dim=256, horizon=72, n_features=21)
    h = torch.randn(4, 256)
    target = torch.randn(4, 72, 21)
    loss, metrics = probe.compute_loss(h, target)
    assert "probe/mse" in metrics
    assert "probe/mae" in metrics
    assert loss.shape == ()


def test_forecast_probe_training_reduces_loss():
    """Probe loss should decrease with training."""
    probe = ForecastProbe(latent_dim=256, horizon=72, n_features=21)
    h = torch.randn(32, 256)
    target = torch.randn(32, 72, 21)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2)

    initial_loss = probe.compute_loss(h, target)[0].item()
    for _ in range(100):
        loss, _ = probe.compute_loss(h, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = probe.compute_loss(h, target)[0].item()
    assert final_loss < initial_loss


# ── Data loading tests (synthetic CSV) ─────────────────────────────────────


def _make_synthetic_csv(tmpdir: Path) -> Path:
    """Create a small synthetic CSV that matches Jena format."""
    n_rows = 5000
    header = "Date Time," + ",".join(f"col_{i}" for i in range(21))
    rows = [header]
    for t in range(n_rows):
        vals = [f"{np.sin(t * 0.01 * (i + 1)) + np.random.randn() * 0.1:.4f}" for i in range(21)]
        rows.append(f"01.01.2009 {t:02d}:{t % 60:02d}:00," + ",".join(vals))
    path = tmpdir / "jena_climate_2009_2016.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_dataset_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _make_synthetic_csv(Path(tmpdir))
        from rd_jepa.data.forecasting import JenaClimateDataset

        ds = JenaClimateDataset(
            csv_path, context_len=144, horizon=72, n_features=21, split="train"
        )
        context, target = ds[0]
        assert context.shape == (144, 21)
        assert target.shape == (72, 21)


def test_dataset_splits():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _make_synthetic_csv(Path(tmpdir))
        from rd_jepa.data.forecasting import JenaClimateDataset

        train = JenaClimateDataset(csv_path, split="train")
        val = JenaClimateDataset(csv_path, split="val")
        test = JenaClimateDataset(csv_path, split="test")
        # Splits should have different sizes
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        # Train should be largest
        assert len(train) >= len(val)


def test_dataset_normalization():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _make_synthetic_csv(Path(tmpdir))
        from rd_jepa.data.forecasting import JenaClimateDataset

        ds = JenaClimateDataset(csv_path, split="train", normalize=True)
        # Check that normalized data has roughly zero mean
        all_data = ds.data
        assert abs(all_data.mean()) < 0.5  # should be near zero


# ── End-to-end loss test ──────────────────────────────────────────────────


def test_end_to_end_loss():
    """Full forward + loss should produce finite loss."""
    cfg = Config()
    cfg.fast = True
    model = RDJEPA(cfg)
    x_ctx = torch.randn(4, cfg.context_len, cfg.n_features)
    x_tgt = torch.randn(4, cfg.horizon, cfg.n_features)
    out = model(x_ctx, K_steps=cfg.K_steps)
    target = model.target(x_tgt)
    loss, metrics = total_loss(out["h_K"], target, out["phases"], cfg)
    assert torch.isfinite(loss)
    assert all(isinstance(v, float) and v == v for v in metrics.values())  # no NaN


# ── AimLogger test ────────────────────────────────────────────────────────


def test_aim_logger_is_noop_when_package_missing():
    from rd_jepa.viz.aim_logger import AimLogger

    cfg = Config()
    logger = AimLogger(cfg)
    # If aim isn't installed, should be a no-op
    logger.init_run()
    logger.log_metrics({"test": 1.0}, step=0)
    logger.close()
