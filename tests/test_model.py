import pytest
import torch

from rd_jepa.config import Config
from rd_jepa.models.rd_jepa import RDJEPA


@pytest.fixture
def cfg():
    return Config(batch_size=4, K=5, latent_dim=256, hidden_dim=512)


def test_model_shapes_flat(cfg):
    model = RDJEPA(cfg)
    s_t = torch.randn(cfg.batch_size, 1, 64, 64)
    action = torch.rand(cfg.batch_size, 3)
    out = model(s_t, action, use_checkpoint=False)
    assert out["h_K"].shape == (cfg.batch_size, 256)
    assert out["all_h"].shape == (cfg.K, cfg.batch_size, 256)
    assert out["violations"].shape == (cfg.K, cfg.batch_size)
    assert out["k_used"].shape == (cfg.batch_size,)
    assert out["k_used"].max() <= cfg.K


def test_model_target(cfg):
    model = RDJEPA(cfg)
    s_tp1 = torch.randn(cfg.batch_size, 1, 64, 64)
    target = model.target(s_tp1)
    assert target.shape == (cfg.batch_size, 256)
    assert not target.requires_grad


def test_early_exit_reduces_k_used(cfg):
    model = RDJEPA(cfg)
    s_t = torch.randn(cfg.batch_size, 1, 64, 64)
    action = torch.rand(cfg.batch_size, 3)
    out = model(s_t, action, early_exit=True, tau=1.0, use_checkpoint=False)
    # tau=1.0 is very high; most samples should exit before K
    assert out["k_used"].min() < cfg.K


def test_spatial_latent(cfg):
    from rd_jepa.config import LatentShape

    cfg.latent_shape = LatentShape.SPATIAL
    cfg.latent_channels = 64
    model = RDJEPA(cfg)
    s_t = torch.randn(cfg.batch_size, 1, 64, 64)
    action = torch.rand(cfg.batch_size, 3)
    out = model(s_t, action, use_checkpoint=False)
    flat_dim = cfg.latent_channels * 4 * 4
    assert out["h_K"].shape == (cfg.batch_size, flat_dim)


def test_gradient_checkpoint_runs(cfg):
    model = RDJEPA(cfg)
    s_t = torch.randn(cfg.batch_size, 1, 64, 64, requires_grad=False)
    action = torch.rand(cfg.batch_size, 3)
    out = model(s_t, action, use_checkpoint=True)
    loss = out["h_K"].pow(2).mean()
    loss.backward()
    # at least the lens weights should receive gradients
    assert any(p.grad is not None for p in model.lens.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_vram_under_budget(cfg):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = RDJEPA(cfg).cuda()
    s_t = torch.randn(cfg.batch_size, 1, 64, 64, device="cuda")
    action = torch.rand(cfg.batch_size, 3, device="cuda")
    out = model(s_t, action, use_checkpoint=True)
    loss = out["h_K"].pow(2).mean()
    loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    # generous POC bound; the real budget check happens at the training batch size
    assert peak_gb < 5.5, f"peak VRAM {peak_gb:.2f}GB exceeds 5.5GB budget"
