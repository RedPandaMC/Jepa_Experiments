import pytest
import torch

from rd_jepa.config import Config
from rd_jepa.models.rd_jepa import RDJEPA


@pytest.fixture
def cfg():
    return Config(batch_size=4, K=6, latent_dim=64)


def test_early_exit_trivial_when_tau_above_violation(cfg):
    """With tau well above any violation score, every sample exits at k=1."""
    model = RDJEPA(cfg).eval()
    s_t = torch.randn(cfg.batch_size, 1, 64, 64)
    action = torch.rand(cfg.batch_size, 3)
    out = model(s_t, action, early_exit=True, tau=100.0, use_checkpoint=False)
    assert (out["k_used"] == 1).all()


def test_early_exit_runs_full_when_tau_below_violation(cfg):
    """With tau far below any violation score, no early exit -> k_used == K."""
    model = RDJEPA(cfg).eval()
    s_t = torch.randn(cfg.batch_size, 1, 64, 64)
    action = torch.rand(cfg.batch_size, 3)
    out_full = model(s_t, action, early_exit=False, use_checkpoint=False)
    min_v = float(out_full["violations"].min().item())
    out = model(s_t, action, early_exit=True, tau=min_v - 1.0, use_checkpoint=False)
    assert (out["k_used"] == cfg.K).all()


def test_early_exit_partial_threshold(cfg):
    """A mid-range tau exits some samples early and runs others full."""
    model = RDJEPA(cfg).eval()
    s_t = torch.randn(cfg.batch_size, 1, 64, 64)
    action = torch.rand(cfg.batch_size, 3)
    # compute violations first to pick a tau that splits
    out_full = model(s_t, action, early_exit=False, use_checkpoint=False)
    v = out_full["violations"]
    mid = float(v.median().item())
    out = model(s_t, action, early_exit=True, tau=mid, use_checkpoint=False)
    # at least one sample should have exited and at least one not
    assert out["k_used"].min() < cfg.K
    assert out["k_used"].max() <= cfg.K
