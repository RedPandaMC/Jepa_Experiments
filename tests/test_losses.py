import pytest
import torch

from rd_jepa.config import Config, LossTrajectory
from rd_jepa.losses import (
    latent_prediction_loss,
    total_loss,
    trajectory_loss,
    violation_aux_loss,
)


@pytest.fixture
def cfg():
    return Config(K=4, latent_dim=64, gamma=0.7)


def test_latent_prediction_loss():
    h = torch.randn(8, 64)
    t = torch.randn(8, 64)
    loss = latent_prediction_loss(h, t)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_trajectory_loss_final(cfg):
    cfg.loss_trajectory = LossTrajectory.FINAL
    all_h = torch.randn(cfg.K, 8, 64)
    target = torch.randn(8, 64)
    loss = trajectory_loss(all_h, target, cfg)
    assert loss.ndim == 0


def test_trajectory_loss_discounted(cfg):
    cfg.loss_trajectory = LossTrajectory.DISCOUNTED
    all_h = torch.randn(cfg.K, 8, 64)
    target = torch.randn(8, 64)
    loss = trajectory_loss(all_h, target, cfg)
    assert loss.ndim == 0


def test_violation_aux_loss_penalizes_increase():
    v = torch.tensor([[1.0, 1.0], [0.5, 1.5]])  # step 1: decreases, increases
    loss = violation_aux_loss(v)
    assert loss.item() > 0


def test_violation_aux_loss_zero_on_decrease():
    v = torch.tensor([[1.0, 1.0], [0.5, 0.3]])
    loss = violation_aux_loss(v)
    assert loss.item() == 0.0


def test_total_loss_returns_metrics(cfg):
    all_h = torch.randn(cfg.K, 8, 64)
    h_final = all_h[-1]
    target = torch.randn(8, 64)
    violations = torch.randn(cfg.K, 8)
    loss, metrics = total_loss(all_h, h_final, target, violations, cfg)
    assert "loss/total" in metrics
    assert "loss/trajectory" in metrics
    assert "loss/violation_aux" in metrics
    assert loss.ndim == 0
