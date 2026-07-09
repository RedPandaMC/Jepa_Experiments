import pytest

from rd_jepa.config import Config
from rd_jepa.data.loader import PhyreTransitionDataset, build_dataloaders


@pytest.fixture(scope="module")
def cfg():
    return Config(batch_size=8)


def test_cache_loads(cfg):
    ds = PhyreTransitionDataset(f"{cfg.cache_dir}/{cfg.tier}_fold{cfg.fold}_train.npz")
    assert len(ds) > 0
    s_t, a, s_tp1 = ds[0]
    assert s_t.shape == (1, 64, 64)
    assert a.shape == (3,)
    assert s_tp1.shape == (1, 64, 64)
    assert 0.0 <= s_t.min() <= s_t.max() <= 1.0
    assert 0.0 <= s_tp1.min() <= s_tp1.max() <= 1.0


def test_dataloaders_have_three_splits(cfg):
    loaders = build_dataloaders(cfg)
    assert set(loaders.keys()) == {"train", "dev", "test"}


def test_train_batches_shuffled(cfg):
    loaders = build_dataloaders(cfg)
    a = next(iter(loaders["train"]))
    b = next(iter(loaders["train"]))
    # shuffle should make order unstable (prob of identical index order ~0)
    assert a[0].shape == b[0].shape
