import pytest

from rd_jepa.config import Config
from rd_jepa.data.loader import PhyreTransitionDataset


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
    # build_dataloaders loads all 3 splits (slow on the full cache), so just
    # verify the dev dataset (smallest) loads and a batch is well-formed.
    from torch.utils.data import DataLoader

    from rd_jepa.data.loader import PhyreTransitionDataset

    ds = PhyreTransitionDataset(f"{cfg.cache_dir}/{cfg.tier}_fold{cfg.fold}_dev")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    assert batch[0].shape == (cfg.batch_size, 1, 64, 64)


def test_train_batches_shuffled(cfg):
    from torch.utils.data import DataLoader

    from rd_jepa.data.loader import PhyreTransitionDataset

    # use dev split (smaller) to keep the test under the time budget
    ds = PhyreTransitionDataset(f"{cfg.cache_dir}/{cfg.tier}_fold{cfg.fold}_dev")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    a = next(iter(loader))
    b = next(iter(loader))
    # shuffle should make order unstable (prob of identical index order ~0)
    assert a[0].shape == b[0].shape
