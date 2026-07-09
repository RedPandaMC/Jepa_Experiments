import torch

from rd_jepa.viz.decoder import VizDecoder
from rd_jepa.viz.gif_writer import (
    _colorize,
    _to_pil,
    write_deliberation_gif,
    write_rollout_gif,
)


def test_decoder_output_shape():
    dec = VizDecoder(latent_dim=64)
    h = torch.randn(4, 64)
    out = dec(h)
    assert out.shape == (4, 1, 64, 64)
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_decoder_loss_detached():
    dec = VizDecoder(latent_dim=64)
    h = torch.randn(4, 64, requires_grad=True)
    s_t = torch.rand(4, 1, 64, 64)
    loss = dec.decoder_loss(h, s_t)
    loss.backward()
    # decoder params get gradients, but h does NOT (detached inside decoder_loss)
    assert h.grad is None
    assert any(p.grad is not None for p in dec.parameters())


def test_colorize_shape():
    frame = torch.rand(64, 64).numpy()
    rgb = _colorize(frame)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == "uint8"


def test_to_pil_returns_rgb():
    t = torch.rand(1, 64, 64)
    img = _to_pil(t, size=128)
    assert img.size == (128, 128)
    assert img.mode == "RGB"


def test_deliberation_gif(tmp_path):
    dec = VizDecoder(latent_dim=64)
    all_h = torch.randn(5, 2, 64)
    out = write_deliberation_gif(all_h, dec, sample_idx=0, out_path=tmp_path / "del.gif")
    assert out.exists() and out.suffix == ".gif"


def test_rollout_gif(tmp_path):
    dec = VizDecoder(latent_dim=64)
    h_final = torch.randn(64)
    s_tp1 = torch.rand(1, 64, 64)
    out = write_rollout_gif(h_final, s_tp1, dec, tmp_path / "roll.gif")
    assert out.exists() and out.suffix == ".gif"
