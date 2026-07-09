r"""Gif writer: render the K-step deliberation as an animated gif.

Two gif types:
  1. Deliberation gif: decode each h_k -> frame, stack K frames, show the
     lens "twisting" the latent into focus.
  2. Rollout gif: side-by-side predicted h_K -> frame vs ground-truth s_{t+1}.

Since the viz decoder produces [B,1,64,64] grayscale scene-id maps, we
upscale to 256x256 and apply a viridis-style colormap so different
object ids are visually distinct. Gifs are saved to runs/<exp>/gifs/
and also logged to Aim as images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import Config


def _colorize(frame: np.ndarray) -> np.ndarray:
    """Map a [H, W] grayscale/scene-id frame in [0,1] to an RGB uint8 frame.

    Uses a simple fixed palette so object ids are visually distinct.
    """
    h, w = frame.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # 6 distinct colors for PhyRE scene ids 0..6
    palette = np.array(
        [
            [0, 0, 0],        # 0 background
            [255, 0, 0],      # 1
            [0, 255, 0],      # 2
            [0, 0, 255],      # 3
            [255, 255, 0],    # 4
            [0, 255, 255],    # 5
            [255, 0, 255],    # 6
        ],
        dtype=np.uint8,
    )
    ids = (frame * 6).clip(0, 6).astype(np.uint8)
    rgb = palette[ids]
    return rgb


def _to_pil(frame_tensor: torch.Tensor, size: int = 256) -> Image.Image:
    """[1, H, W] float tensor -> PIL RGB image upscaled to `size`."""
    frame = frame_tensor.squeeze(0).detach().cpu().numpy()
    rgb = _colorize(frame)
    img = Image.fromarray(rgb)
    if size != img.size[0]:
        img = img.resize((size, size), Image.NEAREST)
    return img


def write_deliberation_gif(
    all_h: torch.Tensor,  # [K, B, d]
    decoder,
    sample_idx: int,
    out_path: Path,
    size: int = 256,
    duration: int = 200,
) -> Path:
    """Save one sample's K latent refinements as an animated gif."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in range(all_h.shape[0]):
        h = all_h[k, sample_idx]  # [d]
        frame = decoder(h.unsqueeze(0))  # [1, 1, H, W]
        frames.append(_to_pil(frame[0], size=size))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return out_path


def write_rollout_gif(
    h_final: torch.Tensor,  # [d]
    s_tp1: torch.Tensor,  # [1, H, W]
    decoder,
    out_path: Path,
    size: int = 256,
) -> Path:
    """Side-by-side predicted vs ground-truth target frame, single gif."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_frame = decoder(h_final.unsqueeze(0))[0]  # [1, H, W]
    pred_img = _to_pil(pred_frame, size=size)
    gt_img = _to_pil(s_tp1, size=size)
    # stitch horizontally
    combined = Image.new("RGB", (size * 2, size), (255, 255, 255))
    combined.paste(pred_img, (0, 0))
    combined.paste(gt_img, (size, 0))
    combined.save(out_path)
    return out_path


def render_rollout_for_eval(
    model,
    decoder,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
    out_dir: Path,
    sample_idx: int = 0,
) -> dict[str, Path]:
    """Run the model on one batch and render both gif types for sample_idx.

    v2 batch format: (context[2,H,W], action[3], target[2,H,W], solved[bool]).
    For visualization, we display the second channel of target (s_{t+1}) as GT.
    """
    out_dir = Path(out_dir)
    # v2: context is 2-channel stack (s_{t-1}, s_t), target is (s_t, s_{t+1})
    s_context, action, s_target, _solved = batch
    s_context = s_context.cuda()
    action = action.cuda()
    s_target = s_target.cuda()
    with torch.no_grad():
        out = model(s_context, action, K=cfg.K, early_exit=cfg.early_exit, tau=cfg.violation_tau)
    # For GT visualization, use the second channel of target (s_{t+1})
    gt_frame = s_target[sample_idx, 1:2, :, :]  # [1, H, W]
    deliberation = write_deliberation_gif(
        out["all_h"], decoder, sample_idx, out_dir / f"deliberation_{sample_idx}.gif"
    )
    rollout = write_rollout_gif(
        out["h_K"][sample_idx], gt_frame, decoder, out_dir / f"rollout_{sample_idx}.gif"
    )
    return {"deliberation": deliberation, "rollout": rollout}
