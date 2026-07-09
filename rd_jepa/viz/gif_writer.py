r"""Gif writer: render the K-step deliberation as an animated gif.

Two gif types:
  1. Deliberation gif: decode each h_k -> frame, stack K frames, show the
     lens "twisting" the latent into focus.
  2. Rollout gif: side-by-side predicted h_K -> frame vs ground-truth s_{t+1}.

The viz decoder produces [B,C,64,64] RGB frames (C=3 for MOVi). We upscale
to 256x256 for visibility. Gifs are saved to runs/<exp>/gifs/ and also
logged to Aim as images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import Config


def _to_pil(frame_tensor: torch.Tensor, size: int = 256) -> Image.Image:
    """[C, H, W] float tensor in [0,1] -> PIL RGB image upscaled to `size`.

    If C==1 the single channel is replicated across RGB. If C==3 it is used
    directly.
    """
    frame = frame_tensor.detach().cpu().numpy()  # [C, H, W]
    if frame.shape[0] == 1:
        frame = np.repeat(frame, 3, axis=0)  # [3, H, W]
    frame = np.clip(frame, 0.0, 1.0)
    rgb = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3]
    img = Image.fromarray(rgb)
    if size != img.size[0]:
        img = img.resize((size, size), Image.BILINEAR)
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
        frame = decoder(h.unsqueeze(0))  # [1, C, H, W]
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
    s_tp1: torch.Tensor,  # [C, H, W]
    decoder,
    out_path: Path,
    size: int = 256,
) -> Path:
    """Side-by-side predicted vs ground-truth target frame, single gif."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_frame = decoder(h_final.unsqueeze(0))[0]  # [C, H, W]
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
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
    out_dir: Path,
    sample_idx: int = 0,
) -> dict[str, Path]:
    """Run the model on one batch and render both gif types for sample_idx.

    v3 batch format: (context[2*C,H,W], target[2*C,H,W], violation_gt[1]).
    For visualization, we display the second frame of target (s_{t+1}) as GT.
    """
    out_dir = Path(out_dir)
    s_context, s_target, _violation_gt = batch
    s_context = s_context.cuda()
    s_target = s_target.cuda()
    with torch.no_grad():
        out = model(s_context, K=cfg.K, early_exit=cfg.early_exit, tau=cfg.violation_tau)
    # GT visualization: second frame of target (s_{t+1}), channels [C:2C].
    C = cfg.img_channels
    gt_frame = s_target[sample_idx, C : 2 * C, :, :]  # [C, H, W]
    deliberation = write_deliberation_gif(
        out["all_h"], decoder, sample_idx, out_dir / f"deliberation_{sample_idx}.gif"
    )
    rollout = write_rollout_gif(
        out["h_K"][sample_idx], gt_frame, decoder, out_dir / f"rollout_{sample_idx}.gif"
    )
    return {"deliberation": deliberation, "rollout": rollout}
