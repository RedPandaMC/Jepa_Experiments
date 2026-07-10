r"""Gif writer: render the K-step deliberation as an animated gif.

Two viz types:
  1. Deliberation gif: decode each h_k -> frame, stack K frames, show the
     lens "twisting" the latent into focus. Saved as an animated .gif.
  2. Rollout image: side-by-side predicted h_K -> frame vs ground-truth
     s_{t+1}. Saved as .png — a single static frame needs no animation and
     GIF's 256-color palette heavily posterizes a two-image side-by-side.

The viz decoder produces [B,C,64,64] RGB frames (C=3 for MOVi). We upscale
to 384x384 for visibility. Files are saved to runs/<exp>/gifs/ and also
logged to Aim as images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import Config


def _select_viz_indices(total_steps: int, frame_stride: int = 2, max_frames: int = 4) -> list[int]:
    """Choose a sparse subset of latent steps to decode for visualization."""
    if total_steps <= 1:
        return [0]
    indices = list(range(0, total_steps, frame_stride))
    if indices[-1] != total_steps - 1:
        indices.append(total_steps - 1)
    if len(indices) > max_frames:
        step = max(1, len(indices) // max_frames)
        indices = indices[::step]
        if indices[-1] != total_steps - 1:
            indices.append(total_steps - 1)
    return indices


def _to_pil(frame_tensor: torch.Tensor, size: int = 384) -> Image.Image:
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
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return img


def write_deliberation_gif(
    all_h: torch.Tensor,  # [K, B, d]
    decoder,
    sample_idx: int,
    out_path: Path,
    size: int = 384,
    duration: int = 200,
    frame_stride: int = 2,
    max_frames: int = 4,
) -> tuple[Path, list[Image.Image]]:
    """Save a sparse sample of the latent refinements as an animated gif."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = _select_viz_indices(all_h.shape[0], frame_stride=frame_stride, max_frames=max_frames)
    frames: list[Image.Image] = []
    for k in indices:
        h = all_h[k, sample_idx]  # [d]
        frame = decoder(h.unsqueeze(0))  # [1, C, H, W]
        frames.append(_to_pil(frame[0], size=size))
    if not frames:
        return out_path, []
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return out_path, frames


def write_rollout_image(
    h_final: torch.Tensor,  # [d]
    s_tp1: torch.Tensor,  # [C, H, W]
    decoder,
    out_path: Path,
    size: int = 384,
) -> tuple[Path, Image.Image]:
    """Side-by-side predicted vs ground-truth target frame.

    Saved as PNG (lossless, full RGB) rather than GIF — a single static
    comparison frame gains nothing from animation, and GIF's 256-color
    palette would posterize two different RGB images side by side.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_frame = decoder(h_final.unsqueeze(0))[0]  # [C, H, W]
    pred_img = _to_pil(pred_frame, size=size)
    gt_img = _to_pil(s_tp1, size=size)
    # stitch horizontally
    combined = Image.new("RGB", (size * 2, size), (255, 255, 255))
    combined.paste(pred_img, (0, 0))
    combined.paste(gt_img, (size, 0))
    combined.save(out_path, format="PNG")
    return out_path, combined


def render_rollout_for_eval(
    model,
    decoder,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Config,
    out_dir: Path,
    sample_idx: int = 0,
) -> dict[str, Any]:
    """Run the model on one batch and render both gif types for sample_idx.

    v3 batch format: (context[2*C,H,W], target[2*C,H,W], violation_gt[1]).
    For visualization, we display the second frame of target (s_{t+1}) as GT.

    Returns a dict with the saved file paths plus the in-memory PIL frames so
    the caller can log them to Aim without re-reading from disk:
      deliberation_path  : Path to the saved animated .gif.
      rollout_path       : Path to the saved side-by-side .png.
      deliberation_frames: list[PIL.Image] of the K latent refinements.
      rollout_img        : PIL.Image of the side-by-side prediction vs GT.
    """
    out_dir = Path(out_dir)
    s_context, s_target, _violation_gt = batch
    # Match the model's device (CPU in smoke tests, CUDA in training).
    device = next(model.parameters()).device
    s_context = s_context.to(device)
    s_target = s_target.to(device)
    with torch.no_grad():
        out = model(s_context, K=cfg.K_max)
    # GT visualization: second frame of target (s_{t+1}), channels [C:2C].
    C = cfg.img_channels
    gt_frame = s_target[sample_idx, C : 2 * C, :, :]  # [C, H, W]
    deliberation_path, deliberation_frames = write_deliberation_gif(
        out["all_h"],
        decoder,
        sample_idx,
        out_dir / f"deliberation_{sample_idx}.gif",
        size=cfg.viz_size,
        frame_stride=cfg.viz_frame_stride,
        max_frames=cfg.viz_max_frames,
    )
    rollout_path, rollout_img = write_rollout_image(
        out["h_K"][sample_idx],
        gt_frame,
        decoder,
        out_dir / f"rollout_{sample_idx}.png",
        size=cfg.viz_size,
    )
    return {
        "deliberation_path": deliberation_path,
        "rollout_path": rollout_path,
        "deliberation_frames": deliberation_frames,
        "rollout_img": rollout_img,
    }
