r"""Top-level RD-JEPA model: encode -> K-step lens refinement -> early exit.

Implements the Lens Paradigm (spec §2.2): a single shared refinement
function $F_\theta$ is applied K times to iteratively focus the latent
until it is physically sharp. Weight sharing keeps VRAM flat in K.

Cache v3 input (MOVi): two stacked RGB frames (s_{t-1}, s_t) for context
and (s_t, s_{t+1}) for target, providing velocity information. There is no
action modality in MOVi, so the lens refines a purely visual latent.

The loop returns:
  - h_K: the final (or early-exited) latent per sample.
  - k_used: per-sample number of steps actually taken (<= K).
  - all_h: stack of intermediate latents [K, B, d] for the discounted loss.
  - violations: [K, B] violation scores at each step (for early exit + aux loss).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import Config
from .deliberation import DeliberationStep, ViolationHead
from .ema import EMATargetEncoder
from .encoder import Encoder


class RDJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        spatial = cfg.latent_shape.value == "spatial"
        d = cfg.latent_total_dim

        # v3: stacked RGB frames (s_{t-1}, s_t) -> 2 * img_channels input
        self.encoder = Encoder(
            in_channels=cfg.encoder_in_channels,
            channels=cfg.encoder_channels,
            latent_dim=cfg.latent_dim,
            spatial=spatial,
            latent_channels=cfg.latent_channels,
        )
        # If spatial, flatten the [C,4,4] latent for the deliberation MLPs and
        # reshape back for the target loss. Flat path is the default POC config.
        self.spatial = spatial
        self.flat_dim = d

        self.lens = DeliberationStep(
            latent_dim=d,
            hidden_dim=cfg.hidden_dim,
            gate=cfg.gate.value,
        )
        self.violation = ViolationHead(latent_dim=d, hidden_dim=d)

        # Optional LayerNorm on encoder output for training stability
        self.latent_norm = nn.LayerNorm(d) if cfg.latent_layernorm else nn.Identity()

        # EMA target encoder (not trained by gradients)
        self.target_encoder = EMATargetEncoder(self.encoder, decay=cfg.ema_decay)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, 4, 4] -> [B, C*16] if spatial, else passthrough."""
        return x.flatten(1) if self.spatial else x

    def _unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C*16] -> [B, C, 4, 4] if spatial, else passthrough."""
        if self.spatial:
            return x.view(-1, self.cfg.latent_channels, 4, 4)
        return x

    def _refine_step(
        self, h: torch.Tensor, use_checkpoint: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One lens application; returns (h_next, violation_k)."""

        def run(hh: torch.Tensor) -> torch.Tensor:
            return self.lens(hh)

        if use_checkpoint and h.requires_grad:
            h_next = checkpoint(run, h, use_reentrant=False)
        else:
            h_next = run(h)
        v = self.violation(h_next)
        return h_next, v

    def forward(
        self,
        s_context: torch.Tensor,
        K: int | None = None,
        early_exit: bool = False,
        tau: float = 0.1,
        use_checkpoint: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run the deliberation loop.

        Args:
            s_context: [B, C_in, H, W] stacked context frames (s_{t-1}, s_t).
                C_in = 2 * img_channels (e.g. 6 for RGB).
            K: override cfg.K if set.
            early_exit: enable per-sample early exit on violation < tau.
            tau: violation threshold for early exit.
            use_checkpoint: gradient-checkpoint each lens step.
        """
        K = K if K is not None else self.cfg.K
        B = s_context.shape[0]
        device = s_context.device

        h = self._flatten(self.encoder(s_context))
        h = self.latent_norm(h)

        all_h = []
        all_v = []
        k_used = torch.full((B,), K, device=device, dtype=torch.long)
        exited = torch.zeros(B, dtype=torch.bool, device=device)

        for k in range(K):
            h, v = self._refine_step(h, use_checkpoint)
            all_h.append(h)
            all_v.append(v)

            if early_exit and not exited.all():
                below = (v < tau) & (~exited)
                # mark first-exit step for samples crossing the threshold
                newly = below & (k_used == K)
                k_used = torch.where(newly, torch.full_like(k_used, k + 1), k_used)
                exited = exited | below
                if exited.all():
                    # we still keep all_h/all_v truncated at k for exited samples,
                    # but for simplicity pad with the last value to keep tensors
                    # rectangular (the loss masks per-sample by k_used).
                    remainder = K - (k + 1)
                    for _ in range(remainder):
                        all_h.append(h)
                        all_v.append(v)
                    break

        return {
            "h_K": h,
            "k_used": k_used,
            "all_h": torch.stack(all_h, dim=0),  # [K, B, d]
            "violations": torch.stack(all_v, dim=0),  # [K, B]
        }

    def target(self, s_target: torch.Tensor) -> torch.Tensor:
        """Stop-gradient target latent from the EMA encoder (for the loss).

        Args:
            s_target: [B, C_in, H, W] stacked target frames (s_t, s_{t+1}).
        """
        with torch.no_grad():
            return self._flatten(self.target_encoder(s_target))
