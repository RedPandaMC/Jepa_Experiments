r"""Top-level RD-JEPA v4 model: encode -> K-step kernel-lens refinement.

The lens is a ``KernelLens`` — a bank of N depthwise conv kernels on the
spatial latent that mutate per-sample during the K deliberation steps. The
kernel state ``[B, N, C, kH, kW]`` is initialized from learned base kernels
and evolves based on the latent at each step, making test-time compute
meaningful: different inputs produce different kernel trajectories.

The latent is spatial ``[B, latent_channels, 4, 4]`` from the encoder,
flattened to ``[B, d]`` for the deliberation loop (the kernel lens reshapes
internally). This keeps the external interface (losses, decoder, probe)
unchanged from v3.

Cache v3 input (MOVi): two stacked RGB frames (s_{t-1}, s_t) for context
and (s_t, s_{t+1}) for target, providing velocity information.

The loop returns:
  - h_K: the final latent per sample.
  - all_h: stack of intermediate latents [K, B, d] (for visualization).
  - violations: [K, B] violation scores at each step (diagnostic only).
  - gates: [K, B, N] kernel attention weights at each step.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import Config
from .deliberation import KernelLens, ViolationHead
from .ema import EMATargetEncoder
from .encoder import Encoder


class RDJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_total_dim

        self.encoder = Encoder(
            in_channels=cfg.encoder_in_channels,
            channels=cfg.encoder_channels,
            latent_channels=cfg.latent_channels,
        )
        self.spatial = True
        self.flat_dim = d

        self.lens = KernelLens(
            latent_dim=d,
            latent_channels=cfg.latent_channels,
            spatial_side=4,
            n_kernels=cfg.n_kernels,
            kernel_size=cfg.kernel_size,
            hidden_dim=cfg.hidden_dim,
        )
        self.n_kernels = cfg.n_kernels
        self.violation = ViolationHead(latent_dim=d, hidden_dim=cfg.hidden_dim)

        # Optional LayerNorm on encoder output for training stability
        self.latent_norm = nn.LayerNorm(d) if cfg.latent_layernorm else nn.Identity()

        # EMA target encoder (not trained by gradients)
        self.target_encoder = EMATargetEncoder(self.encoder, decay=cfg.ema_decay)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, 4, 4] -> [B, C*16] if spatial, else passthrough."""
        return x.flatten(1) if self.spatial else x

    def _refine_step(
        self, h: torch.Tensor, kernel_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One kernel-lens application; returns (h_next, violation_k, gate_k, ks_new)."""
        h_next, gate, ks_new = self.lens(h, kernel_state)
        v = self.violation(h_next)
        return h_next, v, gate, ks_new

    def forward(
        self,
        s_context: torch.Tensor,
        K: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the deliberation loop.

        Args:
            s_context: [B, C_in, H, W] stacked context frames (s_{t-1}, s_t).
                C_in = 2 * img_channels (e.g. 6 for RGB).
            K: override cfg.K_max if set (used by the curriculum schedule).
        """
        K = K if K is not None else self.cfg.K_max
        B = s_context.shape[0]
        device = s_context.device

        h = self._flatten(self.encoder(s_context))
        h = self.latent_norm(h)

        # Initialize per-sample kernel state from the learned base kernels.
        kernel_state = self.lens.init_kernels(B, device)

        all_h: list[torch.Tensor] = []
        all_v: list[torch.Tensor] = []
        all_gates: list[torch.Tensor] = []

        for _k in range(K):
            h, v, gate, kernel_state = self._refine_step(h, kernel_state)
            all_h.append(h)
            all_v.append(v)
            all_gates.append(gate)

        return {
            "h_K": h,
            "all_h": torch.stack(all_h, dim=0),  # [K, B, d]
            "violations": torch.stack(all_v, dim=0),  # [K, B]
            "gates": torch.stack(all_gates, dim=0),  # [K, B, N]
        }

    def target(self, s_target: torch.Tensor) -> torch.Tensor:
        """Stop-gradient target latent from the EMA encoder (for the loss).

        Args:
            s_target: [B, C_in, H, W] stacked target frames (s_t, s_{t+1}).
        """
        with torch.no_grad():
            return self._flatten(self.target_encoder(s_target))
