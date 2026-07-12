r"""RD-JEPA model: Resonant Decomposition JEPA.

Encodes a context window → decomposes latent into amplitude-phase modes →
runs K steps of coupled-oscillator dynamics → recombines → predicts the
EMA target encoder's representation of the future window.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import Config
from .ema import EMATargetEncoder
from .patch_encoder import PatchEncoder
from .resonator import AnalyticProjection, RecombineProjection, ResonatorBank


class CKJEPA(nn.Module):
    """Resonant Decomposition JEPA."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = PatchEncoder(
            in_channels=cfg.n_features,
            patch_len=cfg.patch_len,
            latent_dim=cfg.latent_dim,
            n_patches=cfg.n_patches,
            hidden_dim=cfg.encoder_hidden,
            n_layers=cfg.encoder_layers,
        )

        self.analytic = AnalyticProjection(cfg.latent_dim, cfg.n_modes)
        self.resonator = ResonatorBank(cfg.latent_dim, cfg.n_modes, dt=cfg.dt)
        self.resonator.set_sparsity(cfg.coupling_sparsity)
        self.recombine = RecombineProjection(cfg.n_modes, cfg.latent_dim)
        self.latent_norm = nn.LayerNorm(cfg.latent_dim)

        self.target_encoder = EMATargetEncoder(self.encoder, decay=cfg.ema_decay)

    def forward(
        self,
        x_context: torch.Tensor,  # [B, L, C]
        K_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: context → oscillator resonance → predicted latent.

        Returns dict with:
            h_K: [B, d] final predicted latent
            phases: [B, N] final phases (for phase-diversity loss)
            all_phases: [K, B, N] phase trajectory
        """
        K = K_steps if K_steps is not None else self.cfg.K_steps

        # Encode context
        z_0 = self.encoder(x_context)  # [B, d]

        # Decompose into amplitude-phase modes
        r, phi = self.analytic(z_0)  # [B, N], [B, N]

        # Run coupled-oscillator dynamics
        r_k, phi_k, all_phases = self.resonator(r, phi, z_0, K)

        # Recombine modes → predicted latent
        z_k = self.recombine(r_k, phi_k)  # [B, d]
        h_k = self.latent_norm(z_k)

        return {
            "h_K": h_k,
            "phases": phi_k,
            "all_phases": all_phases,
        }

    @torch.no_grad()
    def target(self, x_target: torch.Tensor) -> torch.Tensor:
        """Stop-gradient EMA target encoder's representation of the future."""
        return self.target_encoder(x_target)  # type: ignore[no-any-return]

    def update_ema(self, step: int) -> None:
        """Update target encoder EMA."""
        self.target_encoder.update_ema(
            self.encoder, step, warmup=self.cfg.ema_warmup
        )
