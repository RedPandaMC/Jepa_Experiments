r"""Resonator bank: coupled-oscillator latent dynamics.

The ResonatorBank replaces traditional recurrent deliberation with
Kuramoto-inspired coupled-oscillator dynamics in latent space.

Pipeline:
    1. AnalyticProjection: z ∈ R^d → N (amplitude, phase) mode pairs
    2. ResonatorBank: K-step Kuramoto dynamics (input-conditioned ω, K, α)
    3. RecombineProjection: N evolved modes → z_K ∈ R^d
"""
from __future__ import annotations

import torch
from torch import nn


class AnalyticProjection(nn.Module):
    """Decompose latent z ∈ R^d into N amplitude-phase mode pairs.

    Uses a linear projection to produce real/imaginary parts, then
    converts to polar coordinates (r, φ).
    """

    def __init__(self, latent_dim: int, n_modes: int):
        super().__init__()
        self.n_modes = n_modes
        # 2*N outputs: first N = real parts, last N = imaginary parts
        self.proj = nn.Linear(latent_dim, 2 * n_modes)
        # Learnable phase offset to break symmetry
        self.phase_offset = nn.Parameter(torch.randn(n_modes) * 0.1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: [B, d] → (r: [B, N], φ: [B, N])."""
        out = self.proj(z)  # [B, 2N]
        real = out[:, : self.n_modes]
        imag = out[:, self.n_modes :]
        # amplitudes (non-negative via softplus)
        r = torch.nn.functional.softplus(real)
        # phases via atan2
        phi = torch.atan2(imag, real + 1e-6) + self.phase_offset
        return r, phi


class ResonatorBank(nn.Module):
    """Kuramoto-inspired coupled-oscillator dynamics with K resonance steps.

    All parameters (natural frequencies ω, coupling matrix K, amplitude
    relaxation rate α, amplitude equilibrium r_eq) are input-conditioned —
    predicted from the initial latent z_0 via small MLPs. This makes
    test-time compute meaningful: different inputs → different trajectories.
    """

    def __init__(self, latent_dim: int, n_modes: int, dt: float = 0.1):
        super().__init__()
        self.n_modes = n_modes
        self.dt = dt

        # Small MLPs: z_0 → oscillator parameters (input-conditioned)
        freq_dim = max(n_modes, 32)
        self.freq_net = nn.Sequential(
            nn.Linear(latent_dim, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, n_modes),
        )
        # Coupling: output N*N matrix (flattened), then applied with sparsity mask
        coupling_dim = max(n_modes * n_modes, 128)
        self.coupling_net = nn.Sequential(
            nn.Linear(latent_dim, coupling_dim),
            nn.GELU(),
            nn.Linear(coupling_dim, n_modes * n_modes),
        )
        # Amplitude relaxation rate
        self.alpha_net = nn.Sequential(
            nn.Linear(latent_dim, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, n_modes),
        )
        # Amplitude equilibrium
        self.eq_net = nn.Sequential(
            nn.Linear(latent_dim, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, n_modes),
        )

        # Learnable frequency bias (global frequency prior)
        self.freq_bias = nn.Parameter(torch.randn(n_modes) * 0.5)

        # Coupling sparsity mask (registered as buffer, not learned)
        self.register_buffer("coupling_mask", torch.ones(n_modes, n_modes))

    def set_sparsity(self, sparsity: float) -> None:
        """Randomly zero out a fraction of the coupling matrix."""
        if sparsity <= 0:
            self.coupling_mask.fill_(1.0)
            return
        mask = torch.rand_like(self.coupling_mask) > sparsity
        self.coupling_mask.copy_(mask.float())

    def forward(
        self,
        r: torch.Tensor,  # [B, N] amplitudes
        phi: torch.Tensor,  # [B, N] phases
        z_0: torch.Tensor,  # [B, d] initial latent (for input conditioning)
        K_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run K Kuramoto resonance steps.

        Returns:
            r_K: [B, N] final amplitudes
            phi_K: [B, N] final phases
            phases_traj: [K, B, N] all phase states (for diagnostics)
        """
        B = r.shape[0]
        N = self.n_modes

        # Input-conditioned parameters
        omega = self.freq_net(z_0) + self.freq_bias  # [B, N]
        omega = torch.tanh(omega) * 3.0  # bounded natural frequencies

        coupling_flat = self.coupling_net(z_0)  # [B, N*N]
        coupling = coupling_flat.reshape(B, N, N) * self.coupling_mask  # [B, N, N]
        coupling = torch.tanh(coupling)  # bounded coupling

        alpha = torch.sigmoid(self.alpha_net(z_0)) * 0.5  # [B, N] relaxation rate
        r_eq = torch.nn.functional.softplus(self.eq_net(z_0))  # [B, N] equilibria

        # Run K steps
        phases_traj = torch.empty(K_steps, B, N, device=r.device, dtype=r.dtype)
        r_k, phi_k = r, phi
        for k in range(K_steps):
            # Kuramoto phase update: φ_i += (ω_i + Σ_j K_ij sin(φ_j - φ_i)) dt
            phase_diff = phi_k.unsqueeze(2) - phi_k.unsqueeze(1)  # [B, N, N]
            coupling_term = (coupling * torch.sin(phase_diff)).sum(dim=2)  # [B, N]
            phi_k = phi_k + (omega + coupling_term) * self.dt

            # Amplitude relaxation: r_i += α_i (r_eq_i - r_i) dt
            r_k = r_k + alpha * (r_eq - r_k) * self.dt

            phases_traj[k] = phi_k

        return r_k, phi_k, phases_traj


class RecombineProjection(nn.Module):
    """Recombine N evolved modes → latent z_K ∈ R^d.

    z_K = Σ_i r_i * cos(φ_i) * W_i   (learned basis W)
    """

    def __init__(self, n_modes: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(n_modes, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, r: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """r: [B, N], φ: [B, N] → [B, d]."""
        # real-valued mode signals
        mode_signals = r * torch.cos(phi)  # [B, N]
        z = self.proj(mode_signals)
        z = self.norm(z)
        return z  # type: ignore[no-any-return]
