r"""Vision encoder $E_\theta$ mapping a 64x64 scene-id frame to a latent.

A lightweight 4-layer strided CNN ending in a depthwise-conv block
(ConvNeXt-flavored) for cheap spatial inductive bias, then a linear head
to the configured latent dim. ~1.2M params, <2GB activations at batch 64.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> GELU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(out_ch // 4, 4), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseBlock(nn.Module):
    """Depthwise 7x7 conv + pointwise 1x1 (ConvNeXt block, residual)."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(channels // 4, 4), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.pw(self.dw(x))))


class Encoder(nn.Module):
    """E_\\theta: [B, 1, 64, 64] -> [B, latent_dim] (flat) or [B, C, 4, 4] (spatial)."""

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        latent_dim: int = 256,
        spatial: bool = False,
        latent_channels: int = 64,
    ):
        super().__init__()
        self.spatial = spatial
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels

        self.blocks = nn.ModuleList()
        in_ch = 1
        for ch in channels:
            self.blocks.append(ConvBlock(in_ch, ch, stride=2))
            in_ch = ch

        self.depthwise = DepthwiseBlock(channels[-1])

        if spatial:
            # project to latent_channels, keep spatial dims (4x4 after 4 stride-2 convs)
            self.head = nn.Conv2d(channels[-1], latent_channels, kernel_size=1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.depthwise(x)
        if self.spatial:
            return self.head(x)  # [B, C, 4, 4]
        x = self.pool(x).flatten(1)  # [B, channels[-1]]
        return self.head(x)  # [B, latent_dim]
