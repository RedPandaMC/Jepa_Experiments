r"""Vision encoder $E_\theta$ mapping stacked RGB frames to a spatial latent.

A lightweight 4-layer strided CNN ending in a depthwise-conv block
(ConvNeXt-flavored) for cheap spatial inductive bias, then a 1x1 conv head
to `latent_channels`. ~1.2M params, <2GB activations at batch 64.

Output is always spatial: [B, latent_channels, 4, 4] (the divergence-
projection mask in the lens requires spatial axes). Input is a stack of
two RGB frames (s_{t-1}, s_t) for velocity context, giving
`in_channels = 6` by default.
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
    r"""E_\theta: [B, in_channels, 64, 64] -> [B, latent_channels, 4, 4].

    Spatial latent only (v2). in_channels defaults to 6 (two stacked RGB
    frames). The 4 stride-2 conv blocks reduce 64x64 -> 4x4; the 1x1 head
    projects to `latent_channels`.
    """

    def __init__(
        self,
        in_channels: int = 6,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        latent_channels: int = 64,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for ch in channels:
            self.blocks.append(ConvBlock(in_ch, ch, stride=2))
            in_ch = ch

        self.depthwise = DepthwiseBlock(channels[-1])
        # 1x1 conv head -> latent_channels, keeping the 4x4 spatial dims.
        self.head = nn.Conv2d(channels[-1], latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.depthwise(x)
        return self.head(x)  # [B, latent_channels, 4, 4]
