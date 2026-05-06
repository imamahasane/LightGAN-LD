from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import MixStyle, ResidualGhostBlock


class LightGANLDGenerator(nn.Module):
    """Compact Ghost/CondConv/ECA U-Net generator for LDCT restoration."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 48, num_down: int = 4, use_ghost: bool = True, use_condconv: bool = True, use_eca: bool = True, condconv_experts: int = 4, norm: str = "batch", mixstyle_p: float = 0.0, mixstyle_alpha: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        channels = [base_channels * (2**i) for i in range(num_down)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.downs.append(ResidualGhostBlock(prev, ch, use_ghost=use_ghost, use_condconv=use_condconv, use_eca=use_eca, experts=condconv_experts, norm=norm, dropout=dropout))
            self.pools.append(nn.AvgPool2d(2))
            prev = ch
        self.bridge = nn.Sequential(
            ResidualGhostBlock(prev, prev, use_ghost=use_ghost, use_condconv=use_condconv, use_eca=use_eca, experts=condconv_experts, norm=norm, dropout=dropout),
            ResidualGhostBlock(prev, prev, use_ghost=use_ghost, use_condconv=use_condconv, use_eca=use_eca, experts=condconv_experts, norm=norm, dropout=dropout),
        )
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(nn.Conv2d(prev, ch, 1))
            self.dec.append(ResidualGhostBlock(ch + ch, ch, use_ghost=use_ghost, use_condconv=use_condconv, use_eca=use_eca, experts=condconv_experts, norm=norm, dropout=dropout))
            prev = ch
        self.out_conv = nn.Conv2d(prev, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixstyle(x)
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bridge(x)
        for up1x1, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up1x1(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return torch.sigmoid(self.out_conv(x))
