from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import norm2d


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch"):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), norm2d(norm, out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), norm2d(norm, out_ch), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinogramEncoder(nn.Module):
    """Learned sinogram-to-image initializer described in LightGAN-LD."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, norm: str = "batch"):
        super().__init__()
        c = base_channels
        self.down1 = nn.Sequential(ConvBlock(in_channels, c, norm), nn.AvgPool2d(2))
        self.down2 = nn.Sequential(ConvBlock(c, c * 2, norm), nn.AvgPool2d(2))
        self.down3 = nn.Sequential(ConvBlock(c * 2, c * 4, norm), nn.AvgPool2d(2))
        self.refine = nn.Sequential(ConvBlock(c * 4, c * 4, norm), ConvBlock(c * 4, c * 2, norm), ConvBlock(c * 2, c * 2, norm))
        self.head = nn.Sequential(nn.Conv2d(c * 2, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, out_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, sinogram: torch.Tensor, out_size: int | tuple[int, int] = 256) -> torch.Tensor:
        x = self.down1(sinogram)
        x = self.down2(x)
        x = self.down3(x)
        x = self.refine(x)
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.head(x)
