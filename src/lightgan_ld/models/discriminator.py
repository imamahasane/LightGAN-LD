from __future__ import annotations

import torch
from torch import nn

from .blocks import GhostModule, norm2d


def maybe_sn(module: nn.Module, use: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if use else module


class PatchDiscriminator(nn.Module):
    """Fully convolutional PatchGAN discriminator with GhostModules and average-pooling downsampling."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64, num_layers: int = 4, spectral_norm: bool = False, norm: str = "batch"):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i in range(num_layers):
            ch = min(base_channels * (2**i), base_channels * 8)
            layers.extend([
                maybe_sn(nn.Conv2d(prev, ch, 3, padding=1), spectral_norm),
                nn.LeakyReLU(0.2, inplace=True),
                GhostModule(ch, ch, norm=norm),
                nn.AvgPool2d(2),
            ])
            prev = ch
        layers.append(maybe_sn(nn.Conv2d(prev, 1, 1), spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
