import torch
import torch.nn as nn
from .ghost import GhostModule
from .condconv import CondConv2d
from .eca import ECA

class ResidualGhostBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_ghost=True, use_condconv=True, use_eca=True, experts=4):
        super().__init__()
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        layers = []
        if use_ghost:
            layers.append(GhostModule(in_ch, out_ch))
        else:
            layers.extend([nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)])
        if use_condconv:
            layers.append(CondConv2d(out_ch, out_ch, 3, padding=1, K=experts))
        else:
            layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        if use_eca:
            layers.append(ECA(out_ch))
        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.proj is not None:
            x = self.proj(x)
        return self.act(x + y)
