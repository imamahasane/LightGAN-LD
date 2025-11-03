import torch
import torch.nn as nn
from .blocks.ghost import GhostModule

def sn_if(x: nn.Module, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(x) if use_sn else x

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base=64, spectral_norm=False):
        super().__init__()
        c = base
        def block(ic, oc, k=3, s=2, p=1):
            return nn.Sequential(
                sn_if(nn.Conv2d(ic, oc, k, s, p), spectral_norm),
                nn.LeakyReLU(0.2, inplace=True),
                GhostModule(oc, oc),
            )
        self.net = nn.Sequential(
            block(in_ch, c),
            block(c, c*2),
            block(c*2, c*4),
            block(c*4, c*4),
            sn_if(nn.Conv2d(c*4, 1, 1), spectral_norm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
