import torch
import torch.nn as nn
from .blocks.residual_ghost import ResidualGhostBlock

class LightGANLDGenerator(nn.Module):
    def __init__(self, in_ch=1, base=48, num_down=4, use_ghost=True, use_condconv=True, use_eca=True, experts=4):
        super().__init__()
        chs = [base*(2**i) for i in range(num_down)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        last_ch = in_ch
        for c in chs:
            self.downs.append(ResidualGhostBlock(last_ch, c, use_ghost, use_condconv, use_eca, experts))
            self.pools.append(nn.AvgPool2d(2))
            last_ch = c
        self.bottleneck = ResidualGhostBlock(last_ch, last_ch, use_ghost, use_condconv, use_eca, experts)
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.upconvs.append(ResidualGhostBlock(last_ch + c, c, use_ghost, use_condconv, use_eca, experts))
            last_ch = c
        self.out_conv = nn.Conv2d(last_ch, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for d, p in zip(self.downs, self.pools):
            x = d(x); skips.append(x); x = p(x)
        x = self.bottleneck(x)
        for up, blk, skip in zip(self.ups, self.upconvs, reversed(skips)):
            x = up(x)
            if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = blk(x)
        return self.act(self.out_conv(x))
