from typing import Optional
import torch
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, ratio: int = 2, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        init_ch = (out_ch + ratio - 1) // ratio
        new_ch = out_ch - init_ch
        self.primary = nn.Conv2d(in_ch, init_ch, kernel_size, stride, padding, bias=False)
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, kernel_size=3, stride=1, padding=1, groups=init_ch, bias=False),
            nn.Conv2d(new_ch, new_ch, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        if y.shape[1] == 0:
            return x
        c = self.cheap(y)
        out = torch.cat([y, c], dim=1)
        out = self.bn(out)
        return self.act(out)
