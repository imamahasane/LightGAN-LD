import torch
import torch.nn as nn
import torch.nn.functional as F

class SinogramEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, channels: int = 32):
        super().__init__()
        ch = channels
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2))
        self.down2 = nn.Sequential(nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2))
        self.down3 = nn.Sequential(nn.Conv2d(ch*2, ch*4, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2))
        self.refine = nn.Sequential(
            nn.Conv2d(ch*4, ch*4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*4, ch*4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*4, ch*2, 3, padding=1), nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch*2, ch, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, out_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, sino: torch.Tensor, out_size: int) -> torch.Tensor:
        x = self.down1(sino)
        x = self.down2(x)
        x = self.down3(x)
        x = self.refine(x)
        x = self.up(x)
        return torch.nn.functional.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
