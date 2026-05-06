from __future__ import annotations

import torch
from torch import nn

from lightgan_ld.metrics import ssim


class SSIMLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(pred, target).mean()
