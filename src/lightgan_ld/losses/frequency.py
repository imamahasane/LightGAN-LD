from __future__ import annotations

import torch
from torch import nn


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for image reconstruction.

    This implementation weights Fourier discrepancies by detached normalized magnitude, which avoids
    trivial low-frequency dominance and remains stable under AMP.
    """

    def __init__(self, alpha: float = 1.0, log_matrix: bool = False, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.fft2(pred.float(), norm="ortho")
        target_f = torch.fft.fft2(target.float(), norm="ortho")
        diff = pred_f - target_f
        dist = diff.real.pow(2) + diff.imag.pow(2)
        weight = dist.detach().sqrt().pow(self.alpha)
        if self.log_matrix:
            weight = torch.log1p(weight)
        weight = weight / (weight.mean(dim=(-2, -1), keepdim=True) + self.eps)
        return (weight * dist).mean().to(pred.dtype)
