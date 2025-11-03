import torch
from ..utils.metrics import ssim as _ssim
def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - _ssim(pred, target).mean()
