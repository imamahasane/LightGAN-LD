from __future__ import annotations

import torch
import torch.nn.functional as F


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target), dim=(1, 2, 3))


def rmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3)) + eps)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1), reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range**2) / (mse + eps))


def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype).repeat(c, 1, 1, 1)
    pad = window_size // 2
    mu_x = F.conv2d(pred, kernel, padding=pad, groups=c)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=c)
    sigma_x = F.conv2d(pred * pred, kernel, padding=pad, groups=c) - mu_x.pow(2)
    sigma_y = F.conv2d(target * target, kernel, padding=pad, groups=c) - mu_y.pow(2)
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=c) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    return val.mean(dim=(1, 2, 3))


class LPIPSMetric:
    def __init__(self, net: str = "vgg", device: torch.device | str | None = None):
        self.available = False
        self.model = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            import lpips
            self.model = lpips.LPIPS(net=net).to(device).eval()
            self.available = True
        except Exception:
            self.available = False

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available or self.model is None:
            return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
        pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
        target3 = target.repeat(1, 3, 1, 1) * 2 - 1
        return self.model(pred3, target3).flatten()
