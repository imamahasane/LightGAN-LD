import torch
import torch.nn.functional as F
from lpips import LPIPS

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1,2,3])
    return 10 * torch.log10(1.0 / (mse + eps))

def _ssim_map(x, y, C1=0.01**2, C2=0.03**2, window_size=11):
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, 1, pad)
    mu_y = F.avg_pool2d(y, window_size, 1, pad)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, pad) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window_size, 1, pad) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, pad) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return num / (den + 1e-8)

def ssim(pred: torch.Tensor, target: torch.Tensor, window_size=11) -> torch.Tensor:
    ssim_map = _ssim_map(pred, target, window_size=window_size)
    return ssim_map.mean(dim=[1,2,3])

class LPIPSMetric:
    def __init__(self, net: str = "vgg"):
        self.lpips = LPIPS(net=net)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred3 = pred.repeat(1,3,1,1) * 2 - 1
        tgt3  = target.repeat(1,3,1,1) * 2 - 1
        return self.lpips(pred3, tgt3).squeeze()
