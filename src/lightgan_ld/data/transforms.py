import torch
import random

def mixstyle(x: torch.Tensor, p: float = 0.5, alpha: float = 0.1, training: bool = True) -> torch.Tensor:
    if (not training) or random.random() > p:
        return x
    B, C, H, W = x.size()
    x = x.clone()
    mu = x.mean(dim=[2,3], keepdim=True)
    var = x.var(dim=[2,3], keepdim=True, unbiased=False)
    sig = (var + 1e-6).sqrt()
    perm = torch.randperm(B, device=x.device)
    mu2, sig2 = mu[perm], sig[perm]
    lam = torch.distributions.Beta(alpha, alpha).sample((B,1,1,1)).to(x.device)
    mu_mix = lam * mu + (1 - lam) * mu2
    sig_mix = lam * sig + (1 - lam) * sig2
    x_norm = (x - mu) / (sig + 1e-6)
    return x_norm * sig_mix + mu_mix

def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)/4.0
    kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)/4.0
    gx = torch.nn.functional.conv2d(x, kernel_x, padding=1, groups=x.size(1))
    gy = torch.nn.functional.conv2d(x, kernel_y, padding=1, groups=x.size(1))
    return torch.sqrt(gx**2 + gy**2 + 1e-8)
