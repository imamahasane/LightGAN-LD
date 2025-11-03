import torch
def focal_frequency_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    def fft2(x): return torch.fft.rfft2(x, norm="ortho")
    Pf = fft2(pred); Tf = fft2(target)
    diff = torch.abs(Pf - Tf)
    weight = diff.detach() ** alpha
    loss = (weight * diff).mean()
    return loss
