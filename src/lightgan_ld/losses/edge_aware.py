import torch
from ..data.transforms import sobel_edges
def edge_aware_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gp = sobel_edges(pred); gt = sobel_edges(target)
    return torch.mean(torch.abs(gp - gt))
