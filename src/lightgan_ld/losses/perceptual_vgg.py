import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptual(nn.Module):
    def __init__(self, layers=("features.16","features.23"), l1=True):
        super().__init__()
        v = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        self.layers = layers
        self.net = nn.ModuleDict({str(i): v[i] for i in range(len(v))})
        for p in self.parameters(): p.requires_grad = False
        self.crit = nn.L1Loss() if l1 else nn.MSELoss()

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None,:,None,None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None,:,None,None]
        feats = {}
        h = (x - mean) / std
        for i, layer in self.net.items():
            h = layer(h)
            name = f"features.{i}"
            if name in self.layers:
                feats[name] = h
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p3 = pred.repeat(1,3,1,1)
        t3 = target.repeat(1,3,1,1)
        pf = self.encode(p3)
        tf = self.encode(t3)
        loss = 0.0
        for k in self.layers:
            loss = loss + self.crit(pf[k], tf[k])
        return loss
