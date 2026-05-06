from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers: tuple[int, ...] = (8, 15, 22), pretrained: bool = False, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        try:
            from torchvision.models import VGG16_Weights, vgg16
            weights = VGG16_Weights.DEFAULT if pretrained else None
            features = vgg16(weights=weights).features.eval()
            self.blocks = nn.ModuleList()
            start = 0
            for end in layers:
                self.blocks.append(features[start:end])
                start = end
            for p in self.parameters():
                p.requires_grad_(False)
            self.enabled = True
        except Exception:
            self.blocks = nn.ModuleList()
            self.enabled = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False) if x.shape[-1] < 224 else x
        return (x.clamp(0, 1) - self.mean.to(x)) / self.std.to(x)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.weight <= 0:
            return pred.new_tensor(0.0)
        x = self._prep(pred)
        y = self._prep(target)
        loss = pred.new_tensor(0.0)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return loss * self.weight
