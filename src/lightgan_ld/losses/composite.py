from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .adversarial import generator_hinge_loss
from .edge import EdgeAwareLoss
from .frequency import FocalFrequencyLoss
from .perceptual import VGGPerceptualLoss
from .structural import SSIMLoss


@dataclass
class LossBreakdown:
    total: torch.Tensor
    adv: torch.Tensor
    l1: torch.Tensor
    perceptual: torch.Tensor
    ssim: torch.Tensor
    ffl: torch.Tensor
    edge: torch.Tensor

    def as_scalars(self) -> dict[str, float]:
        return {k: float(v.detach().cpu()) for k, v in self.__dict__.items()}


class CompositeGeneratorLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.weights = cfg.get("loss", {})
        pcfg = self.weights.get("perceptual", {})
        fcfg = self.weights.get("ffl", {})
        self.perceptual = VGGPerceptualLoss(pretrained=pcfg.get("pretrained", False), weight=1.0)
        self.ssim = SSIMLoss()
        self.ffl = FocalFrequencyLoss(alpha=fcfg.get("alpha", 1.0), log_matrix=fcfg.get("log_matrix", False))
        self.edge = EdgeAwareLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, fake_logits: torch.Tensor) -> LossBreakdown:
        w = self.weights
        adv = generator_hinge_loss(fake_logits) * w.get("adv_weight", 1.0)
        l1 = F.l1_loss(pred, target) * w.get("l1_weight", 1.0)
        perc = self.perceptual(pred, target) * w.get("perceptual_weight", 0.05)
        ssim_l = self.ssim(pred, target) * w.get("ssim_weight", 0.2)
        ffl_l = self.ffl(pred, target) * w.get("ffl_weight", 0.1)
        edge_l = self.edge(pred, target) * w.get("edge_weight", 0.1)
        total = adv + l1 + perc + ssim_l + ffl_l + edge_l
        return LossBreakdown(total, adv, l1, perc, ssim_l, ffl_l, edge_l)
