from __future__ import annotations

from typing import Any

from torch import nn

from .discriminator import PatchDiscriminator
from .generator import LightGANLDGenerator
from .sinogram_encoder import SinogramEncoder


class LightGANLD(nn.Module):
    """Container module holding encoder, generator, and discriminator."""

    def __init__(self, generator: LightGANLDGenerator, discriminator: PatchDiscriminator, encoder: SinogramEncoder):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder


def build_models(cfg: dict[str, Any]) -> tuple[LightGANLDGenerator, PatchDiscriminator, SinogramEncoder]:
    mc = cfg.get("model", {})
    gcfg = mc.get("generator", {})
    dcfg = mc.get("discriminator", {})
    ecfg = mc.get("sinogram_encoder", {})
    generator = LightGANLDGenerator(**gcfg)
    discriminator = PatchDiscriminator(**dcfg)
    encoder = SinogramEncoder(**{k: v for k, v in ecfg.items() if k not in {"enabled", "penc"}})
    return generator, discriminator, encoder


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
