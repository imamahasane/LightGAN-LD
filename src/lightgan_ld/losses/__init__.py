from .adversarial import discriminator_hinge_loss, generator_hinge_loss
from .composite import CompositeGeneratorLoss
from .edge import EdgeAwareLoss, sobel_edges
from .frequency import FocalFrequencyLoss
from .perceptual import VGGPerceptualLoss
from .structural import SSIMLoss

__all__ = [
    "discriminator_hinge_loss", "generator_hinge_loss", "CompositeGeneratorLoss", "EdgeAwareLoss",
    "sobel_edges", "FocalFrequencyLoss", "VGGPerceptualLoss", "SSIMLoss",
]
