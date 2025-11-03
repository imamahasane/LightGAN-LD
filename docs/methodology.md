# Methodology
- Sinogram Encoder: 3× downsample + 3× refine + bilinear upsampling.
- Generator: U-Net layout with ResidualGhostBlocks (Ghost, CondConv, ECA).
- Discriminator: Patch-based logits (hinge adversarial).
- Loss: adv + perceptual (VGG16) + SSIM + focal frequency + edge-aware.
- Training: Adam (2e-4), AMP, OneCycleLR, SWA(start@100), early stopping, MixStyle.
