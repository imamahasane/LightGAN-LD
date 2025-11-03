# LightGAN-LD: Lightweight GAN for Low-Dose CT Reconstruction

Fast, high-fidelity LDCT reconstruction with a **Sinogram Encoder**, lightweight **Generator** (Ghost/CondConv/ECA), and **PatchGAN** discriminator. Composite loss (adv + perceptual + SSIM + focal-frequency + edge-aware). Trained with **AMP + OneCycleLR + SWA + MixStyle**.

## Quickstart
```bash
conda env create -f environment.yml
conda activate lightgan-ld
# Or pip install -r requirements.txt
python -m src.cli.train --config configs/default.yaml
python -m src.cli.eval --config configs/default.yaml --ckpt checkpoints/best.pt
```

## Structure
- `src/lightgan_ld/models/` — Sinogram Encoder, Generator (Ghost+CondConv+ECA), Patch Discriminator.
- `src/lightgan_ld/losses/` — adv (hinge), VGG perceptual, SSIM, FFL, edge-aware.
- `src/lightgan_ld/engine/` — Algorithm 1 training loop + AMP/OneCycle/SWA, evaluator.
- `docs/` — methodology, datasets, results reproduction, FAQ.

**Research only; not for clinical use.**
