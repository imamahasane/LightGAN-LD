# LightGAN-LD: A Lightweight Generative Adversarial Network for Efficient Low-Dose CT Reconstruction with Sinogram Encoding and Edge-Aware Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the paper:

> **“LightGAN-LD: A Lightweight Generative Adversarial Network for Efficient Low-Dose CT Reconstruction with Sinogram Encoding and Edge-Aware Learning”**  
> *Md Imam Ahasan, Guangchao Yang, A. F. M. Abdun Noor, Mohammad Azam Khan*  
> Submitted to **PeerJ Computer Science**, November 2025.

---

## Overview
Low-Dose CT (LDCT) lowers radiation exposure but introduces noise and streak artifacts. LightGAN-LD reconstructs high-fidelity images with real-time efficiency using:
- A learnable Sinogram Encoder (projection → image bootstrap)
- A lightweight Generator built from GhostModules, CondConv (input-adaptive kernels), and ECA attention
- A PatchGAN Discriminator with hinge loss
- A composite objective: adversarial + perceptual (VGG16) + SSIM + focal-frequency + edge-aware losses
- A robust training stack: AMP, OneCycleLR, SWA, MixStyle, early stopping

**Highlights**
- ~5.2M params; reference ~28 ms/slice @ 256×256 (RTX 3090)
- Reproducible PSNR/SSIM on LoDoPaB-CT & Mayo
- Modular blocks & config-driven ablations (toggle Ghost/CondConv/ECA/loss terms)

---

## Key Features

- Learnable sinogram bootstrap with configurable probability mixing vs. FBP
- Lightweight U-Net-style generator (Residual Ghost Blocks + CondConv + ECA)
- Patch Discriminator (hinge) for local realism
- Composite loss emphasizing edges and frequency fidelity
- Hydra-like YAMLs (single default.yaml here; extendable to full Hydra)
- Training ergonomics: AMP, SWA, OneCycle, TB logging, early stop
- CLI Tools: train, eval, export (TorchScript/ONNX), ablate
- CI/Tests/Docker for lab-grade reproducibility

---

## Installation

**Option A:**
```bash
conda env create -f environment.yml
conda activate lightgan-ld
```
**Option B:**
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation
LightGAN-LD expects paired (sinogram, NDCT) HDF5 files:
- Keys: sino [N,1,Hs,Ws], optional fbp [N,1,H,W], ndct [N,1,H,W]
- LoDoPaB-CT: 362×362 (training uses resize/center-crop)
- Mayo: 256×256, normalized to [0,1]

Scripts are provided as placeholders—adapt to your storage and preprocessing:
```bash
bash scripts/prepare_lodopab.sh
bash scripts/prepare_mayo.sh
```

Place resulting files as:
```bash
data/
  lodopab_train.h5
  lodopab_val.h5
  mayo_train.h5
  mayo_val.h5
```

**Download Public Datasets**
- Mayo-2016 LDCT Dataset
- LoDoPaB-CT Dataset

Tip: If you don’t have datasets ready, use the built-in DummyPairs dataset for a smoke test.
**Quickstart (Dummy Data)**
```bash
# Train on dummy data (CPU/GPU) & evaluate
python -m src.cli.train --config configs/default.yaml
python -m src.cli.eval  --config configs/default.yaml --ckpt checkpoints/best.pt
```
---

## Training on Real Data

**LoDoPaB-CT (362×362)**
```bash
# 1) Prepare data (edit script as needed)
bash scripts/prepare_lodopab.sh

# 2) Train & evaluate
python -m src.cli.train --config configs/experiment/lodopab_full.yaml
python -m src.cli.eval  --config configs/experiment/lodopab_full.yaml --ckpt checkpoints/best.pt
```

**Mayo (256×256)**
```bash
bash scripts/prepare_mayo.sh

python -m src.cli.train --config configs/experiment/mayo_full.yaml
python -m src.cli.eval  --config configs/experiment/mayo_full.yaml --ckpt checkpoints/best.pt
```

Core training configuration (see configs/default.yaml):
- Optimizer: Adam (lr=2e-4, β=(0.5, 0.999))
- Scheduler: OneCycleLR (warm-up + cosine decay)
- Precision: AMP on by default
- Stabilization: SWA (start @ epoch 100), early stop on SSIM
- Input mixing: with prob p_enc use SinogramEncoder(sino) else use FBP (Algorithm-style branch)

---

## Evaluation
```bash
python -m src.cli.eval \
  --config configs/experiment/mayo_full.yaml \
  --ckpt   checkpoints/best.pt
```

Outputs:
- Metrics: mean PSNR, SSIM, LPIPS
- Visuals (optional): input / prediction / ground-truth saved to reports/visuals/

---

## Project Structure
```bash
lightgan-ld/
├── configs/ # YAML configs (default + dataset presets)
├── docs/ # Overview, methodology, datasets, results, FAQ
├── scripts/ # Data prep, repro runners, visualization
├── src/
│ ├── cli/ # train.py, eval.py, export.py, ablate.py
│ └── lightgan_ld/
│ ├── data/ # LoDoPaB, Mayo, Dummy, transforms (MixStyle, Sobel)
│ ├── models/ # SinogramEncoder, Generator(Ghost/CondConv/ECA), Discriminator
│ ├── losses/ # hinge, perceptual(VGG16), SSIM, FFL, edge-aware
│ ├── engine/ # Trainer (AMP/OneCycle/SWA), Evaluator
│ ├── utils/ # metrics (PSNR/SSIM/LPIPS), TB logger, seeding
│ └── registry.py # builders for datasets/models
├── tests/ # Unit tests for blocks, losses, steps, IO
├── .github/workflows/ # CI and nightly smoke bench
├── Dockerfile # Repro container
├── environment.yml # Conda env
├── requirements.txt # Pip requirements
├── MODEL_CARD.md # Intended use & limitations
├── LICENSE 
└── README.md 
```
---

## Reproducibility & Good Practices

- Fixed seeds for torch, numpy, random
- Deterministic cuDNN toggles set by default
- AMP & OneCycle configured for stability from step 1
- CI runs CPU unit tests on PRs; nightly runs a smoke train/eval
- Pre-commit: Black, isort, flake8, YAML/TOML checks
- DVC (optional): version preprocessed HDF5 and reports

---

## Contact

1. Md Imam Ahasan - emamahasane@gmail.com
2. A. F. M Abdun Noor - abdunnoor11@gmail.com

---

> LightGAN-LD blends physics-aware inputs with efficient adversarial learning to deliver dose-friendly, real-time reconstruction. Safe scans, sharper images.

---

This project is released under the MIT License.
© 2025 LightGAN-LD Contributors. All rights reserved.
