# LightGAN-LD: A Lightweight Generative Adversarial Network for Efficient Low-Dose CT Reconstruction with Sinogram Encoding and Edge-Aware Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the paper:

> **“LightGAN-LD: A Lightweight Generative Adversarial Network for Efficient Low-Dose CT Reconstruction with Sinogram Encoding and Edge-Aware Learning”**  
> *Md Imam Ahasan, Guangchao Yang, A. F. M. Abdun Noor, Mohammad Azam Khan*  
> Submitted to **PeerJ Computer Science (AI Application category)**, November 2025.

---

## 1. Project Description

**LightGAN-LD** is a lightweight **Generative Adversarial Network (GAN)** for **low-dose CT (LDCT) image reconstruction**. It minimizes radiation exposure while preserving diagnostic image quality, emphasizing computational efficiency and reproducibility.

**Core features :**
- **Sinogram Encoder** for geometry-aware projection-to-image inversion.
- **Lightweight Generator** integrating:
  - GhostModules (efficient convolutional feature extraction),
  - CondConv (input-adaptive kernels),
  - ECA attention (lightweight channel reweighting).
- **PatchGAN Discriminator** for local realism.
- **Composite loss** combining adversarial, perceptual (VGG16), SSIM, focal frequency, and edge-aware components.
- **Training optimizations:** AMP, OneCycleLR, Stochastic Weight Averaging (SWA), MixStyle, early stopping.

LightGAN-LD achieves **high-fidelity reconstructions** (up to +1.9 dB PSNR gain) using **<5.2M parameters** and runs in **≈28 ms per 256×256 slice** on an RTX 3090 GPU.

---

## 2. Dataset Information

This project uses two publicly available benchmark datasets.  You must obtain them separately under their respective terms.

**LoDoPaB-CT Dataset :**
- **Reference:** Leuschner et al., *Scientific Data* 8, 109 (2021)  
- **DOI:** https://doi.org/10.1038/s41597-021-00893-1  
- **Website:** https://lodopab.in.tum.de/  
- **Description:** Synthetic LDCT dataset derived from the LIDC-IDRI thoracic CT collection.  
- **Usage here:**  
  - 40 000 training, 3 500 validation, 3 500 test slices  
  - Sinogram → NDCT image pairs  
  - 362×362 → 256×256 (resized)  
  - Normalized to [0, 1]

**NIH–AAPM–Mayo Low-Dose CT Dataset :**
- **Reference:** Moen et al., *Medical Physics* 48(2):902–911 (2021)  
- **URL:** https://www.aapm.org/grandchallenge/lowdosect/  
- **Description:** Abdominal CT volumes with both normal-dose (NDCT) and simulated low-dose (LDCT) images.  
- **Usage here:**  
  - 8 patients for training (~4 800 slices)  
  - 2 patients for testing (~1 100 slices)  
  - Resized to 256×256, normalized to [0, 1]

---

## 3. Code Information

**Repository layout :**
```bash
LightGAN-LD/
├── configs/                  # YAML experiment configs
│   ├── default.yaml
│   └── experiment/
│       ├── lodopab_full.yaml
│       └── mayo_full.yaml
├── scripts/
│   ├── prepare_lodopab.sh    # LoDoPaB preprocessing template
│   ├── prepare_mayo.sh       # Mayo preprocessing template
│   └── reproduce_results.py  # Reproducibility script (see below)
├── src/
│   ├── cli/                  # Entry points
│   │   ├── train.py
│   │   └── eval.py
│   └── lightgan_ld/
│       ├── data/             # Datasets & transforms
│       ├── models/           # SinogramEncoder, Generator, Discriminator
│       ├── losses/           # Adversarial, SSIM, FFL, Edge-aware, etc.
│       ├── engine/           # Trainer (AMP/SWA/OneCycle)
│       ├── utils/            # Metrics, logging, seeding
│       └── registry.py
├── tests/                    # Unit tests
├── environment.yml           # Conda environment
├── requirements.txt          # Pip dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## 4. Installation

**Option A :**
```bash
conda env create -f environment.yml
conda activate lightgan-ld
```
**Option B :**
```bash
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

**Workflow :**
```bash
# LoDoPaB-CT
bash scripts/prepare_lodopab.sh

# NIH–AAPM–Mayo
bash scripts/prepare_mayo.sh
```

**Place resulting files as :**
```bash
data/
  lodopab_train.h5
  lodopab_val.h5
  mayo_train.h5
  mayo_val.h5
```
---

## 6. Quickstart
**Run a smoke test :**
```bash
python -m src.cli.train --config configs/default.yaml
python -m src.cli.eval  --config configs/default.yaml --ckpt checkpoints/best.pt
```
**Train on LoDoPaB-CT :**
```bash
bash scripts/prepare_lodopab.sh
python -m src.cli.train --config configs/experiment/lodopab_full.yaml
python -m src.cli.eval  --config configs/experiment/lodopab_full.yaml --ckpt checkpoints/best.pt
```
**Train on NIH–AAPM–Mayo :**
```bash
bash scripts/prepare_mayo.sh
python -m src.cli.train --config configs/experiment/mayo_full.yaml
python -m src.cli.eval  --config configs/experiment/mayo_full.yaml --ckpt checkpoints/best.pt
```
---
## 7. Requirements

To ensure reproducibility and compatibility, install the following dependencies:

```bash
# Core environment
Python >= 3.10
PyTorch >= 2.4
torchvision >= 0.19
CUDA Toolkit 12.2   # for GPU acceleration

# Scientific libraries
numpy >= 1.24
h5py >= 3.10
scikit-image >= 0.22

# Metrics and visualization
lpips >= 0.1
tqdm >= 4.65
tensorboard >= 2.0

# Optional (for CI and reproducibility)
matplotlib
pandas
wandb
flake8
black
isort
```
**Recommended setup :**
```bash
# Using conda
conda env create -f environment.yml
conda activate lightgan-ld

# or using pip
pip install -r requirements.txt
```
**Tip:** GPU with 24 GB VRAM (NVIDIA RTX 3090) is recommended for full-scale experiments. CPU-only runs are supported for testing and evaluation.

---
## 8. Methodology Summary

The LightGAN-LD training and reconstruction framework consists of three main components:  
- A geometry-aware *Sinogram Encoder*,  
- A lightweight *Generator–Discriminator* pair, and  
- A composite multi-objective *loss function* with efficient optimization.

**Input Strategy :**
- Low-dose sinogram \( S \) is processed by a **probabilistic dual-branch** pathway:
  - With probability \( p_{enc} \): a *learnable Sinogram Encoder* \( E_{\psi}(S) \) performs projection-to-image mapping.  
  - Otherwise: *Filtered Back Projection (FBP)* is applied.  
- This hybrid approach enhances robustness under sparse-view or noisy conditions.

**Generator Architecture :**
- Encoder–decoder backbone with:
  - **GhostModules** → efficient feature generation via intrinsic and cheap convolutions.  
  - **CondConv** → conditionally parameterized filters adapting to each input.  
  - **ECA (Efficient Channel Attention)** → lightweight 1D convolution-based channel weighting.  
  - **Residual Ghost Blocks** → maintain structural integrity and stable gradients.
- Parameter count: **< 5.2 million**.

**Discriminator :**
- Based on **PatchGAN** (fully convolutional).  
- Equipped with GhostModules and stride-2 average pooling.  
- Outputs patch-wise realism scores \( D_{\phi}(X) \in \mathbb{R}^{1 \times H' \times W'} \).  
- Encourages *local structural realism* and suppresses blurry artifacts.

**Loss Functions :**
A composite objective combines multiple complementary terms:

| Loss Term | Description | Purpose |
|------------|-------------|----------|
| \( L_{adv} \) | Adversarial (hinge) | Match generated & real distributions |
| \( L_{perc} \) | Perceptual (VGG16) | Maintain perceptual similarity |
| \( L_{SSIM} \) | Structural Similarity | Enforce structural consistency |
| \( L_{FFL} \) | Focal Frequency Loss | Preserve high-frequency detail |
| \( L_{edge} \) | Edge-Aware (Sobel) | Maintain anatomical boundaries |

Final generator loss:
\[
L_G = L_{adv} + \lambda_{perc} L_{perc} + \lambda_{SSIM} L_{SSIM} + \lambda_{FFL} L_{FFL} + \lambda_{edge} L_{edge}
\]

**Optimization Strategy :**
- Optimizer: **Adam** (lr = 2×10⁻⁴, β₁ = 0.5, β₂ = 0.999)  
- Scheduler: **OneCycleLR** (warm-up + cosine decay)  
- Mixed precision: **AMP** for faster training  
- Stability: **SWA (Stochastic Weight Averaging)** and **early stopping** on SSIM metric  
- Regularization: **MixStyle** for domain generalization

**Evaluation Metrics :**
Model performance is quantitatively assessed using:
- **PSNR (Peak Signal-to-Noise Ratio)** — measures fidelity.  
- **SSIM (Structural Similarity Index)** — measures perceptual similarity.  
- **LPIPS (Learned Perceptual Image Patch Similarity)** — measures perceptual quality.

**Summary :**
LightGAN-LD achieves superior reconstruction performance by integrating physics-aware sinogram encoding with computationally efficient adversarial learning.  Its balanced design enables **real-time inference**, **cross-domain robustness**, and **clinically deployable reconstruction quality**.

---
## 9. Reproducibility & Reproduction Script

LightGAN-LD is designed to meet the **PeerJ Computer Science AI Application reproducibility standards**.  This repository provides all necessary configuration files, training scripts, and evaluation pipelines to replicate the results presented in the manuscript.

**Reproducibility Checklist :**
- Fixed random seeds (for Python, NumPy, PyTorch)  
- Deterministic cuDNN settings (optional)  
- Configuration-driven experiments via `configs/*.yaml`  
- Fully documented data preprocessing scripts in `scripts/`  
- Evaluation metrics consistent with the paper (PSNR, SSIM, LPIPS)  
- Pretrained weights available (see `checkpoints/` or Zenodo link when released)  
- TensorBoard logging and optional Weights & Biases integration  

**Reproduce Main Results :**
Once datasets and pretrained checkpoints are available, you can reproduce the main benchmark results reported in the paper (Table 1).
```bash
# Run reproduction script
python scripts/reproduce_results.py
```
**Expected Outputs :**
```bash
reports/reproduction_results.csv
```

| Dataset       | PSNR (dB)    | SSIM          | LPIPS |
| ------------- | ------------ | ------------- | ----- |
| LoDoPaB-CT    | 36.93 ± 0.04 | 0.896 ± 0.002 | 0.078 |
| NIH–AAPM–Mayo | 35.98 ± 0.05 | 0.899 ± 0.003 | 0.074 |

**Tip:** For deterministic replication, use the same environment as defined in `environment.yml.` All reported results in the paper were generated with Python 3.10, PyTorch 2.4, and CUDA 12.2.

---

## 10. Citation
```bash
@article{Ahasan2025LightGANLD,
  title   = {LightGAN-LD: A Lightweight Generative Adversarial Network for Efficient Low-Dose CT Reconstruction with Sinogram Encoding and Edge-Aware Learning},
  author  = {Md Imam Ahasan, Guangchao Yang, A. F. M. Abdun Noor, Mohammad Azam Khan},
  journal = {PeerJ Computer Science},
  year    = {2025},
  note    = {Under review}
}
```

## 11. License & Contributions

**License :**
Released under the MIT License. © 2025 LightGAN-LD Authors. All rights reserved.

**Contribution Guidelines :**
We welcome pull requests and improvements.


## 12. Contact

1. Md Imam Ahasan - emamahasane@gmail.com
2. A. F. M Abdun Noor - abdunnoor11@gmail.com

---

## 13. Acknowledgements
This work was conducted at the **College of Computer Science, Chongqing University** and the **Department of Software Engineering, Daffodil International University.**

All scientific content, data processing, and results were **independently verified and approved** by the authors.

---

> LightGAN-LD unites geometry-aware encoding with efficient adversarial learning toward dose-friendly, real-time CT reconstruction.

---
