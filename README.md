![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

# LightGAN-LD

**LightGAN-LD** is a lightweight and physics-informed GAN framework designed for high-quality low-dose CT reconstruction. It combines efficient architectural modules (GhostModule, MixStyle, CondConv, ECA) with advanced loss functions and domain generalization strategies, making it suitable for deployment in clinical and resource-constrained environments.

---

##  Table of Contents

- [ Features](#-features)
- [ Setup](#-setup)
- [ Training](#-training)
- [ Evaluation](#-evaluation)
- [ Directory Structure](#-directory-structure)
- [ Citation](#-citation)
- [ License](#-license)

---

##  Features

-  Modular PyTorch code (datasets, models, training loop)
-  GhostModule, CondConv, MixStyle, and ECA blocks for efficient design
-  Physics-informed loss terms (Cycle-consistency, Focal Frequency Loss, Edge Loss)
-  Mixed-precision support using `torch.amp`
-  TensorBoard integration for monitoring
-  Early stopping and OneCycleLR scheduler

---

##  Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LightGAN-LD.git
   cd LightGAN-LD

## 
