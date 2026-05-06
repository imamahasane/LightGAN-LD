from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def to_numpy01(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().squeeze().numpy()
    return np.clip(arr, 0.0, 1.0)


def save_triplet(low: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, path: str | Path, title: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    imgs = [to_numpy01(low), to_numpy01(pred), to_numpy01(target), np.abs(to_numpy01(pred) - to_numpy01(target))]
    names = ["Input", "LightGAN-LD", "Target", "Abs error"]
    for ax, img, name in zip(axes, imgs, names):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def soft_tissue_window_hu(x: np.ndarray, width: float = 400.0, level: float = 40.0) -> np.ndarray:
    low = level - width / 2
    high = level + width / 2
    return np.clip((x - low) / (high - low), 0.0, 1.0)
