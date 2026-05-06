from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def ensure_chw(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None]
    elif x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
        x = np.moveaxis(x, -1, 0)
    return x.astype(np.float32, copy=False)


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.all():
        x = np.nan_to_num(x)
    mn, mx = float(x.min()), float(x.max())
    if 0.0 <= mn and mx <= 1.0:
        return x
    return (x - mn) / max(mx - mn, eps)


def clip_or_window(x: np.ndarray, window: tuple[float, float] | None = None, clip: tuple[float, float] | None = None) -> np.ndarray:
    if window is not None:
        width, level = window
        low, high = level - width / 2.0, level + width / 2.0
        return np.clip((x - low) / (high - low), 0.0, 1.0).astype(np.float32)
    if clip is not None:
        lo, hi = clip
        x = np.clip(x, lo, hi)
    return normalize01(x)


def resize_tensor(x: torch.Tensor, size: int | tuple[int, int] | None, mode: str = "bilinear") -> torch.Tensor:
    if size is None:
        return x
    if isinstance(size, int):
        size = (size, size)
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    batched = x.ndim == 4
    if not batched:
        x = x.unsqueeze(0)
    x = F.interpolate(x, size=size, mode=mode, align_corners=False if mode in {"bilinear", "bicubic"} else None)
    return x if batched else x.squeeze(0)


@dataclass
class CTTransform:
    image_size: int | None = 256
    normalize: bool = True
    window: tuple[float, float] | None = None
    clip: tuple[float, float] | None = None
    augment: bool = False

    def __call__(self, arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, arr in arrays.items():
            arr = ensure_chw(arr)
            if self.normalize:
                arr = clip_or_window(arr, self.window if key != "sinogram" else None, self.clip)
            ten = torch.from_numpy(np.ascontiguousarray(arr)).float()
            if key != "sinogram":
                ten = resize_tensor(ten, self.image_size)
            out[key] = ten
        if self.augment and "target" in out:
            if torch.rand(()) < 0.5:
                for key in ("fbp", "low_dose", "target"):
                    if key in out:
                        out[key] = torch.flip(out[key], dims=(-1,))
            if torch.rand(()) < 0.5:
                for key in ("fbp", "low_dose", "target"):
                    if key in out:
                        out[key] = torch.flip(out[key], dims=(-2,))
        return out
