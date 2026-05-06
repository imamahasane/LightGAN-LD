from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lightgan_ld.datasets.transforms import normalize01, resize_tensor
from lightgan_ld.inference import load_lightgan_checkpoint, reconstruct_tensor
from lightgan_ld.utils.config import load_config


def _load_array(path: str) -> torch.Tensor:
    arr = np.load(path).astype("float32")
    arr = normalize01(arr)
    ten = torch.from_numpy(arr)
    if ten.ndim == 2:
        ten = ten[None, None]
    elif ten.ndim == 3:
        ten = ten[None]
    return ten.float()


def _save_png(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.squeeze().numpy()
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LightGAN-LD inference on a .npy image/sinogram")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-kind", choices=["fbp", "sinogram"], default="fbp")
    ap.add_argument("--image-size", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, encoder = load_lightgan_checkpoint(cfg, args.ckpt, device)
    x = _load_array(args.input)
    if args.input_kind == "fbp":
        size = args.image_size or cfg["data"].get("image_size", 256)
        x = resize_tensor(x, size)
    y = reconstruct_tensor(generator, encoder, x, device=device, input_kind=args.input_kind, out_size=args.image_size or cfg["data"].get("image_size", 256), amp=cfg["train"].get("amp", True))
    _save_png(y[0], args.output)
    print(args.output)


if __name__ == "__main__":
    main()
