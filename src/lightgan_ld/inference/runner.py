from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lightgan_ld.models import build_models
from lightgan_ld.utils.checkpoint import load_checkpoint


def load_lightgan_checkpoint(cfg: dict[str, Any], ckpt_path: str | Path, device: torch.device):
    """Load generator and sinogram encoder for deployment/inference."""
    generator, _, encoder = build_models(cfg)
    generator.to(device).eval()
    encoder.to(device).eval()
    state = load_checkpoint(ckpt_path, map_location=device)
    generator.load_state_dict(state["generator"], strict=True)
    if "encoder" in state:
        encoder.load_state_dict(state["encoder"], strict=False)
    return generator, encoder


@torch.no_grad()
def reconstruct_tensor(
    generator: torch.nn.Module,
    encoder: torch.nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    input_kind: str = "fbp",
    out_size: int | tuple[int, int] | None = None,
    amp: bool = True,
) -> torch.Tensor:
    """Run single-pass LightGAN-LD inference on an FBP image or sinogram tensor."""
    x = x.to(device)
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[None]
    enabled = amp and device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, enabled=enabled):
        source = encoder(x, out_size=out_size or x.shape[-2:]) if input_kind == "sinogram" else x
        y = generator(source).clamp(0, 1)
    return y.detach().cpu()
