from __future__ import annotations

import argparse
import time

import torch

from lightgan_ld.models import build_models
from lightgan_ld.models.lightgan import count_parameters
from lightgan_ld.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark LightGAN-LD parameter count, FLOPs, and latency")
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, D, E = build_models(cfg); G.to(device).eval(); D.to(device).eval(); E.to(device).eval()
    size = cfg["data"].get("image_size", 256)
    x = torch.randn(1, 1, size, size, device=device)
    print({"generator_params": count_parameters(G), "discriminator_params": count_parameters(D), "encoder_params": count_parameters(E)})
    try:
        from thop import profile
        flops, params = profile(G, inputs=(x,), verbose=False)
        print({"generator_flops": int(flops), "generator_profile_params": int(params)})
    except Exception as exc:
        print(f"FLOPs skipped: {exc}")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = G(x)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = G(x)
        if device.type == "cuda": torch.cuda.synchronize()
    print({"latency_ms": (time.perf_counter() - t0) * 1000 / args.iters})


if __name__ == "__main__":
    main()
