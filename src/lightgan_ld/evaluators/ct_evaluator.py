from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightgan_ld.datasets import build_dataset
from lightgan_ld.metrics import LPIPSMetric, mae, psnr, rmse, ssim
from lightgan_ld.models import build_models
from lightgan_ld.utils.checkpoint import load_checkpoint, load_model_weights
from lightgan_ld.utils.images import save_triplet


class CTEvaluator:
    def __init__(self, cfg: dict[str, Any], ckpt: str | Path, device: torch.device):
        self.cfg, self.device = cfg, device
        self.G, self.D, self.E = build_models(cfg)
        self.G.to(device).eval(); self.E.to(device).eval()
        state = load_checkpoint(ckpt, map_location=device)
        load_model_weights(self.G, state, "generator", strict=False)
        if "encoder" in state:
            load_model_weights(self.E, state, "encoder", strict=False)
        self.lpips = LPIPSMetric(device=device)

    @torch.no_grad()
    def run(self, split: str = "test", out_dir: str | Path = "reports/eval", max_visuals: int = 16) -> dict[str, float]:
        out_dir = Path(out_dir); (out_dir / "visuals").mkdir(parents=True, exist_ok=True)
        ds = build_dataset(self.cfg, split)
        loader = DataLoader(ds, batch_size=self.cfg["data"].get("val_batch_size", 4), shuffle=False, num_workers=self.cfg["data"].get("num_workers", 2), pin_memory=self.cfg["data"].get("pin_memory", True))
        rows = []
        use_encoder = self.cfg.get("eval", {}).get("use_encoder", False)
        for idx, batch in enumerate(tqdm(loader, desc=f"eval {split}")):
            target = batch["target"].to(self.device)
            source = batch["fbp"].to(self.device)
            if use_encoder:
                source = self.E(batch["sinogram"].to(self.device), out_size=target.shape[-2:])
            pred = self.G(source).clamp(0, 1)
            metrics = {
                "psnr": psnr(pred, target),
                "ssim": ssim(pred, target),
                "rmse": rmse(pred, target),
                "mae": mae(pred, target),
                "lpips": self.lpips(pred, target),
            }
            ids = batch.get("id", [str(idx * pred.shape[0] + i) for i in range(pred.shape[0])])
            for b in range(pred.shape[0]):
                rows.append({"id": ids[b], **{k: float(v[b].cpu()) for k, v in metrics.items()}})
                if len(rows) <= max_visuals:
                    save_triplet(source[b], pred[b], target[b], out_dir / "visuals" / f"{len(rows):04d}.png", str(ids[b]))
        with open(out_dir / f"{split}_per_slice.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id", "psnr", "ssim", "rmse", "mae", "lpips"])
            writer.writeheader(); writer.writerows(rows)
        summary = {k: float(sum(r[k] for r in rows) / max(1, len(rows))) for k in ["psnr", "ssim", "rmse", "mae", "lpips"]}
        with open(out_dir / f"{split}_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader(); writer.writerow(summary)
        return summary
