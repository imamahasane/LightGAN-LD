from __future__ import annotations

import csv
import logging
import random
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from lightgan_ld.datasets import build_dataset
from lightgan_ld.losses import CompositeGeneratorLoss, discriminator_hinge_loss
from lightgan_ld.metrics import LPIPSMetric, psnr, ssim, rmse, mae
from lightgan_ld.models import build_models
from lightgan_ld.utils.checkpoint import save_checkpoint
from lightgan_ld.utils.distributed import all_reduce_mean, is_main_process
from lightgan_ld.utils.images import save_triplet
from lightgan_ld.utils.logger import ExperimentLogger


def _amp_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


class GANTrainer:
    def __init__(self, cfg: dict[str, Any], device: torch.device, distributed: bool = False, rank: int = 0):
        self.cfg, self.device, self.distributed, self.rank = cfg, device, distributed, rank
        self.out_dir = Path(cfg["train"].get("output_dir", "outputs/default"))
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.report_dir = self.out_dir / "reports"
        self.vis_dir = self.out_dir / "visualizations"
        if is_main_process():
            for p in (self.ckpt_dir, self.report_dir, self.vis_dir):
                p.mkdir(parents=True, exist_ok=True)
        self.train_set = build_dataset(cfg, "train")
        self.val_set = build_dataset(cfg, "val")
        self.train_sampler = DistributedSampler(self.train_set, shuffle=True) if distributed else None
        self.val_sampler = DistributedSampler(self.val_set, shuffle=False) if distributed else None
        dc = cfg["data"]
        self.train_loader = DataLoader(self.train_set, batch_size=dc.get("batch_size", 8), shuffle=self.train_sampler is None, sampler=self.train_sampler, num_workers=dc.get("num_workers", 4), pin_memory=dc.get("pin_memory", True), drop_last=True)
        self.val_loader = DataLoader(self.val_set, batch_size=dc.get("val_batch_size", dc.get("batch_size", 8)), shuffle=False, sampler=self.val_sampler, num_workers=dc.get("num_workers", 4), pin_memory=dc.get("pin_memory", True))
        self.G, self.D, self.E = build_models(cfg)
        self.G.to(device); self.D.to(device); self.E.to(device)
        if distributed:
            self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[device.index] if device.type == "cuda" else None)
            self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[device.index] if device.type == "cuda" else None)
            self.E = torch.nn.parallel.DistributedDataParallel(self.E, device_ids=[device.index] if device.type == "cuda" else None)
        opt = cfg["train"].get("optimizer", {})
        lr = opt.get("lr", 2e-4)
        betas = tuple(opt.get("betas", [0.5, 0.999]))
        wd = opt.get("weight_decay", 0.0)
        self.opt_g = torch.optim.Adam(chain(self.G.parameters(), self.E.parameters()), lr=lr, betas=betas, weight_decay=wd)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas, weight_decay=0.0)
        total_steps = max(1, len(self.train_loader) * cfg["train"].get("epochs", 200))
        if cfg["train"].get("onecycle", {}).get("enabled", True):
            oc = cfg["train"].get("onecycle", {})
            self.sch_g = torch.optim.lr_scheduler.OneCycleLR(self.opt_g, max_lr=lr, total_steps=total_steps, pct_start=oc.get("pct_start", 0.1), div_factor=oc.get("div_factor", 10), final_div_factor=oc.get("final_div_factor", 100))
            self.sch_d = torch.optim.lr_scheduler.OneCycleLR(self.opt_d, max_lr=lr, total_steps=total_steps, pct_start=oc.get("pct_start", 0.1), div_factor=oc.get("div_factor", 10), final_div_factor=oc.get("final_div_factor", 100))
        else:
            self.sch_g = self.sch_d = None
        self.amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.loss_fn = CompositeGeneratorLoss(cfg).to(device)
        self.lpips = LPIPSMetric(device=device)
        self.swa_enabled = cfg["train"].get("swa", {}).get("enabled", True)
        self.swa_start = cfg["train"].get("swa", {}).get("start_epoch", 100)
        self.swa_g = AveragedModel(self.G) if self.swa_enabled else None
        self.swa_e = AveragedModel(self.E) if self.swa_enabled else None
        self.logger = ExperimentLogger(str(self.out_dir / "tb"), cfg["train"].get("wandb", {}).get("enabled", False) and is_main_process(), cfg["train"].get("wandb", {}).get("project", "lightgan-ld"), cfg) if is_main_process() else None
        self.best = -1e9
        self.global_step = 0

    def _select_input(self, batch: dict[str, Any], train: bool) -> torch.Tensor:
        target = batch["target"].to(self.device, non_blocking=True)
        if train and self.cfg["model"].get("sinogram_encoder", {}).get("enabled", True):
            if random.random() < self.cfg["model"]["sinogram_encoder"].get("penc", 0.5):
                return self.E(batch["sinogram"].to(self.device, non_blocking=True), out_size=target.shape[-2:])
        return batch["fbp"].to(self.device, non_blocking=True)

    def train_one_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.G.train(); self.D.train(); self.E.train()
        target = batch["target"].to(self.device, non_blocking=True)
        x = self._select_input(batch, train=True)
        device_type = self.device.type
        self.opt_d.zero_grad(set_to_none=True)
        with _amp_context(device_type, self.amp_enabled):
            pred = self.G(x)
            d_loss = discriminator_hinge_loss(self.D(target), self.D(pred.detach()))
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.opt_d)
        self.opt_g.zero_grad(set_to_none=True)
        with _amp_context(device_type, self.amp_enabled):
            fake_logits = self.D(pred)
            breakdown = self.loss_fn(pred, target, fake_logits)
        self.scaler.scale(breakdown.total).backward()
        if self.cfg["train"].get("grad_clip_norm", 0) > 0:
            self.scaler.unscale_(self.opt_g)
            nn.utils.clip_grad_norm_(chain(self.G.parameters(), self.E.parameters()), self.cfg["train"]["grad_clip_norm"])
        self.scaler.step(self.opt_g)
        self.scaler.update()
        if self.sch_g is not None:
            self.sch_g.step(); self.sch_d.step()
        scalars = breakdown.as_scalars()
        scalars["d_loss"] = float(d_loss.detach().cpu())
        with torch.no_grad():
            scalars["psnr"] = float(psnr(pred, target).mean().detach().cpu())
            scalars["ssim"] = float(ssim(pred, target).mean().detach().cpu())
        return scalars

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.G.eval(); self.E.eval()
        vals: dict[str, list[torch.Tensor]] = {"psnr": [], "ssim": [], "rmse": [], "mae": [], "lpips": []}
        iterator = tqdm(self.val_loader, disable=not is_main_process(), desc=f"val {epoch}")
        for i, batch in enumerate(iterator):
            target = batch["target"].to(self.device, non_blocking=True)
            source = batch["fbp"].to(self.device, non_blocking=True)
            if self.cfg.get("eval", {}).get("use_encoder", False):
                source = self.E(batch["sinogram"].to(self.device, non_blocking=True), out_size=target.shape[-2:])
            pred = self.G(source).clamp(0, 1)
            vals["psnr"].append(psnr(pred, target))
            vals["ssim"].append(ssim(pred, target))
            vals["rmse"].append(rmse(pred, target))
            vals["mae"].append(mae(pred, target))
            vals["lpips"].append(self.lpips(pred, target))
            if is_main_process() and i == 0:
                save_triplet(source[0], pred[0], target[0], self.vis_dir / f"epoch_{epoch:04d}.png")
        out = {}
        for k, items in vals.items():
            if not items:
                continue
            v = torch.cat(items).mean()
            out[k] = float(all_reduce_mean(v).cpu())
        if self.logger:
            for k, v in out.items():
                self.logger.scalar(f"val/{k}", v, self.global_step)
        return out

    def fit(self) -> None:
        epochs = self.cfg["train"].get("epochs", 200)
        patience = self.cfg["train"].get("early_stop", {}).get("patience", 10)
        monitor = self.cfg["train"].get("early_stop", {}).get("metric", "ssim")
        bad = 0
        for epoch in range(1, epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            iterator = tqdm(self.train_loader, disable=not is_main_process(), desc=f"train {epoch}")
            for batch in iterator:
                scalars = self.train_one_step(batch)
                if self.logger and self.global_step % self.cfg["train"].get("log_every", 50) == 0:
                    for k, v in scalars.items():
                        self.logger.scalar(f"train/{k}", v, self.global_step)
                self.global_step += 1
            if self.swa_enabled and epoch >= self.swa_start:
                self.swa_g.update_parameters(self.G)
                self.swa_e.update_parameters(self.E)
            val = self.validate(epoch)
            metric = val.get(monitor, val.get("ssim", 0.0))
            if is_main_process():
                row = {"epoch": epoch, "step": self.global_step, **val}
                self._append_csv(self.report_dir / "validation.csv", row)
                if metric > self.best:
                    self.best = metric; bad = 0
                    save_checkpoint(self.ckpt_dir / "best.pt", generator=self.G, discriminator=self.D, encoder=self.E, opt_g=self.opt_g, opt_d=self.opt_d, scaler=self.scaler, epoch=epoch, step=self.global_step, best_metric=self.best, cfg=self.cfg)
                else:
                    bad += 1
                if epoch % self.cfg["train"].get("save_every_epochs", 10) == 0:
                    save_checkpoint(self.ckpt_dir / f"epoch_{epoch:04d}.pt", generator=self.G, discriminator=self.D, encoder=self.E, epoch=epoch, step=self.global_step, best_metric=self.best, cfg=self.cfg)
                if self.cfg["train"].get("early_stop", {}).get("enabled", True) and bad >= patience:
                    logging.info("Early stopping at epoch %s; best %s=%.5f", epoch, monitor, self.best)
                    break
        if is_main_process():
            save_checkpoint(self.ckpt_dir / "final.pt", generator=self.G, discriminator=self.D, encoder=self.E, opt_g=self.opt_g, opt_d=self.opt_d, scaler=self.scaler, epoch=epochs, step=self.global_step, best_metric=self.best, cfg=self.cfg)
            if self.swa_enabled and self.swa_g is not None:
                try:
                    update_bn(self.train_loader, self.swa_g, device=self.device)
                except Exception:
                    pass
                save_checkpoint(self.ckpt_dir / "swa.pt", generator=self.swa_g, discriminator=self.D, encoder=self.swa_e or self.E, epoch=epochs, step=self.global_step, best_metric=self.best, cfg=self.cfg)
            if self.logger:
                self.logger.close()

    @staticmethod
    def _append_csv(path: Path, row: dict[str, Any]) -> None:
        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
