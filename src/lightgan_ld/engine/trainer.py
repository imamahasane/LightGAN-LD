from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import GradScaler, autocast
from ..losses.adversarial import d_hinge_loss, g_hinge_loss
from ..losses.perceptual_vgg import VGGPerceptual
from ..losses.ssim import ssim_loss
from ..losses.focal_frequency import focal_frequency_loss
from ..losses.edge_aware import edge_aware_l1
from ..utils.metrics import psnr, ssim, LPIPSMetric

@dataclass
class TrainObjects:
    G: nn.Module
    D: nn.Module
    E: nn.Module
    opt_G: optim.Optimizer
    opt_D: optim.Optimizer
    sch_G: Any
    sch_D: Any
    scaler: GradScaler
    swa_model: Any
    swa_sched: Any
    perceptual: VGGPerceptual
    lpips_metric: LPIPSMetric

def make_optimizers(G, D, cfg):
    opt_G = optim.Adam(G.parameters(), lr=cfg["train"]["optimizer"]["lr"],
                       betas=tuple(cfg["train"]["optimizer"]["betas"]), weight_decay=cfg["train"]["optimizer"]["weight_decay"])
    opt_D = optim.Adam(D.parameters(), lr=cfg["train"]["optimizer"]["lr"],
                       betas=tuple(cfg["train"]["optimizer"]["betas"]), weight_decay=0.0)
    sch_G = sch_D = None
    if cfg["train"]["onecycle"]["enabled"]:
        steps_per_epoch = max(1, cfg["_num_train_steps_per_epoch"])
        total_steps = steps_per_epoch * cfg["train"]["max_epochs"]
        sch_G = optim.lr_scheduler.OneCycleLR(
            opt_G, max_lr=cfg["train"]["optimizer"]["lr"],
            total_steps=total_steps,
            pct_start=cfg["train"]["onecycle"]["pct_start"],
            div_factor=cfg["train"]["onecycle"]["div_factor"],
            final_div_factor=cfg["train"]["onecycle"]["final_div_factor"]
        )
        sch_D = optim.lr_scheduler.OneCycleLR(
            opt_D, max_lr=cfg["train"]["optimizer"]["lr"],
            total_steps=total_steps,
            pct_start=cfg["train"]["onecycle"]["pct_start"],
            div_factor=cfg["train"]["onecycle"]["div_factor"],
            final_div_factor=cfg["train"]["onecycle"]["final_div_factor"]
        )
    return opt_G, opt_D, sch_G, sch_D

def train_step(batch, objs: TrainObjects, cfg, global_step: int, tb):
    s, x_fbp, y = batch
    s = s.to(cfg["device"]); x_fbp = x_fbp.to(cfg["device"]); y = y.to(cfg["device"])
    B, _, H, W = y.shape
    if cfg["model"]["sinogram_encoder"]["enabled"] and (torch.rand(1).item() < cfg["model"]["sinogram_encoder"]["penc"]):
        with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
            x = objs.E(s, out_size=H)
    else:
        x = x_fbp

    objs.opt_D.zero_grad(set_to_none=True)
    with autocast(enabled=cfg["train"]["amp"]):
        y_hat = objs.G(x)
        real_logits = objs.D(y)
        fake_logits = objs.D(y_hat.detach())
        d_loss = d_hinge_loss(real_logits, fake_logits)
    objs.scaler.scale(d_loss).backward()
    objs.scaler.step(objs.opt_D)

    objs.opt_G.zero_grad(set_to_none=True)
    with autocast(enabled=cfg["train"]["amp"]):
        fake_logits = objs.D(y_hat)
        g_adv = g_hinge_loss(fake_logits)
        g_perc = objs.perceptual(y_hat, y) * cfg["loss"]["perceptual_weight"]
        g_ssim = ssim_loss(y_hat, y) * cfg["loss"]["ssim_weight"]
        g_ffl  = focal_frequency_loss(y_hat, y) * cfg["loss"]["ffl_weight"]
        g_edge = edge_aware_l1(y_hat, y) * cfg["loss"]["edge_weight"]
        g_loss = cfg["loss"]["adv_weight"]*g_adv + g_perc + g_ssim + g_ffl + g_edge
    objs.scaler.scale(g_loss).backward()
    objs.scaler.step(objs.opt_G)
    objs.scaler.update()

    if objs.sch_G is not None and objs.sch_D is not None:
        objs.sch_G.step(); objs.sch_D.step()

    if (global_step % cfg["train"]["print_every"]) == 0:
        with torch.no_grad():
            p = psnr(y_hat, y).mean().item()
            s_v = ssim(y_hat, y).mean().item()
        tb.add_scalar("train/d_loss", d_loss.item(), global_step)
        tb.add_scalar("train/g_loss", g_loss.item(), global_step)
        tb.add_scalar("train/psnr", p, global_step)
        tb.add_scalar("train/ssim", s_v, global_step)

    return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}

@torch.no_grad()
def validate(val_loader, objs: TrainObjects, cfg, global_step: int, tb):
    objs.G.eval()
    psnrs, ssims, lpips_vals = [], [], []
    for batch in val_loader:
        s, x_fbp, y = [t.to(cfg["device"]) for t in batch]
        x = x_fbp
        with autocast(enabled=cfg["train"]["amp"]):
            y_hat = objs.G(x)
        psnrs.append(psnr(y_hat, y))
        ssims.append(ssim(y_hat, y))
        lpips_vals.append(objs.lpips_metric(y_hat, y))
    psnr_m = torch.cat(psnrs).mean().item()
    ssim_m = torch.cat(ssims).mean().item()
    lpips_m = torch.cat(lpips_vals).mean().item()
    tb.add_scalar("val/psnr", psnr_m, global_step)
    tb.add_scalar("val/ssim", ssim_m, global_step)
    tb.add_scalar("val/lpips", lpips_m, global_step)
    objs.G.train()
    return {"psnr": psnr_m, "ssim": ssim_m, "lpips": lpips_m}

def maybe_update_swa(objs: TrainObjects, epoch: int, cfg):
    if not cfg["train"]["swa"]["enabled"]:
        return
    if epoch >= cfg["train"]["swa"]["start_epoch"]:
        objs.swa_model.update_parameters(objs.G)
        if objs.swa_sched is not None:
            objs.swa_sched.step()

def save_ckpt(G, D, E, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "E": E.state_dict()}, path)
