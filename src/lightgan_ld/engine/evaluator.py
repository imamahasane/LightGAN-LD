from pathlib import Path
import torch
from torch.cuda.amp import autocast
from ..utils.metrics import psnr, ssim, LPIPSMetric
from PIL import Image

@torch.no_grad()
def run_eval(G, loader, device="cuda", save_visuals=False, out_dir="./reports", amp=True):
    lpips_metric = LPIPSMetric()
    G.eval()
    psnrs, ssims, lpips_vals = [], [], []
    vis_dir = Path(out_dir) / "visuals"
    if save_visuals:
        vis_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for s, x_fbp, y in loader:
        x_fbp = x_fbp.to(device); y = y.to(device)
        with autocast(enabled=amp):
            y_hat = G(x_fbp)
        psnrs.append(psnr(y_hat, y))
        ssims.append(ssim(y_hat, y))
        lpips_vals.append(lpips_metric(y_hat, y))
        if save_visuals and idx < 16:
            for b in range(min(4, y.shape[0])):
                _save_img((y[b,0]*255).clamp(0,255).byte().cpu().numpy(), vis_dir / f"{idx:04d}_gt.png")
                _save_img((x_fbp[b,0]*255).clamp(0,255).byte().cpu().numpy(), vis_dir / f"{idx:04d}_input.png")
                _save_img((y_hat[b,0]*255).clamp(0,255).byte().cpu().numpy(), vis_dir / f"{idx:04d}_pred.png")
                idx += 1
    return float(torch.cat(psnrs).mean()), float(torch.cat(ssims).mean()), float(torch.cat(lpips_vals).mean())

def _save_img(arr, path):
    Image.fromarray(arr).save(path)
