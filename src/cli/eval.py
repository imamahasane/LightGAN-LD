import argparse, yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lightgan_ld.registry import build_dataset, build_models
from lightgan_ld.engine.evaluator import run_eval

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    G, _, _ = build_models(cfg, device=device)
    state = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(state["G"], strict=True)
    G.eval()

    ds = build_dataset(cfg, "val")
    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    psnr_m, ssim_m, lpips_m = run_eval(G, dl, device=device, save_visuals=cfg["eval"]["save_visuals"], out_dir=cfg["eval"]["out_dir"], amp=cfg["train"]["amp"])
    print({"psnr": psnr_m, "ssim": ssim_m, "lpips": lpips_m})

if __name__ == "__main__":
    main()
