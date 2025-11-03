import argparse, yaml, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from lightgan_ld.utils.seed import set_all_seeds
from lightgan_ld.utils.logging import make_tb_logger
from lightgan_ld.registry import build_dataset, build_models
from lightgan_ld.engine.trainer import TrainObjects, make_optimizers, train_step, validate, maybe_update_swa, save_ckpt
from lightgan_ld.losses.perceptual_vgg import VGGPerceptual
from lightgan_ld.utils.metrics import LPIPSMetric

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_all_seeds(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    train_ds = build_dataset(cfg, "train")
    val_ds   = build_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    cfg["_num_train_steps_per_epoch"] = max(1, len(train_loader))

    G, D, E = build_models(cfg, device=device)
    opt_G, opt_D, sch_G, sch_D = make_optimizers(G, D, cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])
    swa_model = AveragedModel(G) if cfg["train"]["swa"]["enabled"] else None
    swa_sched = SWALR(opt_G, anneal_epochs=cfg["train"]["swa"]["anneal_epochs"], anneal_strategy=cfg["train"]["swa"]["anneal_strategy"]) if cfg["train"]["swa"]["enabled"] else None

    percl = VGGPerceptual(layers=("features.16","features.23")).to(device)
    lpips_metric = LPIPSMetric()

    tb = make_tb_logger(cfg["train"]["log_dir"])
    objs = TrainObjects(G=G, D=D, E=E, opt_G=opt_G, opt_D=opt_D, sch_G=sch_G, sch_D=sch_D,
                        scaler=scaler, swa_model=swa_model, swa_sched=swa_sched,
                        perceptual=percl, lpips_metric=lpips_metric)

    best_metric = -1e9
    patience = 0
    global_step = 0

    for epoch in range(cfg["train"]["max_epochs"]):
        for batch in train_loader:
            train_step(batch, objs, cfg, global_step, tb)
            global_step += 1
            if (global_step % cfg["train"]["val_every"]) == 0:
                val = validate(val_loader, objs, cfg, global_step, tb)
                metric = val["ssim"] if cfg["train"]["early_stop"]["metric"] == "ssim" else val["psnr"]
                if metric > best_metric:
                    best_metric = metric
                    patience = 0
                    save_ckpt(G, D, E, f'{cfg["train"]["ckpt_dir"]}/best.pt')
                else:
                    patience += 1
                if cfg["train"]["early_stop"]["enabled"] and patience >= cfg["train"]["early_stop"]["patience"]:
                    print(f"Early stopping at step={global_step} with best={best_metric:.4f}")
                    save_ckpt(G, D, E, f'{cfg["train"]["ckpt_dir"]}/final.pt')
                    tb.flush()
                    return
        if cfg["train"]["swa"]["enabled"]:
            from lightgan_ld.engine.trainer import maybe_update_swa
            maybe_update_swa(objs, epoch, cfg)
        if (epoch+1) % 10 == 0:
            save_ckpt(G, D, E, f'{cfg["train"]["ckpt_dir"]}/epoch_{epoch+1}.pt')
    save_ckpt(G, D, E, f'{cfg["train"]["ckpt_dir"]}/final.pt')
    tb.flush()

if __name__ == "__main__":
    main()
