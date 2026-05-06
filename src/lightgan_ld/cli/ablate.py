from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

from lightgan_ld.utils.config import load_config, save_config


VARIANTS = {
    "unet_conv": {"model.generator.use_ghost": False, "model.generator.use_condconv": False, "model.generator.use_eca": False, "model.generator.mixstyle_p": 0.0},
    "ghost": {"model.generator.use_ghost": True, "model.generator.use_condconv": False, "model.generator.use_eca": False},
    "ghost_condconv": {"model.generator.use_ghost": True, "model.generator.use_condconv": True, "model.generator.use_eca": False},
    "full": {"model.generator.use_ghost": True, "model.generator.use_condconv": True, "model.generator.use_eca": True},
    "no_ffl": {"loss.ffl_weight": 0.0},
    "no_edge": {"loss.edge_weight": 0.0},
    "adv_only": {"loss.l1_weight": 0.0, "loss.perceptual_weight": 0.0, "loss.ssim_weight": 0.0, "loss.ffl_weight": 0.0, "loss.edge_weight": 0.0},
}


def set_dot(cfg, key, value):
    cur = cfg
    parts = key.split('.')
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch LightGAN-LD ablation experiments")
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=["unet_conv", "ghost", "ghost_condconv", "full"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    base = load_config(args.config)
    for name in args.variants:
        cfg = copy.deepcopy(base)
        for k, v in VARIANTS[name].items():
            set_dot(cfg, k, v)
        cfg["train"]["output_dir"] = str(Path(base["train"].get("output_dir", "outputs/ablation")) / name)
        path = Path(cfg["train"]["output_dir"]) / "config.yaml"
        save_config(cfg, path)
        cmd = ["lightgan-train", "--config", str(path)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
