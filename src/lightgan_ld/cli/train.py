from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from lightgan_ld.trainers import GANTrainer
from lightgan_ld.utils.config import load_config, save_config
from lightgan_ld.utils.distributed import cleanup_distributed, setup_distributed
from lightgan_ld.utils.logger import setup_text_logger
from lightgan_ld.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGAN-LD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="Dotlist override, e.g. train.epochs=10")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    distributed, rank, world, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.get("seed", 42)) + rank, bool(cfg.get("deterministic", True)))
    out_dir = Path(cfg["train"].get("output_dir", "outputs/default"))
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "resolved_config.yaml")
    setup_text_logger(str(out_dir / "train.log") if rank == 0 else None)
    logging.info("Starting training on rank %s/%s with device=%s", rank, world, device)
    trainer = GANTrainer(cfg, device, distributed=distributed, rank=rank)
    trainer.fit()
    cleanup_distributed()


if __name__ == "__main__":
    main()
