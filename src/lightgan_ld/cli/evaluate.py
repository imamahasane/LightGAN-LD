from __future__ import annotations

import argparse
import torch

from lightgan_ld.evaluators import CTEvaluator
from lightgan_ld.utils.config import load_config
from lightgan_ld.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LightGAN-LD checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="reports/eval")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = CTEvaluator(cfg, args.ckpt, device)
    summary = evaluator.run(args.split, args.out_dir)
    print(summary)


if __name__ == "__main__":
    main()
