from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from lightgan_ld.evaluators import CTEvaluator
from lightgan_ld.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot cross-dataset evaluation on Piglet or another target dataset")
    ap.add_argument("--source-config", required=True, help="Config used to build the trained source model")
    ap.add_argument("--target-config", required=True, help="Config whose test split points to Piglet HDF5")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="reports/cross_dataset")
    ap.add_argument("--name", default="source_to_target")
    args = ap.parse_args()
    source_cfg = load_config(args.source_config)
    target_cfg = load_config(args.target_config)
    source_cfg["data"] = target_cfg["data"]
    source_cfg["eval"] = target_cfg.get("eval", source_cfg.get("eval", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir) / args.name
    summary = CTEvaluator(source_cfg, args.ckpt, device).run("test", out)
    with open(Path(args.out_dir) / "cross_summary.csv", "a", newline="", encoding="utf-8") as f:
        row = {"experiment": args.name, **summary}
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    print(summary)


if __name__ == "__main__":
    main()
