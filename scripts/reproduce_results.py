#!/usr/bin/env python
from __future__ import annotations

import subprocess
from pathlib import Path

COMMANDS = [
    ["lightgan-train", "--config", "configs/lodopab_full.yaml"],
    ["lightgan-eval", "--config", "configs/lodopab_full.yaml", "--ckpt", "outputs/lodopab_full/checkpoints/best.pt", "--split", "test", "--out-dir", "reports/lodopab"],
    ["lightgan-train", "--config", "configs/mayo_full.yaml"],
    ["lightgan-eval", "--config", "configs/mayo_full.yaml", "--ckpt", "outputs/mayo_full/checkpoints/best.pt", "--split", "test", "--out-dir", "reports/mayo"],
    ["lightgan-cross-eval", "--source-config", "configs/lodopab_full.yaml", "--target-config", "configs/piglet_eval.yaml", "--ckpt", "outputs/lodopab_full/checkpoints/best.pt", "--name", "lodopab_to_piglet"],
    ["lightgan-cross-eval", "--source-config", "configs/mayo_full.yaml", "--target-config", "configs/piglet_eval.yaml", "--ckpt", "outputs/mayo_full/checkpoints/best.pt", "--name", "mayo_to_piglet"],
]

Path("reports").mkdir(exist_ok=True)
for cmd in COMMANDS:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
