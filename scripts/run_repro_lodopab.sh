#!/usr/bin/env bash
set -e
python -m src.cli.train --config configs/experiment/lodopab_full.yaml
python -m src.cli.eval --config configs/experiment/lodopab_full.yaml --ckpt checkpoints/best.pt
