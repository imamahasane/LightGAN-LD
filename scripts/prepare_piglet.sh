#!/usr/bin/env bash
set -euo pipefail
# Piglet CT is used only as zero-shot target. Export low-dose and reference slices to arrays, then:
python scripts/standardize_h5.py --fbp data/raw/piglet_ldct.npy --target data/raw/piglet_ndct.npy --out data/piglet/test.h5
