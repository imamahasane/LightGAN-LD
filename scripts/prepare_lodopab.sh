#!/usr/bin/env bash
set -euo pipefail
# The LoDoPaB-CT archive must be downloaded under its license from Zenodo.
# Convert downloaded arrays to the standard schema with:
python scripts/standardize_h5.py \
  --sinogram data/raw/lodopab_sinograms_train.h5 --fbp data/raw/lodopab_fbp_train.h5 --target data/raw/lodopab_ground_truth_train.h5 --out data/lodopab/train.h5
python scripts/standardize_h5.py \
  --sinogram data/raw/lodopab_sinograms_val.h5 --fbp data/raw/lodopab_fbp_val.h5 --target data/raw/lodopab_ground_truth_val.h5 --out data/lodopab/val.h5
python scripts/standardize_h5.py \
  --sinogram data/raw/lodopab_sinograms_test.h5 --fbp data/raw/lodopab_fbp_test.h5 --target data/raw/lodopab_ground_truth_test.h5 --out data/lodopab/test.h5
