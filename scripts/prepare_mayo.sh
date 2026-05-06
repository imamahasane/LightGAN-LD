#!/usr/bin/env bash
set -euo pipefail
# NIH-AAPM-Mayo data requires registration. After exporting LDCT/NDCT arrays, standardize them:
python scripts/standardize_h5.py --fbp data/raw/mayo_ldct_train.npy --target data/raw/mayo_ndct_train.npy --out data/mayo/train.h5
python scripts/standardize_h5.py --fbp data/raw/mayo_ldct_val.npy --target data/raw/mayo_ndct_val.npy --out data/mayo/val.h5
python scripts/standardize_h5.py --fbp data/raw/mayo_ldct_test.npy --target data/raw/mayo_ndct_test.npy --out data/mayo/test.h5
