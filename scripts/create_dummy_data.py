#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from lightgan_ld.datasets.hdf5_ct import create_dummy_h5


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a tiny synthetic HDF5 dataset for smoke tests")
    ap.add_argument("--out-dir", default="data/dummy")
    ap.add_argument("--image-size", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out_dir)
    create_dummy_h5(out / "train.h5", n=16, image_size=args.image_size)
    create_dummy_h5(out / "val.h5", n=8, image_size=args.image_size)
    create_dummy_h5(out / "test.h5", n=8, image_size=args.image_size)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
