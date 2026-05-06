#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_any(path: Path, key: str | None):
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as h5:
            return np.asarray(h5[key or list(h5.keys())[0]])
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data[key or list(data.keys())[0]]
    raise ValueError(f"Unsupported file: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Standardize arrays into LightGAN-LD HDF5 schema")
    ap.add_argument("--sinogram")
    ap.add_argument("--fbp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sinogram-key")
    ap.add_argument("--fbp-key")
    ap.add_argument("--target-key")
    args = ap.parse_args()
    fbp = read_any(Path(args.fbp), args.fbp_key).astype("float32")
    target = read_any(Path(args.target), args.target_key).astype("float32")
    sino = read_any(Path(args.sinogram), args.sinogram_key).astype("float32") if args.sinogram else fbp
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h5:
        h5.create_dataset("sinogram", data=sino, compression="gzip")
        h5.create_dataset("fbp", data=fbp, compression="gzip")
        h5.create_dataset("target", data=target, compression="gzip")
    print(args.out)


if __name__ == "__main__":
    main()
