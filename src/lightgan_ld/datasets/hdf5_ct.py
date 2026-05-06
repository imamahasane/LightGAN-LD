from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import CTTransform, ensure_chw, normalize01

_KEY_ALIASES = {
    "sinogram": ("sinogram", "sino", "s", "ld_sino", "sparse_sino"),
    "fbp": ("fbp", "low_fbp", "ldct", "low_dose", "input", "x"),
    "target": ("target", "ndct", "normal_dose", "gt", "y", "clean"),
    "patient_id": ("patient_id", "patient", "case_id"),
    "slice_id": ("slice_id", "slice", "index"),
}


def _find_key(h5: h5py.File, semantic: str, explicit: str | None = None) -> str | None:
    if explicit and explicit in h5:
        return explicit
    for key in _KEY_ALIASES[semantic]:
        if key in h5:
            return key
    return None


@dataclass
class DatasetSpec:
    path: str
    image_size: int = 256
    sinogram_key: str | None = None
    fbp_key: str | None = None
    target_key: str | None = None
    normalize: bool = True
    window: tuple[float, float] | None = None
    clip: tuple[float, float] | None = None
    augment: bool = False


class HDF5CTDataset(Dataset[dict[str, Any]]):
    """Standardized paired CT dataset backed by HDF5.

    Expected HDF5 datasets can use canonical keys (`sinogram`, `fbp`, `target`) or common aliases.
    The returned dictionary always contains `sinogram`, `fbp`, `target`, and `id`. If sinograms are
    unavailable, the loader uses the FBP/LDCT image as `sinogram` so image-only datasets can still run
    with the sinogram encoder disabled.
    """

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.path = Path(spec.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self._h5: h5py.File | None = None
        with h5py.File(self.path, "r") as h5:
            self.keys = {
                "sinogram": _find_key(h5, "sinogram", spec.sinogram_key),
                "fbp": _find_key(h5, "fbp", spec.fbp_key),
                "target": _find_key(h5, "target", spec.target_key),
                "patient_id": _find_key(h5, "patient_id"),
                "slice_id": _find_key(h5, "slice_id"),
            }
            if self.keys["target"] is None:
                raise KeyError(f"No target/NDCT dataset found in {self.path}. Available keys: {list(h5.keys())}")
            self.length = int(h5[self.keys["target"]].shape[0])
        self.transform = CTTransform(spec.image_size, spec.normalize, spec.window, spec.clip, spec.augment)

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.length

    def _read(self, key: str | None, idx: int) -> np.ndarray | None:
        if key is None:
            return None
        return np.asarray(self.h5[key][idx])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        target = self._read(self.keys["target"], idx)
        fbp = self._read(self.keys["fbp"], idx)
        sino = self._read(self.keys["sinogram"], idx)
        if fbp is None:
            fbp = target.copy()
        if sino is None:
            sino = fbp.copy()
        tensors = self.transform({"sinogram": sino, "fbp": fbp, "target": target})
        item_id = str(idx)
        if self.keys["patient_id"] is not None:
            patient = self.h5[self.keys["patient_id"]][idx]
            item_id = patient.decode() if isinstance(patient, bytes) else str(patient)
            if self.keys["slice_id"] is not None:
                sid = self.h5[self.keys["slice_id"]][idx]
                item_id += f"/{sid}"
        tensors["id"] = item_id
        return tensors

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class LoDoPaBDataset(HDF5CTDataset):
    pass


class MayoDataset(HDF5CTDataset):
    pass


class PigletDataset(HDF5CTDataset):
    pass


def build_dataset(cfg: dict[str, Any], split: str) -> HDF5CTDataset:
    dc = cfg["data"]
    entry = dc.get("splits", {}).get(split, {})
    if not entry and split in dc:
        entry = dc[split]
    path = entry.get("path") or dc.get(f"{split}_path")
    if path is None:
        raise KeyError(f"No path configured for split={split!r}")
    spec = DatasetSpec(
        path=path,
        image_size=dc.get("image_size", 256),
        sinogram_key=entry.get("sinogram_key") or dc.get("sinogram_key"),
        fbp_key=entry.get("fbp_key") or dc.get("fbp_key"),
        target_key=entry.get("target_key") or dc.get("target_key"),
        normalize=dc.get("normalize", True),
        window=tuple(dc["window"]) if dc.get("window") else None,
        clip=tuple(dc["clip"]) if dc.get("clip") else None,
        augment=bool(dc.get("augment", False) and split == "train"),
    )
    name = dc.get("name", "hdf5").lower()
    cls = {"lodopab": LoDoPaBDataset, "mayo": MayoDataset, "piglet": PigletDataset}.get(name, HDF5CTDataset)
    return cls(spec)


def create_dummy_h5(path: str | Path, n: int = 16, image_size: int = 64, sino_shape: tuple[int, int] = (30, 128)) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[-1:1:complex(image_size), -1:1:complex(image_size)]
    base = np.exp(-(xx**2 + yy**2) * 4)
    target, fbp, sino = [], [], []
    for i in range(n):
        blob = np.roll(base, i % 7, axis=0) + 0.2 * np.exp(-((xx - .25)**2 + (yy + .15)**2) * 20)
        blob = normalize01(blob)
        noisy = normalize01(blob + rng.normal(0, 0.06, blob.shape))
        target.append(blob[None])
        fbp.append(noisy[None])
        sino.append(rng.random((1, *sino_shape), dtype=np.float32))
    with h5py.File(path, "w") as h5:
        h5.create_dataset("sinogram", data=np.stack(sino).astype("float32"))
        h5.create_dataset("fbp", data=np.stack(fbp).astype("float32"))
        h5.create_dataset("target", data=np.stack(target).astype("float32"))
