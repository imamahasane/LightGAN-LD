from pathlib import Path

import torch

from lightgan_ld.datasets.hdf5_ct import DatasetSpec, HDF5CTDataset, create_dummy_h5
from lightgan_ld.losses import CompositeGeneratorLoss
from lightgan_ld.models import build_models
from lightgan_ld.metrics import psnr, ssim


def test_dataset_and_forward(tmp_path: Path):
    path = tmp_path / "dummy.h5"
    create_dummy_h5(path, n=4, image_size=32, sino_shape=(30, 64))
    ds = HDF5CTDataset(DatasetSpec(str(path), image_size=32))
    item = ds[0]
    assert item["fbp"].shape == (1, 32, 32)
    cfg = {
        "model": {
            "generator": {"in_channels": 1, "out_channels": 1, "base_channels": 4, "num_down": 2, "use_ghost": True, "use_condconv": False, "use_eca": True, "condconv_experts": 2, "norm": "batch", "mixstyle_p": 0.0, "dropout": 0.0},
            "discriminator": {"in_channels": 1, "base_channels": 4, "num_layers": 2, "spectral_norm": False, "norm": "batch"},
            "sinogram_encoder": {"enabled": True, "penc": 0.5, "in_channels": 1, "out_channels": 1, "base_channels": 4, "norm": "batch"},
        },
        "loss": {"adv_weight": 1.0, "l1_weight": 1.0, "perceptual_weight": 0.0, "ssim_weight": 0.1, "ffl_weight": 0.1, "edge_weight": 0.1, "perceptual": {"pretrained": False}, "ffl": {"alpha": 1.0}},
    }
    G, D, E = build_models(cfg)
    x = item["fbp"].unsqueeze(0)
    y = item["target"].unsqueeze(0)
    pred = G(x)
    assert pred.shape == y.shape
    encoded = E(item["sinogram"].unsqueeze(0), out_size=32)
    assert encoded.shape == y.shape
    loss = CompositeGeneratorLoss(cfg)(pred, y, D(pred)).total
    assert torch.isfinite(loss)
    assert torch.isfinite(psnr(pred, y)).all()
    assert torch.isfinite(ssim(pred, y)).all()
