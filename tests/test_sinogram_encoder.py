import torch
from src.lightgan_ld.models.sinogram_encoder import SinogramEncoder

def test_sino_encoder_outsize():
    E = SinogramEncoder(1,1,channels=8)
    s = torch.randn(2,1,128,128)
    y = E(s, out_size=64)
    assert y.shape == (2,1,64,64)
