import torch
from src.lightgan_ld.models.blocks.ghost import GhostModule
from src.lightgan_ld.models.blocks.condconv import CondConv2d
from src.lightgan_ld.models.blocks.eca import ECA
from src.lightgan_ld.models.blocks.residual_ghost import ResidualGhostBlock

def test_ghost_shape():
    m = GhostModule(8, 16)
    x = torch.randn(2,8,32,32)
    y = m(x)
    assert y.shape == (2,16,32,32)

def test_condconv_forward():
    m = CondConv2d(8, 16, 3, padding=1, K=2)
    x = torch.randn(2,8,32,32)
    y = m(x)
    assert y.shape == (2,16,32,32)

def test_eca_scale():
    m = ECA(16)
    x = torch.randn(2,16,32,32)
    y = m(x)
    assert y.shape == x.shape

def test_residual_block():
    m = ResidualGhostBlock(8, 16, use_ghost=True, use_condconv=False, use_eca=True, experts=2)
    x = torch.randn(2,8,32,32)
    y = m(x)
    assert y.shape == (2,16,32,32)
