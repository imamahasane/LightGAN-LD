import torch
from src.lightgan_ld.losses.adversarial import g_hinge_loss, d_hinge_loss
from src.lightgan_ld.losses.ssim import ssim_loss
from src.lightgan_ld.losses.focal_frequency import focal_frequency_loss
from src.lightgan_ld.losses.edge_aware import edge_aware_l1

def test_adv_losses():
    real = torch.randn(4,1,8,8)
    fake = torch.randn(4,1,8,8)
    dl = d_hinge_loss(real, fake)
    gl = g_hinge_loss(fake)
    assert dl.dim()==0 and gl.dim()==0

def test_ssim_ffl_edge():
    a = torch.rand(4,1,32,32)
    b = torch.rand(4,1,32,32)
    assert ssim_loss(a,b).item() >= 0.0
    assert focal_frequency_loss(a,b) >= 0.0
    assert edge_aware_l1(a,b) >= 0.0
