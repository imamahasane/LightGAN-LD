import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    def __init__(self, inp: int, oup: int, ratio: int = 2,
                 primary_kernel: int = 1, dw_kernels=(3,5)):
        super().__init__()
        init_c = math.ceil(oup / ratio)
        cheap_c = init_c * (ratio - 1)
        self.primary = nn.Conv2d(inp, init_c, primary_kernel,
                                  padding=primary_kernel//2, bias=False)
        self.dw_convs = nn.ModuleList(
            nn.Conv2d(init_c, init_c, k,
                      padding=k//2, groups=init_c, bias=False)
            for k in dw_kernels
        )
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        x1 = self.primary(x)
        x2 = torch.cat([dw(x1) for dw in self.dw_convs], dim=1)[:, :-(x1.size(1))]
        out = torch.cat([x1, x2], dim=1)
        return self.bn(out)

class CondConv(nn.Module):
    def __init__(self, inp: int, oup: int, exp: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(
            nn.Conv2d(inp, oup, 3, padding=1, bias=False)
            for _ in range(exp)
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, exp, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.attn(x)
        return sum(w[:, i:i+1] * conv(x)
                   for i, conv in enumerate(self.experts))

class ECABlock(nn.Module):
    def __init__(self, c: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(c) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1d = nn.Conv1d(1, 1, k, padding=k//2, bias=False)

    def forward(self, x):
        y = x.mean(dim=(2,3), keepdim=True)
        y = self.conv1d(y.squeeze(-1).transpose(-1,-2))
        w = y.sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w

class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B, C, H, W = x.size()
        mu = x.mean([2,3], keepdim=True)
        sig = x.var([2,3], keepdim=True).sqrt().add(1e-6)
        xn = (x - mu) / sig
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        lam = self.beta.sample((B,1,1,1)).to(x.device)
        mu_m  = mu * lam + mu2 * (1 - lam)
        sig_m = sig * lam + sig2 * (1 - lam)
        return xn * sig_m + mu_m

class SinogramEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,padding=1,stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=1,stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(64,32,3,padding=1),     nn.ReLU(inplace=True),
            nn.Conv2d(32,1,3,padding=1),
            # Upsample back to target image size
            nn.Upsample(size=Config.IMG_SHAPE, mode='bilinear', align_corners=False)
        )
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms     = MixStyle()
        self.init   = GhostModule(1,16)
        # self.d1,a1  = CondConv(16,32,exp=2), ECABlock(32)
        # self.d2,a2  = CondConv(32,64,exp=3), ECABlock(64)
        self.d1, self.a1 = CondConv(16,32,exp=2), ECABlock(32)
        self.d2, self.a2 = CondConv(32,64,exp=3), ECABlock(64)
        self.pool   = nn.AvgPool2d(2)
        self.res    = nn.Sequential(*[GhostModule(64,64) for _ in range(2)])
        self.bridge = ECABlock(64)
        # <<-- FIXED: use keyword args for Upsample
        self.u1     = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            GhostModule(64,64)
        )
        self.u2     = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            GhostModule(64,32)
        )
        self.final  = nn.Conv2d(32,1,3,padding=1)
        self.output_resize = nn.Upsample(size=Config.IMG_SHAPE, mode='bilinear', align_corners=False)
        self.act    = nn.ReLU(inplace=True)
    def _crop_to_match(self, tensor1, tensor2):
        """Crop tensor1 to match the spatial dimensions of tensor2."""
        _, _, h1, w1 = tensor1.size()
        _, _, h2, w2 = tensor2.size()
        h = min(h1, h2)
        w = min(w1, w2)
        return tensor1[:, :, :h, :w]

    def forward(self, x):
        x   = self.ms(x)
        e0  = self.act(self.init(x))
        e1  = self.act(self.a1(self.d1(e0))); e1p = self.pool(e1)
        e2  = self.act(self.a2(self.d2(e1p))); e2p = self.pool(e2)
        r   = self.res(e2p); b = self.bridge(r)
        u1_out = self.u1(b)
        e2_cropped = self._crop_to_match(e2, u1_out)
        d1  = self.act(u1_out + e2_cropped)
        u2_out = self.u2(d1)
        e1_cropped = self._crop_to_match(e1, u2_out)
        d2  = self.act(u2_out + e1_cropped)
        out = torch.tanh(self.final(d2))
        return self.output_resize(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            GhostModule(1,8),  nn.AvgPool2d(2),
            GhostModule(8,16), nn.AvgPool2d(2),
            GhostModule(16,32), nn.Conv2d(32,1,3,padding=1)
        )
    def forward(self,x): return self.net(x)

# LOSSES
def edge_loss(pred, tgt):
    k = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],
                     device=pred.device,dtype=pred.dtype).view(1,1,3,3)
    return F.l1_loss(F.conv2d(pred,k,padding=1),
                     F.conv2d(tgt,k,padding=1))

def focal_frequency_loss(pred,tgt,chi=1.0):
    dev = pred.device
    if dev.type=="mps":  # CPU fallback
        p,t = pred.float().cpu(), tgt.float().cpu()
        Yp = torch.fft.rfft2(p,norm='ortho'); Yt = torch.fft.rfft2(t,norm='ortho')
        wf = torch.log(torch.abs(Yt)**2+1e-8)**chi
        return F.l1_loss(Yp*wf,Yt*wf).to(dev)
    Yp = torch.fft.rfft2(pred,norm='ortho'); Yt = torch.fft.rfft2(tgt,norm='ortho')
    wf = torch.log(torch.abs(Yt)**2+1e-8)**chi
    return F.l1_loss(Yp*wf,Yt*wf)