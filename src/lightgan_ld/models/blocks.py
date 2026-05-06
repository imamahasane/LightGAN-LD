from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def norm2d(kind: str, channels: int) -> nn.Module:
    kind = (kind or "batch").lower()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    if kind in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unknown norm: {kind}")


class GhostModule(nn.Module):
    """GhostNet-style cheap feature generation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, ratio: int = 2, stride: int = 1, norm: str = "batch", activation: bool = True):
        super().__init__()
        init_ch = math.ceil(out_ch / ratio)
        cheap_ch = out_ch - init_ch
        pad = kernel_size // 2
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride, pad, bias=False),
            norm2d(norm, init_ch),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
        )
        self.cheap_ch = cheap_ch
        cheap_out = max(init_ch, cheap_ch)
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, cheap_out, 3, 1, 1, groups=init_ch, bias=False),
            norm2d(norm, cheap_out),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
        )
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        if self.cheap_ch <= 0:
            return y[:, : self.out_ch]
        z = self.cheap(y)[:, : self.cheap_ch]
        return torch.cat([y, z], dim=1)[:, : self.out_ch]


class CondConv2d(nn.Module):
    """Conditionally parameterized 2-D convolution with batched grouped implementation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None, experts: int = 4, reduction: int = 4):
        super().__init__()
        self.in_ch, self.out_ch, self.kernel_size, self.stride = in_ch, out_ch, kernel_size, stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.experts = experts
        self.weight = nn.Parameter(torch.empty(experts, out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(experts, out_ch))
        hidden = max(16, in_ch // reduction)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, experts))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        alpha = torch.softmax(self.router(x), dim=-1)
        weight = torch.einsum("be,eocij->bocij", alpha, self.weight).reshape(b * self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)
        bias = torch.einsum("be,eo->bo", alpha, self.bias).reshape(b * self.out_ch)
        x_grouped = x.reshape(1, b * c, h, w)
        y = F.conv2d(x_grouped, weight, bias, stride=self.stride, padding=self.padding, groups=b)
        return y.reshape(b, self.out_ch, y.shape[-2], y.shape[-1])


class ECAAttention(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x).squeeze(-1).transpose(-1, -2)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class MixStyle(nn.Module):
    """MixStyle regularization for domain-generalized image restoration."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or torch.rand(()) > self.p or x.size(0) < 2:
            return x
        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(b, device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


class ResidualGhostBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_ghost: bool = True, use_condconv: bool = True, use_eca: bool = True, experts: int = 4, norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        if use_ghost:
            conv1 = GhostModule(in_ch, out_ch, norm=norm)
        else:
            conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), norm2d(norm, out_ch), nn.ReLU(inplace=True))
        conv2: nn.Module = CondConv2d(out_ch, out_ch, 3, padding=1, experts=experts) if use_condconv else nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        layers = [conv1, conv2, norm2d(norm, out_ch)]
        if use_eca:
            layers.append(ECAAttention(out_ch))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x) + self.net(x))
