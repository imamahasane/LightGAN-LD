import torch
import torch.nn as nn

class CondConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int=1, padding: int=0, K: int=4):
        super().__init__()
        self.K = K
        self.experts = nn.Parameter(torch.randn(K, out_ch, in_ch, kernel_size, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(K, out_ch))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Linear(in_ch, max(16, in_ch//4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, in_ch//4), K),
            nn.Softmax(dim=-1)
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.experts
        b = self.bias
        g = self.pool(x).flatten(1)
        alpha = self.router(g)
        weff = torch.einsum('bk,koihw->boihw', alpha, w)
        beff = torch.einsum('bk,ko->bo', alpha, b)
        outs = []
        for i in range(B):
            outs.append(
                nn.functional.conv2d(
                    x[i:i+1], weff[i], beff[i], stride=self.stride, padding=self.padding
                )
            )
        return torch.cat(outs, dim=0)
