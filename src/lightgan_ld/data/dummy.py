import torch
from torch.utils.data import Dataset

class DummyPairs(Dataset):
    def __init__(self, n: int = 64, img_size: int = 256, sino_shape=(1,512,720)):
        self.n = n; self.img_size = img_size; self.sino_shape = sino_shape
    def __len__(self): return self.n
    def __getitem__(self, idx):
        H = W = self.img_size
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
        phantom = ((xx**2 + yy**2) < 0.5**2).float().unsqueeze(0)
        noise = 0.05 * torch.randn_like(phantom)
        y = (phantom + noise).clamp(0,1)
        x_fbp = (phantom + 0.15*torch.randn_like(phantom)).clamp(0,1)
        s = torch.randn(*self.sino_shape)
        return s, x_fbp, y
