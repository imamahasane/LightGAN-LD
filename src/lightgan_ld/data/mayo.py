import h5py
import torch
from torch.utils.data import Dataset
import random

class MayoPair(Dataset):
    def __init__(self, h5_path: str, resize: int = 256, augment: bool = False):
        self.h5 = h5py.File(h5_path, 'r')
        self.sino = self.h5['sino']
        self.ndct = self.h5['ndct']
        self.fbp  = self.h5['fbp'] if 'fbp' in self.h5 else None
        self.resize = resize
        self.augment = augment

    def __len__(self):
        return len(self.ndct)

    def __getitem__(self, idx: int):
        s = self.sino[idx]
        y = self.ndct[idx]
        if self.fbp is not None:
            x_fbp = self.fbp[idx]
        else:
            x_fbp = y
        if self.augment:
            if random.random() < 0.5:
                y = y[..., ::-1].copy(); x_fbp = x_fbp[..., ::-1].copy()
            if random.random() < 0.5:
                y = y[..., :, ::-1].copy(); x_fbp = x_fbp[..., :, ::-1].copy()
        return torch.from_numpy(s).float(), torch.from_numpy(x_fbp).float(), torch.from_numpy(y).float()
