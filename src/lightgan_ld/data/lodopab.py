from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2

class LoDoPaBPair(Dataset):
    def __init__(self, h5_path: str, resize: int = 362, augment: bool = False):
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
        y = self._resize_img(y, self.resize)
        x_fbp = self._resize_img(x_fbp, self.resize)
        if self.augment:
            if random.random() < 0.5:
                y = y[..., ::-1].copy(); x_fbp = x_fbp[..., ::-1].copy()
            if random.random() < 0.5:
                y = y[..., :, ::-1].copy(); x_fbp = x_fbp[..., :, ::-1].copy()
        s = torch.from_numpy(s).float()
        y = torch.from_numpy(y).float()
        x_fbp = torch.from_numpy(x_fbp).float()
        return s, x_fbp, y

    @staticmethod
    def _resize_img(img: np.ndarray, size: int) -> np.ndarray:
        H, W = img.shape[-2:]
        if H == size and W == size:
            return img
        out = cv2.resize(img[0], (size, size), interpolation=cv2.INTER_CUBIC)
        return out[None, ...]
