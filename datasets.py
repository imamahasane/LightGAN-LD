import os
import h5py
import torchrom torch.utils.data import Dataset

class LoDoPaBSinogramDataset(Dataset):
    """
    Loads sinogram and ground-truth pairs without quantization.
    """
    def __init__(self, sino_dir: str, gt_dir: str):
        self.sino_files = sorted(
            os.path.join(sino_dir, f)
            for f in os.listdir(sino_dir)
            if f.endswith('.hdf5')
        )
        self.gt_files = sorted(
            os.path.join(gt_dir, f)
            for f in os.listdir(gt_dir)
            if f.endswith('.hdf5')
        )
        assert len(self.sino_files) and len(self.gt_files), \
            f"Empty data directory: {sino_dir} or {gt_dir}"
        assert len(self.sino_files) == len(self.gt_files), \
            "Mismatched file counts"

        self.indices = []
        for idx, (sf, gf) in enumerate(zip(self.sino_files, self.gt_files)):
            with h5py.File(sf, 'r') as fs, h5py.File(gf, 'r') as fg:
                n = min(len(fs['data']), len(fg['data']))
            self.indices += [(idx, i) for i in range(n)]
        print(f"Loaded {len(self.indices)} total slices")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        file_idx, slice_idx = self.indices[idx]
        sf = self.sino_files[file_idx]
        gf = self.gt_files[file_idx]
        with h5py.File(sf, 'r') as fs:
            sino = fs['data'][slice_idx].astype('float32')
        with h5py.File(gf, 'r') as fg:
            img  = fg['data'][slice_idx].astype('float32')

        # normalize to [0,1]
        sino = (sino - sino.min()) / (sino.max() - sino.min() + 1e-8)
        img  = (img  - img.min())  / (img.max()  - img.min()  + 1e-8)

        # directly to tensor, no uint8 round-trip
        sino_t = torch.from_numpy(sino).unsqueeze(0)
        img_t  = torch.from_numpy(img).unsqueeze(0)
        return sino_t, img_t