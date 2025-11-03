from pathlib import Path
import h5py, numpy as np
from PIL import Image

def save_grid(imgs, path):
    Image.fromarray(imgs).save(path)

if __name__ == "__main__":
    p = Path("data/lodopab_val.h5")
    if not p.exists():
        print("[WARN] HDF5 not found; run prepare script.")
    else:
        with h5py.File(p, "r") as f:
            y = f["ndct"][:8,0]
            y = (y * 255).clip(0,255).astype("uint8")
            Image.fromarray(y[0]).save("sample_ndct.png")
            print("Saved sample_ndct.png")
