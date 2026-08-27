from scipy.ndimage import convolve
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py

from pathlib import Path
import yaml
config = yaml.safe_load(open(Path(__file__).resolve().parents[2] /'config.yaml', 'r'))

def convolve_lr(images: np.array, stride: int = 4, kernel: np.array = np.ones((3, 3, 3)) / 27) -> np.array:
    n_images = images.shape[0]
    n_channels = images.shape[1]
    convolved_size = images.shape[2] // stride
    convolved_images = np.empty((n_images, n_channels, convolved_size, convolved_size, convolved_size))
    for b in range(n_images):
        for c in range(n_channels):
            convolved_images[b, c] = convolve(images[b, c], kernel, mode = 'reflect', cval = 0.0)[::stride, ::stride, ::stride]
    return convolved_images

class dataset_sr(Dataset):
    def __init__(self,
                 kernel=np.ones((3, 3, 3)) / 27,
                 stride : int = 4,
                 h5_path=Path(__file__).resolve().parents[2] / 'data/jalegria/final_states.h5',
                 cache_path=Path(__file__).resolve().parents[2] / 'data/lr_states/lr_states.npz',
                 load_cache=True,
                 max_samples=None):
        
        print(f"Loading final states from HDF5 file: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            if max_samples is not None:
                self.final_states = f["states"][:max_samples]
                print(f"Loaded {self.final_states.shape[0]} final states (limited to {max_samples})")
            else:
                self.final_states = f["states"][:]
                print(f"Loaded {self.final_states.shape[0]} final states (all data)")
           
        if max_samples is not None:
            cache_dir = cache_path.parent
            cache_name = cache_path.stem + f"_max{max_samples}" + cache_path.suffix
            cache_path = cache_dir / cache_name
        
        # For the moment im transforming all the images as soon as calling the dataset  
        if os.path.exists(cache_path) and load_cache:
            print(f"\nLoading cached LR states from {cache_path}")
            self.lr_states = np.load(cache_path)["lr_states"]
        else:
            print("\nConvolving and caching LR states...")
            self.lr_states = convolve_lr(self.final_states, stride, kernel).astype(np.float32)
            np.savez_compressed(cache_path, lr_states=self.lr_states)
            
    def __getitem__(self, index):
        return (
            torch.from_numpy(self.final_states[index]).float(),
            torch.from_numpy(self.lr_states[index]).float()
        )    
    
    def __len__(self):
        return len(self.final_states)
