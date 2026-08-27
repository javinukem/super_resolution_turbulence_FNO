from scipy.ndimage import convolve
from torch.utils.data import Dataset
import glob
import numpy as np
import torch
import os
import h5py

from pathlib import Path


def convolve_lr_2d(images: np.array, stride: int = 4, kernel: np.array = np.ones((3, 3)) / 9) -> np.array:
    """
    Applies 2D convolution to each (H, W) slice of the input array.
    Expects input shape: (N, C, H, W)
    Returns downsampled LR images with same shape but lower resolution.
    """
    n_images = images.shape[0]
    n_channels = images.shape[1]
    convolved_size = images.shape[2] // stride
    convolved_images = np.empty((n_images, n_channels, convolved_size, convolved_size))
    for b in range(n_images):
        for c in range(n_channels):
            convolved_images[b, c] = convolve(images[b, c], kernel, mode='reflect')[::stride, ::stride]
    return convolved_images

class dataset_sr_2d(Dataset):
    def __init__(self,
                 kernel=np.ones((3, 3)) / 9,
                 stride: int = 4,
                 h5_path=Path(__file__).resolve().parents[2] / 'data/final_states_h5/final_states.h5',
                 cache_path=Path(__file__).resolve().parents[2] / 'data/lr_states/lr_states_2d.npz',
                 load_cache=True,
                 max_samples=None):
        
        print(f"Loading final states from HDF5 file: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            if max_samples is not None:
                final_states = f["states"][:max_samples]
                print(f"Loaded {final_states.shape[0]} final states (limited to {max_samples})")
            else:
                final_states = f["states"][:]
                print(f"Loaded {final_states.shape[0]} final states (all data)")

        # From (N, 5, D, H, W) → treat each slice along D as a 2D sample
        # Output shape: (N * D, 5, H, W)
        final_states = final_states.transpose(0, 2, 1, 3, 4)
        self.final_states = final_states.reshape(-1, 5, 128, 128)

        if max_samples is not None:
            cache_dir = cache_path.parent
            cache_name = cache_path.stem + f"_max{max_samples}" + cache_path.suffix
            cache_path = cache_dir / cache_name

        if os.path.exists(cache_path) and load_cache:
            print(f"\nLoading cached LR states from {cache_path}")
            self.lr_states = np.load(cache_path)["lr_states"]
        else:
            print("\nConvolving and caching LR states...")
            self.lr_states = convolve_lr_2d(self.final_states, stride, kernel).astype(np.float32)
            np.savez_compressed(cache_path, lr_states=self.lr_states)

        # Final check
        assert self.final_states.shape[0] == self.lr_states.shape[0], \
            f"Mismatch: {self.final_states.shape[0]} HR vs {self.lr_states.shape[0]} LR"

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.final_states[index]).float(),
            torch.from_numpy(self.lr_states[index]).float()
        )    
    
    def __len__(self):
        return len(self.final_states)
