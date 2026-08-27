from scipy.ndimage import convolve
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py

from pathlib import Path


class dataset_sr(Dataset):
    def __init__(
        self,
        h5_path=Path(__file__).resolve().parents[6]
        / "data/jalegria/turbulence/2d_states.h5",
        snapshot_index: int = None,
        max_samples=None,
    ):
        print(f"Opening HDF5 file: {h5_path}")
        self.h5file = h5py.File(h5_path, "r")

        self.hr_states = self.h5file["hr_states"]
        self.lr_states = self.h5file["lr_states"]
        self.energy = self.h5file["first_snapshot_energy"]
        self.mass = self.h5file["first_snapshot_mass"]

        self.snapshots_per_sim = 100
        total_snapshots = len(self.hr_states)
        total_sims = total_snapshots // self.snapshots_per_sim

        self.indices = []

        if snapshot_index is not None:
            print(f"Selecting snapshot index {snapshot_index} from each simulation...")
            self.indices = [
                i * self.snapshots_per_sim + snapshot_index for i in range(total_sims)
            ]
        else:
            print("Using all available snapshots...")
            self.indices = list(range(total_snapshots))

        if max_samples is not None:
            self.indices = self.indices[:max_samples]
            print(f"Limiting to {max_samples} snapshots")

        print(f"Final dataset size: {len(self.indices)} samples")

    def __getitem__(self, index):
        idx = self.indices[index]
        return (
            torch.from_numpy(self.hr_states[idx]).float(),
            torch.from_numpy(self.lr_states[idx]).float(),
            self.energy[idx],
            self.mass[idx],
        )

    def __len__(self):
        return len(self.indices)

    def __del__(self):
        if hasattr(self, "h5file"):
            self.h5file.close()
