from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

from pathlib import Path
from src.utils.mean_std import compute_mean_std_dataset
from src.utils.paths import TRAIN_H5
from typing import Optional


class dataset_sr(Dataset):
    def __init__(
        self,
        h5_path=TRAIN_H5,
        snapshot_index: Optional[int] = None,
        max_samples=None,
        use_printing=False,
        use_normalizing=False,
        only_normalize_lr=False,
        means=None,
        stds=None,
        mean_std_path=None,
    ):
        if use_printing:
            print(f"Opening HDF5 file: {h5_path}")
        self.h5file = h5py.File(h5_path, "r")

        self.hr_states = self.h5file["hr_states"]
        self.lr_states = self.h5file["lr_states"]
        self.energy = self.h5file["first_snapshot_energy"]
        self.mass = self.h5file["first_snapshot_mass"]

        self.snapshots_per_sim = 80
        total_snapshots = len(self.hr_states)
        total_sims = total_snapshots // self.snapshots_per_sim
        self.use_normalizing = use_normalizing
        self.only_normalize_lr = only_normalize_lr
        self.indices = []

        if snapshot_index is not None:
            if use_printing:
                print(
                    f"Selecting snapshot index {snapshot_index} from each simulation..."
                )
            self.indices = [
                i * self.snapshots_per_sim + snapshot_index for i in range(total_sims)
            ]
        else:
            if use_printing:
                print("Using all available snapshots...")
            self.indices = list(range(total_snapshots))

        if max_samples is not None:
            self.indices = self.indices[:max_samples]
            if use_printing:
                print(f"Limiting to {max_samples} snapshots")
        if use_printing:
            print(f"Final dataset size: {len(self.indices)} samples")

        if use_normalizing:
            if means is not None and stds is not None:
                self.mean_hr, self.mean_lr = means
                self.std_hr, self.std_lr = stds
            elif mean_std_path is not None and Path(mean_std_path).exists():
                data = np.load(mean_std_path)
                self.mean_hr, self.mean_lr = data["mean_hr"], data["mean_lr"]
                self.std_hr, self.std_lr = data["std_hr"], data["std_lr"]
            else:
                self.mean_lr, self.std_lr = compute_mean_std_dataset(
                    self.lr_states, self.indices
                )
                self.mean_hr, self.std_hr = compute_mean_std_dataset(
                    self.hr_states, self.indices
                )
                if mean_std_path is not None:
                    Path(mean_std_path).parent.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        mean_std_path,
                        mean_hr=self.mean_hr,
                        mean_lr=self.mean_lr,
                        std_hr=self.std_hr,
                        std_lr=self.std_lr,
                    )

    def __getitem__(self, index):
        idx = self.indices[index]
        if self.use_normalizing:
            if self.only_normalize_lr:
                return (
                    torch.from_numpy(self.hr_states[idx]).float(),
                    torch.from_numpy(
                        (self.lr_states[idx] - self.mean_lr[:, None, None, None])
                        / self.std_lr[:, None, None, None]
                    ).float(),
                    self.energy[idx],
                    self.mass[idx],
                )
            else:
                return (
                    torch.from_numpy(
                        (self.hr_states[idx] - self.mean_hr[:, None, None, None])
                        / self.std_hr[:, None, None, None]
                    ).float(),
                    torch.from_numpy(
                        (self.lr_states[idx] - self.mean_lr[:, None, None, None])
                        / self.std_lr[:, None, None, None]
                    ).float(),
                    self.energy[idx],
                    self.mass[idx],
                )
        else:
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
