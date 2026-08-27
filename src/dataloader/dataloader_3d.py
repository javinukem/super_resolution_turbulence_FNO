from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

from pathlib import Path
from utils.mean_std import compute_mean_std_dataset
from typing import Optional


class dataset_sr(Dataset):
    def __init__(
        self,
        h5_path=Path(__file__).resolve().parents[6]
        / "data/jalegria/full_states_h5/full_states.h5",
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


"""
class lazy_dataset_sr(Dataset):
    def __init__(self,
                kernel=np.ones((3, 3, 3)) / 27,
                stride: int = 4,
                h5_path=Path(__file__).resolve().parents[2] / 'data/final_states_h5/final_states.h5',
                load_cache=True,
                max_samples=None):
        
        self.h5_path = h5_path
        self.kernel = kernel
        self.stride = stride
        self.max_samples = max_samples
        
        # Just get the dataset size, don't load data
        with h5py.File(h5_path, "r") as f:
            self.dataset_size = len(f["states"]) if max_samples is None else min(max_samples, len(f["states"]))
        
        print(f"Dataset initialized with {self.dataset_size} samples")
        
        # Keep HDF5 file reference for lazy loading
        self.h5_file = None
        
    def _ensure_h5_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
    
    def __getitem__(self, index):
        self._ensure_h5_open()
        
        # Load single sample on-demand
        final_state = self.h5_file["states"][index]
        
        # Compute LR state on-the-fly
        lr_state = convolve_lr(final_state[np.newaxis], self.stride, self.kernel)[0].astype(np.float32)
        
        return (
            torch.from_numpy(final_state).float(),
            torch.from_numpy(lr_state).float()
        )
    
    def __len__(self):
        return self.dataset_size
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
"""
