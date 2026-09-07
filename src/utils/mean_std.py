import numpy as np
import torch
from pathlib import Path


def compute_mean_std_dataset(dataset, indices):
    """
    Takes a dataset and computes the mean/std for the chosen samples.
    dataset: HDF5 dataset (e.g. hr_states)
    indices: list of indices to include
    """
    x = dataset[indices[0]]
    C = x.shape[0]

    mean = np.zeros(C, dtype=np.float64)
    var = np.zeros(C, dtype=np.float64)
    n_total = 0

    for idx in indices:
        x = dataset[idx]
        x = x.reshape(C, -1)
        n_pixels = x.shape[1]

        mean += x.mean(axis=1) * n_pixels
        var += x.var(axis=1) * n_pixels
        n_total += n_pixels

    mean /= n_total
    var /= n_total
    std = np.sqrt(var)

    return mean, std


def load_norm_stats(manifest: dict, device: torch.device) -> dict | None:
    """
    Load cached per-channel mean/std for norm-on experiments.
    """
    stats_path = manifest.get("normalization_stats")
    if stats_path is None:
        return None
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    data = np.load(stats_path)
    return {
        "mean_lr": torch.tensor(data["mean_lr"], device=device, dtype=torch.float32),
        "std_lr": torch.tensor(data["std_lr"], device=device, dtype=torch.float32),
        "mean_hr": torch.tensor(data["mean_hr"], device=device, dtype=torch.float32),
        "std_hr": torch.tensor(data["std_hr"], device=device, dtype=torch.float32),
    }
