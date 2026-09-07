"""

Path resolution

Every data / output location used by the scripts is defined here and can be
overridden with environment variables, so the repository runs on any machine.
Defaults are repo-relative:

    data/full_states_h5/    train/val HDF5 splits      ($TURBULENCE_SR_DATA)
    data/full_states_h5/    unsplit source dataset     ($TURBULENCE_SR_SOURCE_DATA)
    runs/experiments/       experiment run groups      ($TURBULENCE_SR_SCRATCH)

"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, default: str) -> Path:
    value = os.environ.get(env_var)
    return Path(value) if value else ROOT / default


# Train/val HDF5 splits (full_states.h5, full_states_val.h5).
DATA_DIR = _resolve("TURBULENCE_SR_DATA", "data/full_states_h5")
TRAIN_H5 = DATA_DIR / "full_states.h5"
VAL_H5 = DATA_DIR / "full_states_val.h5"

# Unsplit 500-simulation source dataset (input of split_dataset.py).
SOURCE_DATA_DIR = _resolve("TURBULENCE_SR_SOURCE_DATA", "data/full_states_h5")

# Experiment run-group output dirs (weights.pt, losses.csv, manifest.json).
SCRATCH_ROOT = _resolve("TURBULENCE_SR_SCRATCH", "runs/experiments")
