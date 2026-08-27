"""
Split the full_states.h5 dataset into training and validation files
at the simulation boundary (80/20 split).

Layout: 500 simulations × 80 snapshots = 40 000 samples.
  - Training:   sims 0–399  → 32 000 snapshots → full_states.h5
  - Validation:  sims 400–499 → 8 000 snapshots  → full_states_val.h5

The original file is renamed to full_states_original.h5, and the training
subset is written as the new full_states.h5 so that existing code keeps
working without changes.

Usage
-----
    python data/split_dataset.py
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path

SRC_DIR = Path("/export/data/jalegria/full_states_h5")
OUT_DIR = Path("/export/scratch/jalegria/full_states_h5")
ORIGINAL = SRC_DIR / "full_states.h5"
TRAIN_OUT = OUT_DIR / "full_states.h5"
VAL_OUT = OUT_DIR / "full_states_val.h5"

TOTAL_SIMS = 500
SNAPSHOTS_PER_SIM = 80
TRAIN_SIMS = 400  # 80%

CHUNK_SIZE = 80  # copy one simulation at a time


def copy_slice(src_ds, dst_ds, start: int, end: int, chunk: int = CHUNK_SIZE):
    """Copy rows [start, end) from src_ds into dst_ds in chunks."""
    written = 0
    for i in range(start, end, chunk):
        j = min(i + chunk, end)
        dst_ds[written : written + (j - i)] = src_ds[i:j]
        written += j - i
        print(f"    {written}/{end - start}", end="\r")
    print()


def main():
    if not ORIGINAL.exists():
        print(f"Error: {ORIGINAL} not found.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if VAL_OUT.exists() and TRAIN_OUT.exists():
        print(f"Output files already exist in {OUT_DIR}")
        print("Delete them manually if you want to re-split.")
        sys.exit(1)

    train_end = TRAIN_SIMS * SNAPSHOTS_PER_SIM        # 32000
    total = TOTAL_SIMS * SNAPSHOTS_PER_SIM             # 40000
    val_count = total - train_end                       # 8000

    # ── verify source ────────────────────────────────────────────────
    with h5py.File(ORIGINAL, "r") as f:
        n = f["hr_states"].shape[0]
        assert n == total, f"Expected {total} samples, got {n}"
        print(f"Source: {ORIGINAL}")
        print(f"  {n} samples ({TOTAL_SIMS} sims × {SNAPSHOTS_PER_SIM} snapshots)")
        print(f"  Training:   sims 0–{TRAIN_SIMS - 1} → {train_end} samples")
        print(f"  Validation: sims {TRAIN_SIMS}–{TOTAL_SIMS - 1} → {val_count} samples")

    # ── step 1: write training file ──────────────────────────────────
    print(f"\n[1/2] Writing training file: {TRAIN_OUT}")
    with h5py.File(ORIGINAL, "r") as src, h5py.File(TRAIN_OUT, "w") as dst:
        for key in src.keys():
            ds = src[key]
            shape = list(ds.shape)
            shape[0] = train_end
            dst_ds = dst.create_dataset(key, shape=shape, dtype=ds.dtype)
            print(f"  {key} {ds.shape} → {tuple(shape)}")
            copy_slice(ds, dst_ds, 0, train_end)

    # ── step 2: write validation file ────────────────────────────────
    print(f"\n[2/2] Writing validation file: {VAL_OUT}")
    with h5py.File(ORIGINAL, "r") as src, h5py.File(VAL_OUT, "w") as dst:
        for key in src.keys():
            ds = src[key]
            shape = list(ds.shape)
            shape[0] = val_count
            dst_ds = dst.create_dataset(key, shape=shape, dtype=ds.dtype)
            print(f"  {key} {ds.shape} → {tuple(shape)}")
            copy_slice(ds, dst_ds, train_end, total)

    print("\n✓ Done!")
    print(f"  Training:   {TRAIN_OUT} ({train_end} samples)")
    print(f"  Validation: {VAL_OUT} ({val_count} samples)")


if __name__ == "__main__":
    main()
