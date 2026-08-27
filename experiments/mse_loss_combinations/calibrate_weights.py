"""
Calibrate the "light" spectral and L1 weights against MSE on normalized data.

Measures MSE, L1 and velocity-spectral loss on a trilinear-upsampled LR
prediction vs the HR target, on a few batches of the normalized training set.
The ratios give the weights that make each additional term contribute equally
to the MSE at the start of training. Three weights are then derived for each
component:

    w_minor  = w_eq * 0.1   (term is a gentle nudge, MSE dominates)
    w_equal  = w_eq         (term balanced to MSE)
    w_major  = w_eq * 10    (term dominates, aggressive matching)

This mirrors the convention in
``experiments/mse_spectral_weighting/calibrate_weights.py`` and
``experiments/l1_spectral_weighting/calibrate_weights.py``, but combines
both calibrations into a single ``calibration.json`` so the training stage in
this experiment can pick the ``w_minor`` weight for either component.

Only the ``w_minor`` ("light") weights are used for training in this experiment
(``calibration.json -> weights.{spectral,l1}.w_minor``).

Results are saved to ``calibration.json`` and printed; the training script
reads this file automatically.

Usage
-----
    python experiments/mse_loss_combinations/calibrate_weights.py
"""

from autocvd import autocvd

autocvd(num_gpus=1)
import json
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader.dataloader_3d import dataset_sr

# ── Paths & constants ─────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states.h5")
SNAPSHOT_INDEX = 79
N_BATCHES = 10
BATCH_SIZE = 2
SPECTRAL_POOL = 2

OUTPUT_DIR = ROOT / "experiments" / "mse_loss_combinations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_JSON = OUTPUT_DIR / "calibration.json"

# Reuse the l1_spectral_weighting normalization stats so all experiments
# share identical normalization.
NORM_STATS_PATH = ROOT / "experiments" / "l1_spectral_weighting" / "normalization_stats.npz"


# ── Velocity indices (don't hardcode — AGENTS.md) ─────────────────────


@lru_cache(maxsize=1)
def _get_velocity_indices() -> tuple[int, int, int]:
    from jf1uids import SimulationConfig, get_registered_variables
    from jf1uids.option_classes.simulation_config import finalize_config

    cfg = finalize_config(SimulationConfig(dimensionality=3), (5, 128, 128, 128))
    rv = get_registered_variables(cfg)
    return rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z


# ── Spectral loss (same as train_mse_spectral.py) ─────────────────────


class SpectralLoss(nn.Module):
    """Velocity-only torch-FFT log-power spectral loss."""

    def __init__(self, vx_idx: int, vy_idx: int, vz_idx: int, pool: int = SPECTRAL_POOL):
        super().__init__()
        self.vel_idx = [vx_idx, vy_idx, vz_idx]
        self.pool = pool

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_v = pred[:, self.vel_idx]
        target_v = target[:, self.vel_idx]
        if self.pool > 1:
            pred_v = F.avg_pool3d(pred_v, kernel_size=self.pool)
            target_v = F.avg_pool3d(target_v, kernel_size=self.pool)
        pred_fft = torch.fft.rfftn(pred_v, dim=(-3, -2, -1), norm="ortho")
        target_fft = torch.fft.rfftn(target_v, dim=(-3, -2, -1), norm="ortho")
        pred_power = pred_fft.real.square() + pred_fft.imag.square()
        target_power = target_fft.real.square() + target_fft.imag.square()
        pred_power = pred_power.mean(dim=1)
        target_power = target_power.mean(dim=1)
        return F.mse_loss(
            torch.log10(pred_power + 1e-12), torch.log10(target_power + 1e-12)
        )


# ── Main ──────────────────────────────────────────────────────────────


def main():
    print("Loading normalized training dataset …")
    train_ds = dataset_sr(
        h5_path=TRAIN_H5,
        snapshot_index=SNAPSHOT_INDEX,
        use_normalizing=True,
        mean_std_path=NORM_STATS_PATH,
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"  Samples: {len(train_ds)}  |  Batches to measure: {N_BATCHES}")

    vx, vy, vz = _get_velocity_indices()
    spectral_fn = SpectralLoss(vx, vy, vz).to(DEVICE)

    mse_total = 0.0
    l1_total = 0.0
    spec_total = 0.0
    n = 0

    print("Measuring MSE, L1 and spectral loss on trilinear-upsampled LR …")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_BATCHES:
                break
            hr = batch[0].to(DEVICE)
            lr = batch[1].to(DEVICE)
            pred = F.interpolate(
                lr, scale_factor=4, mode="trilinear", align_corners=False
            )
            mse = F.mse_loss(pred, hr).item()
            l1 = F.l1_loss(pred, hr).item()
            spec = spectral_fn(pred, hr).item()
            mse_total += mse
            l1_total += l1
            spec_total += spec
            n += 1
            print(
                f"  batch {n}/{N_BATCHES}: MSE={mse:.6f}  "
                f"L1={l1:.6f}  spectral={spec:.6f}"
            )

    mse_avg = mse_total / n
    l1_avg = l1_total / n
    spec_avg = spec_total / n
    w_eq_spec = mse_avg / spec_avg if spec_avg > 0 else float("inf")
    w_eq_l1 = mse_avg / l1_avg if l1_avg > 0 else float("inf")

    weights = {
        "spectral": {
            "w_minor": w_eq_spec * 0.1,
            "w_equal": w_eq_spec,
            "w_major": w_eq_spec * 10.0,
        },
        "l1": {
            "w_minor": w_eq_l1 * 0.1,
            "w_equal": w_eq_l1,
            "w_major": w_eq_l1 * 10.0,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  CALIBRATION RESULTS ({n} batches)")
    print(f"{'=' * 60}")
    print(f"  MSE       (avg) = {mse_avg:.6f}")
    print(f"  L1        (avg) = {l1_avg:.6f}")
    print(f"  Spectral  (avg) = {spec_avg:.6f}")
    print(f"  Ratio MSE/spec  = {w_eq_spec:.4f}")
    print(f"  Ratio MSE/L1    = {w_eq_l1:.4f}")
    print(f"{'=' * 60}")
    print(f"  spectral weights:")
    for name, w in weights["spectral"].items():
        print(f"    {name:10s} = {w:.6f}")
    print(f"  l1 weights:")
    for name, w in weights["l1"].items():
        print(f"    {name:10s} = {w:.6f}")
    print(f"{'=' * 60}")

    result = {
        "mse_avg": mse_avg,
        "l1_avg": l1_avg,
        "spectral_avg": spec_avg,
        "ratio_mse_over_spectral": w_eq_spec,
        "ratio_mse_over_l1": w_eq_l1,
        "n_batches": n,
        "weights": weights,
    }
    with open(CALIBRATION_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {CALIBRATION_JSON}")


if __name__ == "__main__":
    main()