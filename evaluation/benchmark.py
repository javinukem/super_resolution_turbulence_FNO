"""
Comprehensive validation benchmark for turbulence super-resolution models.

Evaluates the models listed in a manifest (one entry per trained run plus a
trilinear baseline) on multiple metrics:
  - MSE (Mean Squared Error)
  - Per-field losses (pressure, density, vx, vy, vz, |v|, vorticity)
  - Spectral MSE (MSE on the 1-D power spectrum)
  - Perceptual loss (feature-space L2 via a lightweight 3-D encoder)
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index Measure)

Variable-scale models (CFNO and trilinear interpolation baseline) are
benchmarked at every scale listed in their manifest entry's ``eval_scales``
(e.g. x4 and x2).  x2 targets are obtained by trilinear downsampling of the
x4 HR target state.

Usage
-----
    python evaluation/benchmark.py
    python evaluation/benchmark.py --manifest /path/to/manifest.json

The script auto-selects a free GPU via ``autocvd`` when available.
Results are printed to stdout and saved to the ``benchmark_csv`` path declared
in the manifest (default: ``experiments/training_best_models_experiment/
trained_on_last_snapshot/benchmark_results.csv``).
"""

from autocvd import autocvd

autocvd(num_gpus=1)
import argparse
import gc
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataloader.dataloader_3d import dataset_sr
from evaluation.manifest import (
    build_model,
    discover_latest,
    load_manifest,
    load_norm_stats,
)

UPSAMPLE_FACTOR = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
NUM_WORKERS = 2
VAL_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
SNAPSHOT_INDEX = 79  # last snapshot per simulation only

SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
DEFAULT_MANIFEST_GLOB = "best_models_*"


# =====================================================================
# Metrics
# =====================================================================


@lru_cache(maxsize=1)
def _get_registered_variables_3d():
    """Return jf1uids registered variable indices for 3-D primitive states."""
    from jf1uids import SimulationConfig, get_registered_variables
    from jf1uids.option_classes.simulation_config import finalize_config

    cfg = finalize_config(SimulationConfig(dimensionality=3), (5, 128, 128, 128))
    return get_registered_variables(cfg)


class MSEMetric:
    """Standard pixel-wise MSE."""

    name = "MSE"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return F.mse_loss(pred, target).item()


class PSNRMetric:
    """Peak Signal-to-Noise Ratio (channel-averaged)."""

    name = "PSNR"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target).item()
        if mse == 0:
            return float("inf")
        data_range = target.max().item() - target.min().item()
        if data_range == 0:
            return float("inf")
        return 10.0 * np.log10(data_range**2 / mse)


class ChannelMSEMetric:
    """MSE on a single primitive channel."""

    def __init__(self, name: str, channel_idx: int):
        self.name = name
        self.channel_idx = channel_idx

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        ch = self.channel_idx
        return F.mse_loss(pred[:, ch : ch + 1], target[:, ch : ch + 1]).item()


class VelocityNormMSEMetric:
    """MSE on velocity magnitude |v|."""

    name = "Loss_v_norm"

    def __init__(self, vx_idx: int, vy_idx: int, vz_idx: int):
        self._velocity_indices = [vx_idx, vy_idx, vz_idx]

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_v = pred[:, self._velocity_indices]
        target_v = target[:, self._velocity_indices]
        pred_norm = torch.linalg.vector_norm(pred_v, dim=1, keepdim=True)
        target_norm = torch.linalg.vector_norm(target_v, dim=1, keepdim=True)
        return F.mse_loss(pred_norm, target_norm).item()


class VorticityMSEMetric:
    """MSE on vorticity magnitude |curl(v)|."""

    name = "Loss_vorticity"

    def __init__(self, vx_idx: int, vy_idx: int, vz_idx: int):
        self._vx_idx = vx_idx
        self._vy_idx = vy_idx
        self._vz_idx = vz_idx

    @staticmethod
    def _vorticity_magnitude(vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor):
        dvx_dz = torch.gradient(vx, dim=-3)[0]
        dvx_dy = torch.gradient(vx, dim=-2)[0]
        dvy_dz = torch.gradient(vy, dim=-3)[0]
        dvy_dx = torch.gradient(vy, dim=-1)[0]
        dvz_dy = torch.gradient(vz, dim=-2)[0]
        dvz_dx = torch.gradient(vz, dim=-1)[0]

        omega_x = dvz_dy - dvy_dz
        omega_y = dvx_dz - dvz_dx
        omega_z = dvy_dx - dvx_dy
        return torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-12)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        vx_pred = pred[:, self._vx_idx]
        vy_pred = pred[:, self._vy_idx]
        vz_pred = pred[:, self._vz_idx]
        vx_target = target[:, self._vx_idx]
        vy_target = target[:, self._vy_idx]
        vz_target = target[:, self._vz_idx]

        vort_pred = self._vorticity_magnitude(vx_pred, vy_pred, vz_pred)
        vort_target = self._vorticity_magnitude(vx_target, vy_target, vz_target)
        return F.mse_loss(vort_pred, vort_target).item()


class SSIMMetric:
    """
    Structural Similarity computed per-channel on 3-D volumes and averaged.
    Uses a sliding-window approach with a 3-D Gaussian kernel.
    """

    name = "SSIM"

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self._kernel_cache: dict[str, torch.Tensor] = {}

    def _gaussian_kernel_3d(self, channels: int, device: torch.device) -> torch.Tensor:
        key = f"{channels}_{device}"
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        sigma = 1.5
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device)
        coords -= self.window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = g / g.sum()
        kernel_3d = (
            kernel_1d[:, None, None]
            * kernel_1d[None, :, None]
            * kernel_1d[None, None, :]
        )
        kernel_3d = kernel_3d.expand(channels, 1, -1, -1, -1).contiguous()
        self._kernel_cache[key] = kernel_3d
        return kernel_3d

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        C = pred.shape[1]
        kernel = self._gaussian_kernel_3d(C, pred.device)
        pad = self.window_size // 2
        C1 = (0.01 * (target.max() - target.min())).item() ** 2
        C2 = (0.03 * (target.max() - target.min())).item() ** 2

        mu_pred = F.conv3d(pred, kernel, padding=pad, groups=C)
        mu_target = F.conv3d(target, kernel, padding=pad, groups=C)

        mu_pred_sq = mu_pred**2
        mu_target_sq = mu_target**2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv3d(pred**2, kernel, padding=pad, groups=C) - mu_pred_sq
        sigma_target_sq = (
            F.conv3d(target**2, kernel, padding=pad, groups=C) - mu_target_sq
        )
        sigma_cross = F.conv3d(pred * target, kernel, padding=pad, groups=C) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )
        return ssim_map.mean().item()


class SpectralMSEMetric:
    """
    MSE between the 1-D energy power spectra of prediction and target.

    Uses the same physics pipeline as ``figures/spectra.py``: the primitive
    state (density, velocity, pressure) is converted to total energy via
    jf1uids, and the shell-averaged power spectrum P(k) is computed with
    ``Pk_library.Pk``.  The metric is the MSE between the log₁₀ P(k) curves
    (log-space so that all wavenumber decades contribute equally).
    """

    name = "Spectral_MSE"

    def __init__(self):
        import jax.numpy as jnp
        from fractions import Fraction
        from jf1uids import SimulationConfig, get_helper_data, get_registered_variables
        from jf1uids.option_classes.simulation_config import finalize_config

        self._jnp = jnp
        self._gamma = float(Fraction("5/3"))

        hr_shape = (5, 128, 128, 128)
        cfg = SimulationConfig(dimensionality=3)
        cfg = finalize_config(cfg, hr_shape)
        self._config = cfg
        self._helper_data = get_helper_data(cfg)
        self._registered_variables = get_registered_variables(cfg)

    def _energy_spectrum(self, state_np: np.ndarray) -> np.ndarray:
        """Return the 1-D power spectrum for a single (C, N, N, N) state."""
        import Pk_library as PKL
        from jf1uids.fluid_equations.fluid import (
            get_absolute_velocity,
            total_energy_from_primitives,
        )

        rv = self._registered_variables
        prim = self._jnp.array(state_np)

        rho = prim[rv.density_index]
        u = get_absolute_velocity(prim, self._config, rv)
        p = prim[rv.pressure_index]
        energy = np.array(
            total_energy_from_primitives(rho, u, p, self._gamma), dtype=np.float32
        )

        pk = PKL.Pk(
            delta=energy, BoxSize=1, axis=0, MAS="None", threads=4, verbose=False
        )
        return pk.Pk1D

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        B = pred.shape[0]
        total = 0.0
        for b in range(B):
            pk_pred = self._energy_spectrum(pred[b].detach().cpu().float().numpy())
            pk_target = self._energy_spectrum(target[b].detach().cpu().float().numpy())
            log_pred = np.log10(np.clip(pk_pred, 1e-30, None))
            log_target = np.log10(np.clip(pk_target, 1e-30, None))
            total += float(np.mean((log_pred - log_target) ** 2))
        return total / B


class EnergyMassConservationMetric:
    """MSE of conserved totals (total energy + total mass) between SR and HR.

    For each state the *total energy* field is obtained from the primitive
    state via ``jf1uids.fluid_equations.fluid.total_energy_from_primitives``
    (per-cell energy, summed over the volume → scalar), and the *total mass*
    is the volume sum of the density channel.  The metric is the
    batch-averaged sum of squared deviations:

        metric = mean_b [ (E_pred - E_target)² + (M_pred - M_target)² ]

    Lower is better — a value near zero means the SR state conserves the
    total energy and mass of the HR reference.  Energy and mass have
    different scales, so this is a conservation *diagnostic*, not a
    calibrated loss; it complements the per-field / spectral metrics.
    """

    name = "energy_mass_conservation"

    def __init__(self):
        import jax.numpy as jnp
        from fractions import Fraction
        from jf1uids import SimulationConfig, get_registered_variables
        from jf1uids.option_classes.simulation_config import finalize_config

        self._jnp = jnp
        self._gamma = float(Fraction("5/3"))
        cfg = finalize_config(SimulationConfig(dimensionality=3), (5, 128, 128, 128))
        self._config = cfg
        self._registered_variables = get_registered_variables(cfg)

    def _totals(self, state_np: np.ndarray) -> tuple[float, float]:
        from jf1uids.fluid_equations.fluid import (
            get_absolute_velocity,
            total_energy_from_primitives,
        )

        rv = self._registered_variables
        prim = self._jnp.array(state_np)
        rho = prim[rv.density_index]
        u = get_absolute_velocity(prim, self._config, rv)
        p = prim[rv.pressure_index]
        energy_total = float(
            np.array(total_energy_from_primitives(rho, u, p, self._gamma)).sum()
        )
        mass_total = float(np.array(rho).sum())
        return energy_total, mass_total

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        B = pred.shape[0]
        total = 0.0
        for b in range(B):
            e_pred, m_pred = self._totals(
                pred[b].detach().cpu().float().numpy()
            )
            e_target, m_target = self._totals(
                target[b].detach().cpu().float().numpy()
            )
            total += (e_pred - e_target) ** 2 + (m_pred - m_target) ** 2
        return total / B


class PerceptualLoss3D:
    """
    Lightweight perceptual loss for 3-D fields.

    Extracts features at multiple depths of a small 3-D ConvNet (random but
    fixed weights) and computes L2 distance in feature space. This measures
    structural/textural similarity without requiring a pretrained 3-D network.
    """

    name = "Perceptual"

    def __init__(self, device: torch.device):
        self.device = device
        self.encoder = self._build_encoder().to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _build_encoder() -> nn.Module:
        # Fix the random seed so results are reproducible across runs
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        encoder = nn.Sequential(
            nn.Conv3d(5, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        torch.random.set_rng_state(rng_state)
        return encoder

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                feats.append(x)
        return feats

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        feats_pred = self._extract(pred)
        feats_target = self._extract(target)
        loss = sum(
            F.mse_loss(fp, ft).item() for fp, ft in zip(feats_pred, feats_target)
        )
        return loss / len(feats_pred)


# =====================================================================
# Target helper
# =====================================================================


def _target_for_scale(hr: torch.Tensor, upsample_factor: int) -> torch.Tensor:
    """
    Return the target tensor matching a requested SR factor.

    HR data in the dataset corresponds to ``UPSAMPLE_FACTOR`` relative to LR.
    For lower factors (e.g. x2), the target is produced by trilinear
    downsampling of HR.
    """
    if upsample_factor == UPSAMPLE_FACTOR:
        return hr
    if UPSAMPLE_FACTOR % upsample_factor != 0:
        raise ValueError(
            f"Unsupported upsample_factor={upsample_factor}; "
            f"must divide UPSAMPLE_FACTOR={UPSAMPLE_FACTOR}."
        )
    target_size = tuple(
        dim * upsample_factor // UPSAMPLE_FACTOR for dim in hr.shape[-3:]
    )
    return F.interpolate(hr, size=target_size, mode="trilinear", align_corners=False)


# =====================================================================
# Evaluation loop
# =====================================================================


def evaluate_model(
    run,
    val_loader: DataLoader,
    metrics: list,
    label: str,
    upsample_factor: int,
) -> dict:
    """Run a model through the validation set and compute all metrics.

    ``run`` is the uniform callable returned by
    :func:`evaluation.manifest.build_model` — ``run(lr, upsample_factor)``.
    """
    accum = {m.name: 0.0 for m in metrics}
    n_batches = 0

    print(f"  Evaluating {label} (x{upsample_factor}, {len(val_loader)} batches) …")
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            hr = data[0].to(DEVICE)
            lr = data[1].to(DEVICE)
            target = _target_for_scale(hr, upsample_factor)

            pred = run(lr, upsample_factor)

            for m in metrics:
                accum[m.name] += m(pred, target)
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"    batch {batch_idx + 1}/{len(val_loader)}", end="\r")

            del hr, lr, target, pred

    elapsed = time.time() - t0
    result = {m.name: accum[m.name] / n_batches for m in metrics}
    result["time_s"] = elapsed
    result["upsample_factor"] = upsample_factor
    result["model"] = label
    print(f"  ✓ {label} x{upsample_factor} done in {elapsed:.1f}s")
    return result


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Path to the experiment folder (containing manifest.json). "
            "If omitted, auto-discovers the newest "
            f"{{DEFAULT_MANIFEST_GLOB}} folder under {SCRATCH_BASE}."
        ),
    )
    args = parser.parse_args()

    manifest_dir = (
        Path("/export/scratch/jalegria/experiments/", args.manifest)
        if args.manifest
        else discover_latest(SCRATCH_BASE, DEFAULT_MANIFEST_GLOB)
    )
    manifest = load_manifest(manifest_dir)
    results_csv = manifest["benchmark_csv"]
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Manifest : {manifest['_manifest_path']}")
    print(f"Experiment: {manifest.get('experiment')}")
    print(f"Results  : {results_csv}")

    # ── dataset ──────────────────────────────────────────────────────
    print("Loading validation dataset …")
    val_ds = dataset_sr(h5_path=VAL_H5, snapshot_index=SNAPSHOT_INDEX)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Validation samples: {len(val_ds)}")

    # ── metrics ──────────────────────────────────────────────────────
    rv = _get_registered_variables_3d()
    vx_idx, vy_idx, vz_idx = (
        rv.velocity_index.x,
        rv.velocity_index.y,
        rv.velocity_index.z,
    )
    metrics = [
        MSEMetric(),
        ChannelMSEMetric("Loss_pressure", rv.pressure_index),
        ChannelMSEMetric("Loss_density", rv.density_index),
        ChannelMSEMetric("Loss_vx", vx_idx),
        ChannelMSEMetric("Loss_vy", vy_idx),
        ChannelMSEMetric("Loss_vz", vz_idx),
        VelocityNormMSEMetric(vx_idx, vy_idx, vz_idx),
        VorticityMSEMetric(vx_idx, vy_idx, vz_idx),
        SpectralMSEMetric(),
        EnergyMassConservationMetric(),
        PerceptualLoss3D(DEVICE),
        PSNRMetric(),
        SSIMMetric(),
    ]

    # ── norm stats (only used by norm-on manifest entries) ───────────
    norm_stats = load_norm_stats(manifest, DEVICE)

    # ── run evaluations (one model at a time) ────────────────────────
    results = []
    for entry in manifest["models"]:
        label = entry["label"]
        eval_scales = entry.get("eval_scales", [UPSAMPLE_FACTOR])
        try:
            print(f"Loading {label} …")
            run = build_model(entry, DEVICE, norm_stats)
            for upsample_factor in eval_scales:
                res = evaluate_model(
                    run=run,
                    val_loader=val_loader,
                    metrics=metrics,
                    label=label,
                    upsample_factor=upsample_factor,
                )
                results.append(res)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ {label}: OOM — skipping")
            else:
                raise
        finally:
            if "run" in locals():
                del run.model
            gc.collect()
            torch.cuda.empty_cache()

    # ── report ───────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    cols = [
        "model",
        "upsample_factor",
        "MSE",
        "Loss_pressure",
        "Loss_density",
        "Loss_vx",
        "Loss_vy",
        "Loss_vz",
        "Loss_v_norm",
        "Loss_vorticity",
        "Spectral_MSE",
        "energy_mass_conservation",
        "Perceptual",
        "PSNR",
        "SSIM",
        "time_s",
    ]
    df = df[[c for c in cols if c in df.columns]]

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")


if __name__ == "__main__":
    main()
