"""
Skip-connection ablation for CFNO shift=8 with L1 + spectral loss (w_minor).

This single self-contained script trains two CFNO shift=8 (normalized) models
that activate the FNO_2 ``skip_connection`` branch, one per interpolation mode
(``nearest`` and ``trilinear`` — the skip-branch mode is tied to the main
``interpolation_mode`` per the experiment design).  All loss/training
hyper-parameters mirror ``experiments/l1_spectral_weighting/train_l1_spectral.py``
using only the ``w_minor`` spectral weight from ``calibration.json``.

After training, the script:

1. Evaluates the two trained skip runs PLUS the prior best l1_spectral
   ``w_minor`` model (skip=False, loaded from the existing l1_spectral_weighting
   run dir) and a trilinear baseline, using the benchmark metrics from
   ``evaluation.benchmark`` (MSE, per-channel, vorticity, spectral,
   perceptual, PSNR, SSIM) at scales x4 and x2.
2. Writes ``benchmark_metrics.csv`` (scratch + repo copy) and emits two
   grouped bar-chart PNGs comparing the runs at x4.
3. Runs a fresh jf1uids turbulent simulation (seed 1234, 128^3), block-averages
   to LR, runs every model, and produces a final-snapshot state-grid
   comparison (HR | skip-nearest | skip-trilinear | skip=False ref |
   Trilinear | LR).  States are cached to ``comparison_states.npy``.

Usage
-----
    bash experiments/l1_spectral_skip_connection/run.sh
    python experiments/l1_spectral_skip_connection/train_l1_spectral_skip_connection.py
    python experiments/l1_spectral_skip_connection/train_l1_spectral_skip_connection.py \
        --regen            # force fresh sim + inference
    python experiments/l1_spectral_skip_connection/train_l1_spectral_skip_connection.py \
        --manifest <dir>   # eval/plot an existing run-group (skip training)
"""

from autocvd import autocvd

autocvd(num_gpus=1)
import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader.dataloader_3d import dataset_sr
from src.model.models_fno_2 import FNO_2

from evaluation.benchmark import (
    _get_registered_variables_3d,
    evaluate_model as benchmark_evaluate_model,
    MSEMetric,
    ChannelMSEMetric,
    VelocityNormMSEMetric,
    VorticityMSEMetric,
    SpectralMSEMetric,
    PerceptualLoss3D,
    PSNRMetric,
    SSIMMetric,
)
from evaluation.manifest import (
    build_model,
    discover_latest,
    load_manifest,
    load_norm_stats,
)

# jf1uids sim imports (used only in the final-snapshot section)
from astropy import units as u
import astropy.constants as c
import jax.numpy as jnp
from jf1uids import (
    CodeUnits,
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
)
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids.initial_condition_generation.turb import create_turb_field
from jf1uids.option_classes.simulation_config import HLL, FORWARDS, finalize_config
from jf1uids.time_stepping.time_integration import time_integration

# ── Paths ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states.h5")
VAL_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
OUTPUT_ROOT = Path("/export/scratch/jalegria/experiments")
FOLDER_NAME = "l1_spectral_skip_connection_"
REPO_EXP_DIR = ROOT / "experiments" / "l1_spectral_skip_connection"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the l1_spectral_weighting normalization stats + calibration (do NOT
# regenerate) so results are directly comparable to that experiment.
L1_SPECTRAL_DIR = ROOT / "experiments" / "l1_spectral_weighting"
CALIBRATION_JSON = L1_SPECTRAL_DIR / "calibration.json"
NORM_STATS_PATH = L1_SPECTRAL_DIR / "normalization_stats.npz"
L1_SPECTRAL_SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
L1_SPECTRAL_GLOB = "l1_spectral_weighting_*"

# ── Data ──────────────────────────────────────────────────────────────

SNAPSHOT_INDEX = 79
UPSAMPLE_FACTOR = 4

# ── Training hyper-parameters (mirror l1_spectral_weighting) ───────────

EPOCHS = 250
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NOISE_STD = 0.01
NUM_WORKERS = 2
USE_AMP = False
GRAD_ACCUM = 8
EARLY_STOP_PATIENCE = 40
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 15

# ── CFNO hyper-parameters (shift=8) ───────────────────────────────────

SHIFTING_MODES = 8

CFNO_PARAMS = dict(
    in_channel=5,
    n_channels=32,
    n_residual_blocks=3,
    n_operator_blocks=2,
    modes=16,
    apply_constraint=False,
    last_layer_kernel=3,
)

APPLY_POSITIVITY_RELU = False

SPECTRAL_POOL = 2

# ── Fallback weight (matches l1_spectral_weighting FALLBACK_WEIGHTS) ──

FALLBACK_W_MINOR = 0.001

# ── Final-snapshot sim constants ───────────────────────────────────────

HR_NUM_CELLS = 128
SEED = 1234
CHANNEL_NAMES = ["density", "vx", "vy", "vz", "pressure"]


# =====================================================================
# Losses (verbatim from l1_spectral_weighting/train_l1_spectral.py)
# =====================================================================


class SpectralLoss(nn.Module):
    """Velocity-only torch-FFT log-power spectral loss."""

    def __init__(
        self, vx_idx: int, vy_idx: int, vz_idx: int, pool: int = SPECTRAL_POOL
    ):
        super().__init__()
        self.vel_idx = [vx_idx, vy_idx, vz_idx]
        self.pool = pool

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
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
        pred_log = torch.log10(pred_power + 1e-12)
        target_log = torch.log10(target_power + 1e-12)
        spec = F.mse_loss(pred_log, target_log)
        return spec, {"spectral": spec.item()}


class L1SpectralLoss(nn.Module):
    """L1 + weighted velocity spectral loss."""

    def __init__(self, spectral_weight: float, vx_idx: int, vy_idx: int, vz_idx: int):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.spectral = SpectralLoss(vx_idx, vy_idx, vz_idx)
        self.spectral_weight = spectral_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = self.l1(pred, target)
        spec, _ = self.spectral(pred, target)
        total = l1 + self.spectral_weight * spec
        return total, {
            "l1": l1.item(),
            "spectral": spec.item(),
            "total": total.item(),
        }


# =====================================================================
# Weight loading
# =====================================================================


def _load_w_minor() -> float:
    """Load the w_minor spectral weight from calibration.json (or fallback)."""
    if CALIBRATION_JSON.exists():
        with open(CALIBRATION_JSON) as f:
            cal = json.load(f)
        w = cal["weights"]["w_minor"]
        print(f"  Loaded w_minor from {CALIBRATION_JSON}: {w:.6f}")
        return w
    print(f"  WARNING: {CALIBRATION_JSON} not found — using fallback.")
    print(f"    w_minor = {FALLBACK_W_MINOR:.6f} (fallback)")
    return FALLBACK_W_MINOR


# =====================================================================
# Run grid
# =====================================================================


def _build_run_grid(w_minor: float) -> list[dict]:
    """Two trained runs: skip on, interp tied to skip_interp."""
    base = {
        "model_type": "cfno",
        "shifting_modes": SHIFTING_MODES,
        "spectral_weight": w_minor,
        "weight_name": "w_minor",
        "skip_connection": True,
        "apply_positivity_relu": APPLY_POSITIVITY_RELU,
        "use_norm": True,
        "eval_scales": [4, 2],
    }
    return [
        {
            **base,
            "name": "cfno_shift8_skip_nearest",
            "label": "skip=nearest",
            "interpolation_mode": "nearest",
            "skip_connection_interpolation_mode": "nearest",
        },
        {
            **base,
            "name": "cfno_shift8_skip_trilinear",
            "label": "skip=trilinear",
            "interpolation_mode": "trilinear",
            "skip_connection_interpolation_mode": "trilinear",
        },
    ]


def _build_model(run_config: dict) -> nn.Module:
    """Instantiate FNO_2 directly with the run's skip/interp settings."""
    return FNO_2(
        **CFNO_PARAMS,
        shifting_modes=run_config["shifting_modes"],
        interpolation_mode=run_config["interpolation_mode"],
        skip_connection=run_config["skip_connection"],
        skip_connection_interpolation_mode=run_config[
            "skip_connection_interpolation_mode"
        ],
        apply_positivity_relu=run_config["apply_positivity_relu"],
    )


def _forward(model: nn.Module, lr: torch.Tensor) -> torch.Tensor:
    return model(lr, upsample_factor=UPSAMPLE_FACTOR)


# =====================================================================
# Artifacts
# =====================================================================


def _save_artifacts(
    run_dir: Path,
    run_config: dict,
    best_state: dict,
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "weights.pt")

    df = pd.DataFrame(
        {
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    df.to_csv(run_dir / "losses.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"{run_config['name']} — spectral_weight={run_config['spectral_weight']:.6f}"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    summary = {
        "run": {
            "name": run_config["name"],
            "label": run_config["label"],
            "skip_connection": run_config["skip_connection"],
            "interpolation_mode": run_config["interpolation_mode"],
            "skip_connection_interpolation_mode": run_config[
                "skip_connection_interpolation_mode"
            ],
            "spectral_weight": run_config["spectral_weight"],
            "weight_name": run_config["weight_name"],
        },
        "model_params": {
            **CFNO_PARAMS,
            "shifting_modes": run_config["shifting_modes"],
            "interpolation_mode": run_config["interpolation_mode"],
            "skip_connection": run_config["skip_connection"],
            "skip_connection_interpolation_mode": run_config[
                "skip_connection_interpolation_mode"
            ],
        },
        "apply_positivity_relu": run_config["apply_positivity_relu"],
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": SCHED_FACTOR,
                "patience": SCHED_PATIENCE,
            },
            "noise_std": NOISE_STD,
            "use_amp": USE_AMP,
            "snapshot_index": SNAPSHOT_INDEX,
        },
        "loss": {
            "type": "l1_spectral",
            "spectral_weight": run_config["spectral_weight"],
            "spectral_pool": SPECTRAL_POOL,
        },
        "normalization": {
            "enabled": True,
            "stats_path": str(NORM_STATS_PATH),
        },
        "results": {
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
        },
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# =====================================================================
# Training loop
# =====================================================================


def _already_trained(output_dir: Path, run_name: str) -> bool:
    return (output_dir / run_name / "weights.pt").exists()


def train_one_run(
    run_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    w_minor: float,
) -> float:
    name = run_config["name"]
    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"  spectral_weight (w_minor)={w_minor:.6f}")
    print(
        f"  skip={run_config['skip_connection']}  interp={run_config['interpolation_mode']}"
    )
    print(f"{'=' * 60}")

    model = _build_model(run_config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
    )

    rv = _get_registered_variables_3d()
    vx, vy, vz = (
        rv.velocity_index.x,
        rv.velocity_index.y,
        rv.velocity_index.z,
    )
    loss_fn = L1SpectralLoss(w_minor, vx, vy, vz)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            hr = batch[0].to(DEVICE, non_blocking=True)
            lr = batch[1].to(DEVICE, non_blocking=True)
            lr = lr + torch.randn_like(lr) * NOISE_STD

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    output = _forward(model, lr)
                    loss, _ = loss_fn(output, hr)
                    loss = loss / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                output = _forward(model, lr)
                loss, _ = loss_fn(output, hr)
                loss = loss / GRAD_ACCUM
                loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                hr = batch[0].to(DEVICE, non_blocking=True)
                lr = batch[1].to(DEVICE, non_blocking=True)
                output = _forward(model, lr)
                loss, _ = loss_fn(output, hr)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"train={epoch_loss:.6f}  val={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | no_improve={epochs_no_improve}/{EARLY_STOP_PATIENCE} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch + 1}.")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    run_dir = output_dir / name
    _save_artifacts(
        run_dir, run_config, best_state, train_losses, val_losses, best_val_loss
    )
    print(f"  Saved to {run_dir}")
    print(f"  Best val loss: {best_val_loss:.6f}")

    del model, best_state, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_loss


# =====================================================================
# Manifest
# =====================================================================


def _adopt_reference_entry() -> dict | None:
    """Pull the w_minor entry from the latest l1_spectral_weighting manifest.

    Returns a manifest entry renamed to ``cfno_shift8_skip_off_ref`` whose
    ``model_params`` are rewritten explicitly with ``skip_connection=False``
    and the FNO_2 interpolation defaults, so it loads cleanly into the patched
    FNO_2 and serves as the skip=False reference row in eval/plots.
    """
    try:
        ref_dir = discover_latest(L1_SPECTRAL_SCRATCH_BASE, L1_SPECTRAL_GLOB)
    except FileNotFoundError:
        print(f"  No l1_spectral_weighting manifest found under {L1_SPECTRAL_GLOB}")
        return None
    ref_manifest = load_manifest(ref_dir)
    for e in ref_manifest["models"]:
        if e.get("model_type") == "cfno" and e.get("weight_name") == "w_minor":
            params = dict(e.get("model_params") or {})
            params.setdefault("interpolation_mode", "trilinear")
            params["skip_connection"] = False
            params["skip_connection_interpolation_mode"] = "nearest"
            return {
                "name": "cfno_shift8_skip_off_ref",
                "label": "skip=False (w_minor ref)",
                "model_type": "cfno",
                "model_params": params,
                "weights": e["weights"],
                "apply_positivity_relu": e.get("apply_positivity_relu", False),
                "use_norm": e.get("use_norm", True),
                "loss": "l1_spectral",
                "weight_name": "w_minor",
                "spectral_weight": e.get("spectral_weight"),
                "skip_connection": False,
                "interpolation_mode": params["interpolation_mode"],
                "skip_connection_interpolation_mode": "nearest",
                "supports_variable_scale": True,
                "eval_scales": [4, 2],
            }
    print("  No w_minor cfno entry in the l1_spectral_weighting manifest.")
    return None


def _write_manifest(output_dir: Path, runs: list[dict], w_minor: float) -> dict:
    models = []
    for run in runs:
        params = {
            **CFNO_PARAMS,
            "shifting_modes": run["shifting_modes"],
            "interpolation_mode": run["interpolation_mode"],
            "skip_connection": run["skip_connection"],
            "skip_connection_interpolation_mode": run[
                "skip_connection_interpolation_mode"
            ],
        }
        models.append(
            {
                "name": run["name"],
                "label": run["label"],
                "model_type": "cfno",
                "model_params": params,
                "weights": str(output_dir / run["name"] / "weights.pt"),
                "apply_positivity_relu": run["apply_positivity_relu"],
                "use_norm": run["use_norm"],
                "loss": "l1_spectral",
                "weight_name": run["weight_name"],
                "spectral_weight": run["spectral_weight"],
                "skip_connection": run["skip_connection"],
                "interpolation_mode": run["interpolation_mode"],
                "skip_connection_interpolation_mode": run[
                    "skip_connection_interpolation_mode"
                ],
                "supports_variable_scale": True,
                "eval_scales": run["eval_scales"],
            }
        )

    ref = _adopt_reference_entry()
    if ref is not None:
        models.append(ref)

    models.append(
        {
            "name": "trilinear_baseline",
            "label": "Trilinear",
            "model_type": "trilinear",
            "model_params": {},
            "weights": None,
            "apply_positivity_relu": False,
            "use_norm": False,
            "loss": "none",
            "supports_variable_scale": True,
            "eval_scales": [4, 2],
        }
    )

    manifest = {
        "experiment": "l1_spectral_skip_connection",
        "output_dir": str(output_dir),
        "benchmark_csv": "experiments/l1_spectral_skip_connection/benchmark_metrics.csv",
        "benchmark_row_key": "run",
        "benchmark_row_value_field": "name",
        "normalization_stats": str(NORM_STATS_PATH),
        "upsample_factor": UPSAMPLE_FACTOR,
        "spectral_weight": w_minor,
        "weight_name": "w_minor",
        "models": models,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


# =====================================================================
# Evaluation
# =====================================================================


def _build_metrics() -> list:
    rv = _get_registered_variables_3d()
    vx, vy, vz = (
        rv.velocity_index.x,
        rv.velocity_index.y,
        rv.velocity_index.z,
    )
    return [
        MSEMetric(),
        ChannelMSEMetric("Loss_pressure", rv.pressure_index),
        ChannelMSEMetric("Loss_density", rv.density_index),
        ChannelMSEMetric("Loss_vx", vx),
        ChannelMSEMetric("Loss_vy", vy),
        ChannelMSEMetric("Loss_vz", vz),
        VelocityNormMSEMetric(vx, vy, vz),
        VorticityMSEMetric(vx, vy, vz),
        SpectralMSEMetric(),
        PerceptualLoss3D(DEVICE),
        PSNRMetric(),
        SSIMMetric(),
    ]


def _evaluate_all(
    manifest: dict, val_loader_raw: DataLoader, metrics: list, norm_stats
) -> list[dict]:
    rows = []
    for entry in manifest["models"]:
        label = entry["label"]
        if entry.get("model_type") == "trilinear":
            run_label = "Trilinear"
        else:
            run_label = entry["name"]
        try:
            run = build_model(entry, DEVICE, norm_stats)
            for scale in entry.get("eval_scales", [UPSAMPLE_FACTOR]):
                res = benchmark_evaluate_model(
                    run=run,
                    val_loader=val_loader_raw,
                    metrics=metrics,
                    label=label,
                    upsample_factor=scale,
                )
                res["run"] = run_label
                res["skip_connection"] = entry.get("skip_connection")
                res["interpolation_mode"] = entry.get("interpolation_mode")
                res["skip_connection_interpolation_mode"] = entry.get(
                    "skip_connection_interpolation_mode"
                )
                rows.append(res)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ {label}: OOM — skipping")
            else:
                print(f"  ✗ {label}: RuntimeError — {e}")
        finally:
            if "run" in locals():
                del run.model
            gc.collect()
            torch.cuda.empty_cache()
    return rows


# =====================================================================
# Bar charts
# =====================================================================


_BAR_GROUPS = [
    (
        "bar_mse_channel.png",
        "L1+spectral skip-connection: MSE & per-channel (x4)",
        [
            ("MSE", "↓"),
            ("Loss_density", "↓"),
            ("Loss_pressure", "↓"),
            ("Loss_vx", "↓"),
            ("Loss_vy", "↓"),
            ("Loss_vz", "↓"),
            ("Loss_v_norm", "↓"),
            ("Loss_vorticity", "↓"),
        ],
    ),
    (
        "bar_spectral_perceptual_psnr_ssim.png",
        "L1+spectral skip-connection: spectral / perceptual / PSNR / SSIM (x4)",
        [
            ("Spectral_MSE", "↓"),
            ("Perceptual", "↓"),
            ("PSNR", "↑"),
            ("SSIM", "↑"),
        ],
    ),
]


def _plot_bar_charts(eval_df: pd.DataFrame, output_dir: Path) -> None:
    x4 = eval_df[eval_df["upsample_factor"] == 4].copy()
    if x4.empty:
        print("  No x4 rows — skipping bar charts")
        return
    # one row per run (use the label column as the series name)
    label_col = "model"
    labels = x4[label_col].tolist()
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(labels)))

    for fname, title, metrics in _BAR_GROUPS:
        present = [m for m, _ in metrics if m in x4.columns]
        if not present:
            continue
        n = len(present)
        n_cols = min(4, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.4 * n_rows))
        axes = axes.flatten() if n > 1 else [axes]
        bar_width = 0.8 / max(len(labels), 1)

        for idx, metric in enumerate(present):
            direction = dict(metrics)[metric]
            ax = axes[idx]
            vals = x4[metric].astype(float).values
            finite = np.isfinite(vals)
            if not finite.any():
                ax.set_visible(False)
                continue
            for i, (lbl, v) in enumerate(zip(labels, vals)):
                if not np.isfinite(v):
                    continue
                ax.bar(i * bar_width, v, bar_width, color=colors[i], label=lbl)
            ax.set_xticks([])
            ax.set_title(f"{metric} {direction}", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            vmin, vmax = float(vals[finite].min()), float(vals[finite].max())
            margin = (vmax - vmin) * 0.15 if vmax > vmin else abs(vmax) * 0.1 + 1e-6
            ax.set_ylim(max(0, vmin - margin), vmax + margin)

        for idx in range(len(present), len(axes)):
            axes[idx].set_visible(False)

        handles, hlabels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                hlabels,
                loc="lower center",
                ncol=min(len(labels), 5),
                fontsize=8,
            )
        fig.suptitle(title, fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        out = output_dir / fname
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# =====================================================================
# Final-snapshot simulation (mirrors l1_spectral_weighting plot script)
# =====================================================================


def _load_turbulent_cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["turbulent_sim"]


def _downaverage_state(state: np.ndarray, downsample_factor: int) -> np.ndarray:
    channels, hx, hy, hz = state.shape
    if hx % downsample_factor or hy % downsample_factor or hz % downsample_factor:
        raise ValueError(
            f"State spatial shape {(hx, hy, hz)} not divisible by {downsample_factor}."
        )
    lx, ly, lz = (
        hx // downsample_factor,
        hy // downsample_factor,
        hz // downsample_factor,
    )
    reshaped = state.reshape(
        channels, lx, downsample_factor, ly, downsample_factor, lz, downsample_factor
    )
    return reshaped.mean(axis=(2, 4, 6))


def _generate_hr_state(num_cells: int, seed: int) -> np.ndarray:
    cfg_data = _load_turbulent_cfg()
    np.random.seed(seed)

    config = SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=False,
        progress_bar=False,
        dimensionality=3,
        num_ghost_cells=int(cfg_data["num_ghost_cells"]),
        box_size=float(cfg_data["box_size"]),
        num_cells=num_cells,
        fixed_timestep=False,
        differentiation_mode=FORWARDS,
        riemann_solver=HLL,
        mhd=False,
        return_snapshots=True,
        num_snapshots=2,
    )
    helper_data = get_helper_data(config)
    reg_vars = get_registered_variables(config)

    code_units = CodeUnits(3 * u.parsec, 1 * u.M_sun, 100 * u.km / u.s)
    t_end = (1.0e4 * u.yr).to(code_units.code_time).value
    params = SimulationParams(
        C_cfl=0.4,
        dt_max=float(cfg_data["dt_max"]),
        gamma=5 / 3,
        t_end=t_end,
    )

    rho_0 = 2 * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B
    rho = jnp.ones((num_cells,) * 3) * rho_0.to(code_units.code_density).value
    p = jnp.ones((num_cells,) * 3) * p_0.to(code_units.code_pressure).value

    for _ in range(8):
        u_x = create_turb_field(
            num_cells,
            1,
            cfg_data["turbulence_slope"],
            cfg_data["kmin"],
            cfg_data["kmax"],
        )
        u_y = create_turb_field(
            num_cells,
            1,
            cfg_data["turbulence_slope"],
            cfg_data["kmin"],
            cfg_data["kmax"],
        )
        u_z = create_turb_field(
            num_cells,
            1,
            cfg_data["turbulence_slope"],
            cfg_data["kmin"],
            cfg_data["kmax"],
        )
        rms = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))
        if not jnp.isfinite(rms) or float(rms) == 0.0:
            continue
        wanted_rms = (
            (float(cfg_data["wanted_rms"]) * u.km / u.s)
            .to(code_units.code_velocity)
            .value
        )
        u_x = u_x / rms * wanted_rms
        u_y = u_y / rms * wanted_rms
        u_z = u_z / rms * wanted_rms

        initial_state = construct_primitive_state(
            config=config,
            registered_variables=reg_vars,
            density=rho,
            velocity_x=u_x,
            velocity_y=u_y,
            velocity_z=u_z,
            gas_pressure=p,
        )
        config_run = finalize_config(config, initial_state.shape)
        result = time_integration(
            initial_state, config_run, params, helper_data, reg_vars
        )
        snapshot = np.array(result.states[-1], dtype=np.float32)
        if np.isfinite(snapshot).all():
            return snapshot
    raise RuntimeError("Could not generate a finite turbulent HR state.")


def _plot_state_grid(
    rows: list[tuple[str, np.ndarray]],
    save_path: Path,
    title: str,
) -> None:
    n_rows = len(rows)
    n_cols = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx in range(n_cols):
        slices = []
        for _, state in rows:
            arr = state[col_idx]
            z_idx = arr.shape[-1] // 2
            slices.append(arr[:, :, z_idx].T)
        finite = [s for s in slices if np.isfinite(s).any()]
        vmin = min(float(np.nanmin(s)) for s in finite)
        vmax = max(float(np.nanmax(s)) for s in finite)
        for row_idx, (row_name, _) in enumerate(rows):
            ax = axes[row_idx, col_idx]
            img = ax.imshow(
                slices[row_idx],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(CHANNEL_NAMES[col_idx])
            if col_idx == 0:
                ax.set_ylabel(row_name)
            fig.colorbar(img, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def _build_states(regen: bool, manifest: dict, states_npy: Path) -> dict:
    if not regen and states_npy.exists():
        print(f"Loading cached states from {states_npy}")
        return np.load(states_npy, allow_pickle=True).item()

    print("Generating HR state via jf1uids …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr = _downaverage_state(hr, UPSAMPLE_FACTOR)
    print(f"  HR {hr.shape}  LR {lr.shape}")

    states = {"hr": hr, "lr": lr}
    norm_stats = load_norm_stats(manifest, DEVICE)

    lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        states["interp"] = (
            F.interpolate(
                lr_t,
                scale_factor=UPSAMPLE_FACTOR,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
    del lr_t

    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        name = entry["name"]
        print(f"  Running {name} …")
        try:
            run = build_model(entry, DEVICE, norm_stats)
            lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
            sr = run(lr_t, UPSAMPLE_FACTOR)
            states[name] = sr.squeeze(0).detach().cpu().numpy()
            del run.model, lr_t
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM running {name} — skipping")
            else:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    np.save(states_npy, states, allow_pickle=True)
    print(f"  Cached states to {states_npy}")
    return states


def _plot_final_snapshot(states: dict, manifest: dict, output_dir: Path) -> None:
    rows: list[tuple[str, np.ndarray]] = [("HR target", states["hr"])]
    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        name = entry["name"]
        if name in states:
            rows.append((entry["label"], states[name]))
    rows.append(("Trilinear", states["interp"]))
    rows.append(("LR", states["lr"]))
    _plot_state_grid(
        rows=rows,
        save_path=output_dir / "final_snapshot_comparison.png",
        title="CFNO shift=8 L1+spectral — x4: skip-connection comparison",
    )


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force fresh jf1uids sim + inference for the final-snapshot plot.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "If given, skip training and only run eval/plots against this "
            "existing run-group folder (containing manifest.json)."
        ),
    )
    args = parser.parse_args()

    rv = _get_registered_variables_3d()
    print(
        f"  velocity_index: x={rv.velocity_index.x} y={rv.velocity_index.y} z={rv.velocity_index.z}"
    )

    # ── eval-only mode (--manifest) ──────────────────────────────────
    if args.manifest:
        manifest_dir = Path(args.manifest)
        manifest = load_manifest(manifest_dir)
        output_dir = Path(manifest["output_dir"])
        print(f"Eval-only mode. output_dir={output_dir}")
        _run_eval_and_plots(manifest, output_dir, args.regen)
        return

    # ── full train + eval + plots ────────────────────────────────────
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    output_dir = OUTPUT_ROOT / f"{FOLDER_NAME}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {output_dir}")
    print(f"Device           : {DEVICE}")
    print(f"Train data       : {TRAIN_H5}")
    print(f"Val data         : {VAL_H5}")
    print(f"Snapshot index   : {SNAPSHOT_INDEX}")
    print(f"Epochs           : {EPOCHS} (patience {EARLY_STOP_PATIENCE})")
    print(
        f"Optimizer        : AdamW(lr={LEARNING_RATE}, wd={WEIGHT_DECAY}) + ReduceLROnPlateau"
    )
    print(f"Noise std        : {NOISE_STD}")
    print(
        f"Batch size       : {BATCH_SIZE} (accum {GRAD_ACCUM} -> eff {BATCH_SIZE * GRAD_ACCUM})"
    )
    print(f"AMP              : {USE_AMP}")

    w_minor = _load_w_minor()
    runs = _build_run_grid(w_minor)
    print(f"\nRun grid: {len(runs)} trained runs")
    for r in runs:
        print(
            f"  - {r['name']}  interp={r['interpolation_mode']}  "
            f"skip_interp={r['skip_connection_interpolation_mode']}"
        )

    # ── datasets (norm-on, reuse l1_spectral stats) ──────────────────
    print("\nLoading training dataset (normalized) …")
    train_ds = dataset_sr(
        h5_path=TRAIN_H5,
        snapshot_index=SNAPSHOT_INDEX,
        use_normalizing=True,
        mean_std_path=NORM_STATS_PATH,
    )
    print(f"  Training samples: {len(train_ds)}")

    print("Loading validation dataset (normalized, reusing train stats) …")
    val_ds = dataset_sr(
        h5_path=VAL_H5,
        snapshot_index=SNAPSHOT_INDEX,
        use_normalizing=True,
        means=(train_ds.mean_hr, train_ds.mean_lr),
        stds=(train_ds.std_hr, train_ds.std_lr),
    )
    print(f"  Validation samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── train ────────────────────────────────────────────────────────
    results = {}
    for run_config in runs:
        if _already_trained(output_dir, run_config["name"]):
            print(f"  Already trained: {run_config['name']} — skipping training.")
            results[run_config["name"]] = {
                "status": "skipped",
                "spectral_weight": run_config["spectral_weight"],
            }
            continue
        try:
            best_val = train_one_run(
                run_config, train_loader, val_loader, output_dir, w_minor
            )
            results[run_config["name"]] = {
                "status": "success",
                "best_val_loss": best_val,
                "spectral_weight": run_config["spectral_weight"],
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[run_config["name"]] = {
                "status": "failed",
                "error": str(e),
                "spectral_weight": run_config["spectral_weight"],
            }
            gc.collect()
            torch.cuda.empty_cache()

    # ── training summary ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        if result["status"] == "success":
            print(f"  {name:45s}  val_loss = {result['best_val_loss']:.6f}")
        else:
            print(f"  {name:45s}  {result['status'].upper()}")

    summary = {
        "output_dir": str(output_dir),
        "runs": results,
        "spectral_weight": w_minor,
        "weight_name": "w_minor",
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(
        [
            {
                "run_name": n,
                "status": r["status"],
                "best_val_loss": r.get("best_val_loss"),
            }
            for n, r in results.items()
        ]
    ).to_csv(output_dir / "training_summary.csv", index=False)

    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    # ── write manifest (incl. reference + trilinear) ─────────────────
    manifest = _write_manifest(output_dir, runs, w_minor)
    print(f"\nManifest written to {output_dir / 'manifest.json'}")

    _run_eval_and_plots(manifest, output_dir, args.regen)
    print(f"\nAll artifacts saved under: {output_dir}")


def _run_eval_and_plots(manifest: dict, output_dir: Path, regen: bool) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION + PLOTS")
    print("=" * 60)

    # raw (non-normalized) val loader — build_model applies norm internally
    print("Loading raw validation dataset (for benchmark) …")
    val_ds_raw = dataset_sr(h5_path=VAL_H5, snapshot_index=SNAPSHOT_INDEX)
    print(f"  Raw validation samples: {len(val_ds_raw)}")
    val_loader_raw = DataLoader(
        val_ds_raw,
        batch_size=2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    metrics = _build_metrics()
    norm_stats = load_norm_stats(manifest, DEVICE)

    eval_rows = _evaluate_all(manifest, val_loader_raw, metrics, norm_stats)
    eval_df = pd.DataFrame(eval_rows)
    preferred = [
        "run",
        "skip_connection",
        "interpolation_mode",
        "skip_connection_interpolation_mode",
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
        "Perceptual",
        "PSNR",
        "SSIM",
        "time_s",
    ]
    eval_df = eval_df[[c for c in preferred if c in eval_df.columns]]
    scratch_csv = output_dir / "benchmark_metrics.csv"
    eval_df.to_csv(scratch_csv, index=False)
    repo_csv = REPO_EXP_DIR / "benchmark_metrics.csv"
    shutil.copy(scratch_csv, repo_csv)
    print(f"  Saved {scratch_csv}")
    print(f"  Copied to {repo_csv}")

    _plot_bar_charts(eval_df, output_dir)

    print("\nBuilding final-snapshot comparison …")
    torch.manual_seed(SEED)
    states = _build_states(regen, manifest, output_dir / "comparison_states.npy")
    _plot_final_snapshot(states, manifest, output_dir)

    del val_loader_raw, val_ds_raw
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

