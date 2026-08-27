"""
Train CFNO shift=8 (normalized) with MSE + spectral loss at 3 spectral weights.

Reads ``calibration.json`` (produced by ``calibrate_weights.py``) to obtain
three spectral weights — ``w_minor`` (0.1x), ``w_equal`` (1x), ``w_major``
(10x) the equalizing ratio — and trains one CFNO shift=8 model per weight.

All CFNO hyper-parameters, normalization stats, dataset, snapshot index,
optimizer, scheduler, skip-connection settings, and early-stopping settings
mirror ``experiments/loss_ablation/train_loss_ablation.py`` so the runs are
directly comparable to that ablation. The only varying axis across the three
runs is the spectral-loss weight.

Normalization statistics are reused from ``experiments/l1_spectral_weighting``
so all weighting/ablation experiments share identical normalization.

If ``calibration.json`` is missing, hardcoded fallback weights are used
(and a warning is printed).

After training, the script:

1. Evaluates the three trained models using the benchmark metrics from
   ``evaluation.benchmark`` (MSE, per-channel, vorticity, spectral,
   perceptual, PSNR, SSIM) at scales x4 and x2.
2. Writes ``benchmark_metrics.csv`` (scratch + repo copy).

Bar charts and the final-snapshot comparison are produced by separate scripts
(``run.sh`` invokes ``evaluation/comparing_models_bar_chart.py`` and
``plot_final_snapshot.py``).

Usage
-----
    python experiments/mse_spectral_weighting/calibrate_weights.py   # run first
    python experiments/mse_spectral_weighting/train_mse_spectral.py
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
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    load_manifest,
    load_norm_stats,
)

# ── Paths ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states.h5")
VAL_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
OUTPUT_ROOT = Path("/export/scratch/jalegria/experiments")
FOLDER_NAME = "mse_spectral_weighting_"
REPO_EXP_DIR = ROOT / "experiments" / "mse_spectral_weighting"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the l1_spectral_weighting normalization stats (do NOT regenerate) so
# results are directly comparable to that experiment and to loss_ablation.
L1_SPECTRAL_DIR = ROOT / "experiments" / "l1_spectral_weighting"
NORM_STATS_PATH = L1_SPECTRAL_DIR / "normalization_stats.npz"

CALIBRATION_JSON = REPO_EXP_DIR / "calibration.json"

# ── Data ──────────────────────────────────────────────────────────────

SNAPSHOT_INDEX = 79
UPSAMPLE_FACTOR = 4

# ── Training hyper-parameters (mirror loss_ablation) ───────────────────

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

# ── Skip-connection config (fixed for all runs; weight is the only axis) ──

SKIP_CONNECTION = True
INTERPOLATION_MODE = "trilinear"
SKIP_CONNECTION_INTERPOLATION_MODE = "trilinear"

# ── Fallback weights (used if calibration.json is missing) ────────────

FALLBACK_WEIGHTS = {
    "w_minor": 0.001,
    "w_equal": 0.01,
    "w_major": 0.1,
}


# =====================================================================
# Velocity indices (don't hardcode — AGENTS.md)
# =====================================================================


@lru_cache(maxsize=1)
def _get_velocity_indices() -> tuple[int, int, int]:
    from jf1uids import SimulationConfig, get_registered_variables
    from jf1uids.option_classes.simulation_config import finalize_config

    cfg = finalize_config(SimulationConfig(dimensionality=3), (5, 128, 128, 128))
    rv = get_registered_variables(cfg)
    return rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z


# =====================================================================
# Losses
# =====================================================================


class SpectralLoss(nn.Module):
    """Velocity-only torch-FFT log-power spectral loss."""

    def __init__(self, vx_idx: int, vy_idx: int, vz_idx: int, pool: int = SPECTRAL_POOL):
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


class MSESpectralLoss(nn.Module):
    """MSE + weighted velocity spectral loss."""

    def __init__(self, spectral_weight: float, vx_idx: int, vy_idx: int, vz_idx: int):
        super().__init__()
        self.mse = nn.MSELoss()
        self.spectral = SpectralLoss(vx_idx, vy_idx, vz_idx)
        self.spectral_weight = spectral_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = self.mse(pred, target)
        spec, _ = self.spectral(pred, target)
        total = mse + self.spectral_weight * spec
        return total, {
            "mse": mse.item(),
            "spectral": spec.item(),
            "total": total.item(),
        }


# =====================================================================
# Helpers
# =====================================================================


def _load_weights() -> dict[str, float]:
    """Load the 3 spectral weights from calibration.json (or fallback)."""
    if CALIBRATION_JSON.exists():
        with open(CALIBRATION_JSON) as f:
            cal = json.load(f)
        weights = cal["weights"]
        print(f"  Loaded weights from {CALIBRATION_JSON}")
        for name, w in weights.items():
            print(f"    {name:10s} = {w:.6f}")
        return weights
    print(f"  WARNING: {CALIBRATION_JSON} not found — using fallback weights.")
    print(f"  Run calibrate_weights.py first for data-driven weights.")
    for name, w in FALLBACK_WEIGHTS.items():
        print(f"    {name:10s} = {w:.6f} (fallback)")
    return FALLBACK_WEIGHTS


def _build_run_grid(weights: dict[str, float]) -> list[dict]:
    """Build 3 runs, one per spectral weight."""
    base = {
        "model_type": "cfno",
        "shifting_modes": SHIFTING_MODES,
        "skip_connection": SKIP_CONNECTION,
        "interpolation_mode": INTERPOLATION_MODE,
        "skip_connection_interpolation_mode": SKIP_CONNECTION_INTERPOLATION_MODE,
        "apply_positivity_relu": APPLY_POSITIVITY_RELU,
        "use_norm": True,
        "eval_scales": [4, 2],
    }
    label_map = {
        "w_minor": "MSE + spectral (minor)",
        "w_equal": "MSE + spectral (equal)",
        "w_major": "MSE + spectral (major)",
    }
    runs = []
    for weight_name, w in weights.items():
        tag = f"cfno_shift8_mse_spectral_{weight_name}"
        runs.append(
            {
                **base,
                "name": tag,
                "label": label_map[weight_name],
                "spectral_weight": w,
                "weight_name": weight_name,
            }
        )
    return runs


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
            "spectral_weight": run_config["spectral_weight"],
            "weight_name": run_config["weight_name"],
            "skip_connection": run_config["skip_connection"],
            "interpolation_mode": run_config["interpolation_mode"],
            "skip_connection_interpolation_mode": run_config[
                "skip_connection_interpolation_mode"
            ],
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
            "type": "mse_spectral",
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
) -> float:
    name = run_config["name"]
    sw = run_config["spectral_weight"]

    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"  spectral_weight={sw:.6f}  ({run_config['weight_name']})")
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

    vx, vy, vz = _get_velocity_indices()
    loss_fn = MSESpectralLoss(sw, vx, vy, vz)
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
    _save_artifacts(run_dir, run_config, best_state, train_losses, val_losses, best_val_loss)
    print(f"  Saved to {run_dir}")
    print(f"  Best val loss: {best_val_loss:.6f}")

    del model, best_state, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_loss


# =====================================================================
# Manifest
# =====================================================================


def _write_manifest(output_dir: Path, runs: list[dict]) -> dict:
    """Manifest with ONLY the three trained cfno models."""
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
                "loss": "mse_spectral",
                "spectral_weight": run["spectral_weight"],
                "weight_name": run["weight_name"],
                "skip_connection": run["skip_connection"],
                "interpolation_mode": run["interpolation_mode"],
                "skip_connection_interpolation_mode": run[
                    "skip_connection_interpolation_mode"
                ],
                "supports_variable_scale": True,
                "eval_scales": run["eval_scales"],
            }
        )

    manifest = {
        "experiment": "mse_spectral_weighting",
        "output_dir": str(output_dir),
        "benchmark_csv": "experiments/mse_spectral_weighting/benchmark_metrics.csv",
        "benchmark_row_key": "run",
        "benchmark_row_value_field": "name",
        "normalization_stats": str(NORM_STATS_PATH),
        "upsample_factor": UPSAMPLE_FACTOR,
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


def _run_eval_and_plots(manifest: dict, output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION")
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

    del val_loader_raw, val_ds_raw
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "If given, skip training and only run eval against this "
            "existing run-group folder (containing manifest.json)."
        ),
    )
    args = parser.parse_args()

    rv = _get_registered_variables_3d()
    print(
        f"  velocity_index: x={rv.velocity_index.x} "
        f"y={rv.velocity_index.y} z={rv.velocity_index.z}"
    )

    # ── eval-only mode (--manifest) ──────────────────────────────────
    if args.manifest:
        manifest_dir = Path(args.manifest)
        manifest = load_manifest(manifest_dir)
        output_dir = Path(manifest["output_dir"])
        print(f"Eval-only mode. output_dir={output_dir}")
        _run_eval_and_plots(manifest, output_dir)
        return

    # ── full train + eval ────────────────────────────────────────────
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

    # ── load calibration weights ────────────────────────────────────
    weights = _load_weights()

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

    # ── run grid ────────────────────────────────────────────────────
    runs = _build_run_grid(weights)
    print(f"\nRun grid: {len(runs)} trained runs")
    for r in runs:
        print(
            f"  - {r['name']}  spectral_weight={r['spectral_weight']:.6f}  "
            f"interp={r['interpolation_mode']}  skip={r['skip_connection']}"
        )

    # ── train ───────────────────────────────────────────────────────
    results = {}
    for run_config in runs:
        if _already_trained(output_dir, run_config["name"]):
            print(f"  Already trained: {run_config['name']} — skipping training.")
            results[run_config["name"]] = {
                "status": "skipped",
                "spectral_weight": run_config["spectral_weight"],
                "weight_name": run_config["weight_name"],
            }
            continue
        try:
            best_val = train_one_run(run_config, train_loader, val_loader, output_dir)
            results[run_config["name"]] = {
                "status": "success",
                "best_val_loss": best_val,
                "spectral_weight": run_config["spectral_weight"],
                "weight_name": run_config["weight_name"],
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[run_config["name"]] = {
                "status": "failed",
                "error": str(e),
                "spectral_weight": run_config["spectral_weight"],
                "weight_name": run_config["weight_name"],
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
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(
        [
            {
                "run_name": n,
                "status": r["status"],
                "best_val_loss": r.get("best_val_loss"),
                "spectral_weight": r.get("spectral_weight"),
            }
            for n, r in results.items()
        ]
    ).to_csv(output_dir / "training_summary.csv", index=False)

    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    # ── write manifest (only the 3 trained cfno models) ──────────────
    manifest = _write_manifest(output_dir, runs)
    print(f"\nManifest written to {output_dir / 'manifest.json'}")

    _run_eval_and_plots(manifest, output_dir)
    print(f"\nAll artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()