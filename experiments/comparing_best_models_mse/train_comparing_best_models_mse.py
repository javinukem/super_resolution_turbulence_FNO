"""
Train the canonical stabilized CFNO (Baseline B config: shift=8, skip=trilinear)
with **MSE-only loss** at 500 epochs, for a head-to-head comparison against the
best MSE-only EDSR (``edsr_norm_skip_on``) and the trilinear interpolation
baseline.

This is a copy of ``experiments/baseline_b_500_epochs/train_baseline_b_500.py``
restricted to the ``clip_p30`` run (lr=1e-3, grad clip 0.8, sched_patience=30)
and with the spectral term dropped — i.e. plain ``nn.MSELoss()``. Every other
hyper-parameter (model architecture, skip/interp settings, normalization stats,
batch/accum, early-stop patience, noise std, snapshot index, eval scales) is
kept identical so the comparison isolates the loss function.

After training, the script evaluates the model with ``evaluation.benchmark``
metrics (MSE, per-channel, vorticity, spectral, perceptual, PSNR, SSIM) at
scales x4 and x2, and writes ``benchmark_metrics.csv`` (scratch + repo copy).

The bar-chart comparison is produced by a separate script invoked from
``run.sh`` (``evaluation/comparing_models_bar_chart.py --preset
comparing_best_models_mse``) — no duplicated plotting code in this file.

Usage
-----
    bash experiments/comparing_best_models_mse/run.sh
    python experiments/comparing_best_models_mse/train_comparing_best_models_mse.py
    python experiments/comparing_best_models_mse/train_comparing_best_models_mse.py \\
        --manifest <dir>   # eval-only against an existing run-group
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
import pandas as pd
import torch
import torch.nn as nn
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
FOLDER_NAME = "comparing_best_models_mse_"
REPO_EXP_DIR = ROOT / "experiments" / "comparing_best_models_mse"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the l1_spectral_weighting normalization stats (do NOT regenerate) so
# results are directly comparable to the other CFNO baselines.
L1_SPECTRAL_DIR = ROOT / "experiments" / "l1_spectral_weighting"
NORM_STATS_PATH = L1_SPECTRAL_DIR / "normalization_stats.npz"

# ── Data ──────────────────────────────────────────────────────────────

SNAPSHOT_INDEX = 79
UPSAMPLE_FACTOR = 4

# ── Training hyper-parameters (mirror baseline_b_500 clip_p30 run) ─────

EPOCHS = 500
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NOISE_STD = 0.01
NUM_WORKERS = 2
USE_AMP = False
GRAD_ACCUM = 8
EARLY_STOP_PATIENCE = 40
SCHED_FACTOR = 0.8
SCHED_PATIENCE = 30
GRAD_CLIP_NORM = 0.8

# ── CFNO hyper-parameters (shift=8, skip=trilinear) ───────────────────

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

SKIP_CONNECTION = True
INTERPOLATION_MODE = "trilinear"
SKIP_CONNECTION_INTERPOLATION_MODE = "trilinear"

# ── Run config ────────────────────────────────────────────────────────

RUN_NAME = "cfno_shift8_mse_only_500e_clip_p30"
RUN_LABEL = "CFNO shift=8, MSE only (500e, clip=0.8, p=30)"
EVAL_SCALES = [4, 2]


def _build_run() -> dict:
    return {
        "name": RUN_NAME,
        "label": RUN_LABEL,
        "model_type": "cfno",
        "shifting_modes": SHIFTING_MODES,
        "skip_connection": SKIP_CONNECTION,
        "interpolation_mode": INTERPOLATION_MODE,
        "skip_connection_interpolation_mode": SKIP_CONNECTION_INTERPOLATION_MODE,
        "apply_positivity_relu": APPLY_POSITIVITY_RELU,
        "use_norm": True,
        "eval_scales": EVAL_SCALES,
        "learning_rate": LEARNING_RATE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "sched_patience": SCHED_PATIENCE,
    }


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
    ax.set_title(f"{run_config['name']} — MSE only")
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
            "learning_rate": run_config["learning_rate"],
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "grad_clip_norm": run_config["grad_clip_norm"],
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": SCHED_FACTOR,
                "patience": run_config["sched_patience"],
            },
            "noise_std": NOISE_STD,
            "use_amp": USE_AMP,
            "snapshot_index": SNAPSHOT_INDEX,
        },
        "loss": {"type": "mse"},
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


def _find_existing_run_dir(run_name: str) -> Path | None:
    """Search existing ``comparing_best_models_mse_*`` folders for a trained run."""
    candidates = sorted(OUTPUT_ROOT.glob(f"{FOLDER_NAME}*/{run_name}/weights.pt"))
    return candidates[-1].parent if candidates else None


def _already_trained(output_dir: Path, run_name: str) -> bool:
    if (output_dir / run_name / "weights.pt").exists():
        return True
    return _find_existing_run_dir(run_name) is not None


def _resolve_weights_path(output_dir: Path, run_name: str) -> Path:
    current = output_dir / run_name / "weights.pt"
    if current.exists():
        return current
    existing = _find_existing_run_dir(run_name)
    if existing is not None:
        return existing / "weights.pt"
    return current


def train_one_run(
    run_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
) -> float:
    name = run_config["name"]
    lr = run_config["learning_rate"]
    grad_clip = run_config["grad_clip_norm"]
    sched_patience = run_config["sched_patience"]

    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"  loss = MSE only")
    print(
        f"  skip={run_config['skip_connection']}  "
        f"interp={run_config['interpolation_mode']}"
    )
    print(f"  lr={lr}  grad_clip={grad_clip}  sched_patience={sched_patience}")
    print(f"{'=' * 60}")

    model = _build_model(run_config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=sched_patience
    )

    loss_fn = nn.MSELoss()
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
                    loss = loss_fn(output, hr) / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                output = _forward(model, lr)
                loss = loss_fn(output, hr) / GRAD_ACCUM
                loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                if USE_AMP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                loss = loss_fn(output, hr)
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
            f"best_val={best_val_loss:.6f} | "
            f"no_improve={epochs_no_improve}/{EARLY_STOP_PATIENCE} | "
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


def _write_manifest(output_dir: Path, run_config: dict) -> dict:
    params = {
        **CFNO_PARAMS,
        "shifting_modes": run_config["shifting_modes"],
        "interpolation_mode": run_config["interpolation_mode"],
        "skip_connection": run_config["skip_connection"],
        "skip_connection_interpolation_mode": run_config[
            "skip_connection_interpolation_mode"
        ],
    }
    model_entry = {
        "name": run_config["name"],
        "label": run_config["label"],
        "model_type": "cfno",
        "model_params": params,
        "weights": str(_resolve_weights_path(output_dir, run_config["name"])),
        "apply_positivity_relu": run_config["apply_positivity_relu"],
        "use_norm": run_config["use_norm"],
        "loss": "mse",
        "skip_connection": run_config["skip_connection"],
        "interpolation_mode": run_config["interpolation_mode"],
        "skip_connection_interpolation_mode": run_config[
            "skip_connection_interpolation_mode"
        ],
        "supports_variable_scale": True,
        "eval_scales": run_config["eval_scales"],
        "n_epochs": EPOCHS,
        "learning_rate": run_config["learning_rate"],
        "grad_clip_norm": run_config["grad_clip_norm"],
        "sched_patience": run_config["sched_patience"],
    }

    manifest = {
        "experiment": "comparing_best_models_mse",
        "output_dir": str(output_dir),
        "benchmark_csv": "experiments/comparing_best_models_mse/benchmark_metrics.csv",
        "benchmark_row_key": "run",
        "benchmark_row_value_field": "name",
        "normalization_stats": str(NORM_STATS_PATH),
        "upsample_factor": UPSAMPLE_FACTOR,
        "models": [model_entry],
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
                print(f"  skip {label}: OOM — skipping")
            else:
                print(f"  skip {label}: RuntimeError — {e}")
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
    print(f"Optimizer        : AdamW(wd={WEIGHT_DECAY}) + ReduceLROnPlateau")
    print(f"Noise std        : {NOISE_STD}")
    eff_bs = BATCH_SIZE * GRAD_ACCUM
    print(f"Batch size       : {BATCH_SIZE} (accum {GRAD_ACCUM} -> eff {eff_bs})")
    print(f"AMP              : {USE_AMP}")

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

    # ── run ───────────────────────────────────────────────────────────
    run_config = _build_run()
    print(
        f"\nRun: {run_config['name']}  lr={run_config['learning_rate']}  "
        f"clip={run_config['grad_clip_norm']}  "
        f"sched_p={run_config['sched_patience']}"
    )

    result = {"status": "skipped"}
    if _already_trained(output_dir, run_config["name"]):
        print(f"  Already trained: {run_config['name']} — skipping training.")
    else:
        try:
            best_val = train_one_run(run_config, train_loader, val_loader, output_dir)
            result = {"status": "success", "best_val_loss": best_val}
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            result = {"status": "failed", "error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    # ── training summary ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    if result["status"] == "success":
        print(f"  {run_config['name']:55s}  val_loss = {result['best_val_loss']:.6f}")
    else:
        print(f"  {run_config['name']:55s}  {result['status'].upper()}")

    summary = {
        "output_dir": str(output_dir),
        "runs": {run_config["name"]: result},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(
        [
            {
                "run_name": run_config["name"],
                "status": result["status"],
                "best_val_loss": result.get("best_val_loss"),
            }
        ]
    ).to_csv(output_dir / "training_summary.csv", index=False)

    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    # ── write manifest ────────────────────────────────────────────────
    manifest = _write_manifest(output_dir, run_config)
    print(f"\nManifest written to {output_dir / 'manifest.json'}")

    _run_eval_and_plots(manifest, output_dir)
    print(f"\nAll artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()

