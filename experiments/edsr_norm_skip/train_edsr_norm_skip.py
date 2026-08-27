"""
Train EDSR with normalization ON, both skip-off and skip-on.

The EDSR trained by ``experiments/training_best_models_experiment/train_best_models.py``
uses non-normalized data, ``nn.MSELoss()``, and no skip connection. This
experiment keeps the EDSR architecture's hyper-parameters
(``n_resblocks=16, n_feats=64, kernel_size=3, scale=4``) but switches to the
norm-on training convention used by ``l1_spectral_weighting``,
``loss_ablation``, and ``trying_new_losses_and_norm``:

    - ``dataset_sr(use_normalizing=True)`` reusing the shared
      ``l1_spectral_weighting/normalization_stats.npz`` (snapshot 79)
    - ``EPOCHS=250`` + ``ReduceLROnPlateau(factor=0.5, patience=15)`` +
      ``EARLY_STOP_PATIENCE=40`` (the schedule that fixed EDSR undertraining,
      per ``trying_new_losses_and_norm``)
    - ``apply_positivity_relu=False`` so the final ReLU doesn't clip ~half of
      the zero-mean normalized density/pressure signal
    - ``activation_f="prelu"`` to restore the inter-stage nonlinearity between
      the two PixelShuffle stages (the documented fix for EDSR blockiness)

The only varying axis across the two runs is ``skip_connection`` (a new flag
on ``src.model.models_edsr.EDSR`` that mirrors ``FNO_2``'s skip: the LR
latent is upsampled and added to the HR latent before the channel projection):

    1. ``edsr_norm_skip_off`` — ``skip_connection=False`` (norm-on baseline)
    2. ``edsr_norm_skip_on``  — ``skip_connection=True`` (trilinear interp)

After training, the script evaluates both models with ``evaluation.benchmark``
metrics and writes ``benchmark_metrics.csv`` (scratch + repo copy). Bar charts
and the final-snapshot comparison are produced by separate scripts invoked by
``run.sh`` (no duplicated plotting code in this file).

Usage
-----
    bash experiments/edsr_norm_skip/run.sh
    python experiments/edsr_norm_skip/train_edsr_norm_skip.py
    python experiments/edsr_norm_skip/train_edsr_norm_skip.py \
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
from src.model.models_edsr import EDSR

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
FOLDER_NAME = "edsr_norm_skip_"
REPO_EXP_DIR = ROOT / "experiments" / "edsr_norm_skip"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the l1_spectral_weighting normalization stats (do NOT regenerate) so
# results are directly comparable to the other norm-on experiments.
L1_SPECTRAL_DIR = ROOT / "experiments" / "l1_spectral_weighting"
NORM_STATS_PATH = L1_SPECTRAL_DIR / "normalization_stats.npz"

# ── Data ──────────────────────────────────────────────────────────────

SNAPSHOT_INDEX = 79
UPSAMPLE_FACTOR = 4

# ── Training hyper-parameters (mirror norm-on convention) ─────────────

EPOCHS = 250
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NOISE_STD = 0.01
NUM_WORKERS = 2
USE_AMP = False
GRAD_ACCUM = 8
EARLY_STOP_PATIENCE = 40
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 3
GRAD_CLIP_NORM = 1.0

# ── EDSR hyper-parameters (n_resblocks/n_feats/kernel_size/scale from
#    train_best_models.py; activation_f = "prelu" per trying_new_losses_and_norm) ──

EDSR_PARAMS = dict(
    input_channels=5,
    n_resblocks=16,
    n_feats=64,
    kernel_size=3,
    scale=4,
    activation_f="prelu",
)
APPLY_POSITIVITY_RELU = False
SKIP_CONNECTION_INTERPOLATION_MODE = "trilinear"


# =====================================================================
# Run grid
# =====================================================================


def _build_run_grid() -> list[dict]:
    """Two trained runs: identical except for the skip connection."""
    base = {
        "model_type": "edsr",
        "apply_positivity_relu": APPLY_POSITIVITY_RELU,
        "use_norm": True,
        "eval_scales": [UPSAMPLE_FACTOR],
    }
    return [
        {
            **base,
            "name": "edsr_norm_skip_off",
            "label": "EDSR norm (skip off)",
            "skip_connection": False,
        },
        {
            **base,
            "name": "edsr_norm_skip_on",
            "label": "EDSR norm (skip on)",
            "skip_connection": True,
        },
    ]


def _build_model(run_config: dict) -> nn.Module:
    """Instantiate EDSR with the run's skip settings."""
    return EDSR(
        **EDSR_PARAMS,
        skip_connection=run_config["skip_connection"],
        skip_connection_interpolation_mode=SKIP_CONNECTION_INTERPOLATION_MODE,
        apply_positivity_relu=run_config["apply_positivity_relu"],
    )


def _forward(model: nn.Module, lr: torch.Tensor) -> torch.Tensor:
    # EDSR is fixed-scale (scale=4 baked in) — no upsample_factor arg.
    return model(lr)


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
    skip_str = "on" if run_config["skip_connection"] else "off"
    ax.set_title(f"{run_config['name']} — skip={skip_str}")
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
        },
        "model_params": {
            **EDSR_PARAMS,
            "skip_connection": run_config["skip_connection"],
            "skip_connection_interpolation_mode": SKIP_CONNECTION_INTERPOLATION_MODE,
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
            "grad_clip_norm": GRAD_CLIP_NORM,
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


def _already_trained(output_dir: Path, run_name: str) -> bool:
    return (output_dir / run_name / "weights.pt").exists()


def train_one_run(
    run_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
) -> float:
    name = run_config["name"]
    skip_str = "on" if run_config["skip_connection"] else "off"
    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"  skip={skip_str}")
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
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
    """Manifest with ONLY the two trained EDSR models (no trilinear row)."""
    models = []
    for run in runs:
        params = {
            **EDSR_PARAMS,
            "skip_connection": run["skip_connection"],
            "skip_connection_interpolation_mode": SKIP_CONNECTION_INTERPOLATION_MODE,
        }
        models.append(
            {
                "name": run["name"],
                "label": run["label"],
                "model_type": "edsr",
                "model_params": params,
                "weights": str(output_dir / run["name"] / "weights.pt"),
                "apply_positivity_relu": run["apply_positivity_relu"],
                "use_norm": run["use_norm"],
                "loss": "mse",
                "skip_connection": run["skip_connection"],
                "supports_variable_scale": False,
                "eval_scales": run["eval_scales"],
            }
        )

    manifest = {
        "experiment": "edsr_norm_skip",
        "output_dir": str(output_dir),
        "benchmark_csv": "experiments/edsr_norm_skip/benchmark_metrics.csv",
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
    runs = _build_run_grid()
    print(f"\nRun grid: {len(runs)} trained runs")
    for r in runs:
        print(
            f"  - {r['name']}  skip={r['skip_connection']}  "
            f"norm={r['use_norm']}"
        )

    # ── train ───────────────────────────────────────────────────────
    results = {}
    for run_config in runs:
        if _already_trained(output_dir, run_config["name"]):
            print(f"  Already trained: {run_config['name']} — skipping training.")
            results[run_config["name"]] = {
                "status": "skipped",
                "skip_connection": run_config["skip_connection"],
            }
            continue
        try:
            best_val = train_one_run(run_config, train_loader, val_loader, output_dir)
            results[run_config["name"]] = {
                "status": "success",
                "best_val_loss": best_val,
                "skip_connection": run_config["skip_connection"],
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[run_config["name"]] = {
                "status": "failed",
                "error": str(e),
                "skip_connection": run_config["skip_connection"],
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
                "skip_connection": r.get("skip_connection"),
            }
            for n, r in results.items()
        ]
    ).to_csv(output_dir / "training_summary.csv", index=False)

    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    # ── write manifest (only the 2 trained EDSR models) ──────────────
    manifest = _write_manifest(output_dir, runs)
    print(f"\nManifest written to {output_dir / 'manifest.json'}")

    _run_eval_and_plots(manifest, output_dir)
    print(f"\nAll artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()