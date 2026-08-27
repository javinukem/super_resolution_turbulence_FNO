"""
Benchmark for the ``trying_new_losses_and_norm`` experiment.

Evaluates every model listed in the experiment manifest (12 trained runs:
2 models × 3 losses × 2 norm settings, plus a trilinear baseline) on the
validation dataset (snapshot 79, x4), computing the same metric suite as
``evaluation/benchmark.py``: MSE, per-channel MSE, velocity-norm, vorticity,
spectral (JAX/Pk_library), perceptual, PSNR, SSIM.

Norm-on runs are handled by ``evaluation.manifest.build_model``: the raw LR
input is normalized with the cached training statistics, the model runs, and
the SR output is denormalized before metrics are computed — so every run's
metrics are directly comparable on the raw physical scale.

Results are saved to the ``benchmark_csv`` path declared in the manifest
(default: ``experiments/trying_new_losses_and_norm/benchmark_results.csv``).

Usage
-----
    python experiments/trying_new_losses_and_norm/benchmark.py
    python experiments/trying_new_losses_and_norm/benchmark.py \
        --manifest /path/to/manifest.json
"""

from autocvd import autocvd

autocvd(num_gpus=1)
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader.dataloader_3d import dataset_sr
from evaluation.manifest import (
    build_model,
    discover_latest,
    load_manifest,
    load_norm_stats,
)
from evaluation.benchmark import (
    MSEMetric,
    PSNRMetric,
    ChannelMSEMetric,
    VelocityNormMSEMetric,
    VorticityMSEMetric,
    SSIMMetric,
    SpectralMSEMetric,
    PerceptualLoss3D,
    _get_registered_variables_3d,
)

# ── Paths & constants ─────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
SNAPSHOT_INDEX = 79
BATCH_SIZE = 2
NUM_WORKERS = 2
UPSAMPLE_FACTOR = 4

SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
DEFAULT_MANIFEST_GLOB = "loss_norm_experiment_*"


# =====================================================================
# Evaluation
# =====================================================================


def evaluate_run(
    run,
    val_loader: DataLoader,
    metrics: list,
) -> dict:
    """Evaluate one run on the validation set at x4."""
    accum = {m.name: 0.0 for m in metrics}
    n_batches = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in val_loader:
            hr = batch[0].to(DEVICE)
            lr = batch[1].to(DEVICE)
            sr = run(lr, UPSAMPLE_FACTOR)
            for m in metrics:
                accum[m.name] += m(sr, hr)
            n_batches += 1
            del hr, lr, sr

    elapsed = time.time() - t0
    return {m.name: accum[m.name] / n_batches for m in metrics} | {
        "time_s": elapsed,
        "upsample_factor": UPSAMPLE_FACTOR,
    }


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
        Path(args.manifest)
        if args.manifest
        else discover_latest(SCRATCH_BASE, DEFAULT_MANIFEST_GLOB)
    )
    manifest = load_manifest(manifest_dir)
    results_csv = manifest["benchmark_csv"]
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Manifest : {manifest['_manifest_path']}")
    print(f"Experiment: {manifest.get('experiment')}")
    print(f"Results  : {results_csv}")

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
        PerceptualLoss3D(DEVICE),
        PSNRMetric(),
        SSIMMetric(),
    ]

    norm_stats = load_norm_stats(manifest, DEVICE)

    results = []
    for entry in manifest["models"]:
        name = entry["name"]
        try:
            print(f"\nLoading {name} …")
            run = build_model(entry, DEVICE, norm_stats)
            res = evaluate_run(run, val_loader, metrics)
            res["run"] = name
            res["model"] = entry.get("model_name")
            res["loss"] = entry.get("loss")
            res["use_norm"] = entry.get("use_norm")
            results.append(res)
            print(f"  ✓ {name} MSE={res['MSE']:.5f}")
        except Exception as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                print(f"  ✗ {name}: OOM — skipping")
            else:
                print(f"  ✗ {name}: FAILED — {e}")
                traceback.print_exc()
        finally:
            if "run" in locals():
                del run.model
            gc.collect()
            torch.cuda.empty_cache()

    # ── report ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    cols = [
        "run",
        "model",
        "loss",
        "use_norm",
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
    df = df[[c for c in cols if c in df.columns]]

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS — trying_new_losses_and_norm")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")


if __name__ == "__main__":
    main()
