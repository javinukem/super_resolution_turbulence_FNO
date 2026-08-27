"""
Train the best CFNO (shift=8) and EDSR with different losses and normalization.

Motivation
----------
The models trained by ``training_best_models_experiment/train_best_models.py``
produce blurry (CFNO) and blocky (EDSR) outputs. Analysis attributes the blur to
``nn.MSELoss`` (posterior-mean regression) plus an unbalanced loss where density
dominates by ~1-2 orders of magnitude, and EDSR's blockiness to an undertrained
upsampler with ``activation_f=False`` (no inter-stage nonlinearity between the
two PixelShuffle stages).

This experiment retrains the two best models (CFNO shift=8 — best x4 MSE on the
benchmark; EDSR) with three losses and two normalization settings:

    losses   : mse_l1 (MSE + L1), mse_spectral (MSE + velocity spectral), l1
    norm     : off (raw) / on (dataloader-level per-channel standardization)

EDSR additionally switches to ``activation_f="prelu"`` (was ``False``) to restore
the inter-stage nonlinearity in the upsampler.

Normalization is applied at the dataloader level
(``dataset_sr(use_normalizing=True)``), so the trained norm-on models are
zero-mean/unit-std in/out. To avoid the final density/pressure ReLU clipping
~half of the normalized signal, both models expose a new
``apply_positivity_relu`` flag (default ``True`` = unchanged behaviour); it is
set to ``False`` for norm-on runs only. Per-channel mean/std are computed once
from the training set and cached to ``normalization_stats.npz`` at the output
root, so re-runs and the val set reuse them without recomputation.

Training schedule is lengthened to give the undertrained EDSR room to converge:
``EPOCHS=250``, ``EARLY_STOP_PATIENCE=40``, plus a
``ReduceLROnPlateau(factor=0.5, patience=15)`` scheduler.

Usage
-----
    python experiments/trying_new_losses_and_norm/train_loss_norm.py
"""

from autocvd import autocvd

autocvd(num_gpus=1)
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader.dataloader_3d import dataset_sr
from src.model.models_edsr import EDSR
from src.model.models_fno_2 import FNO_2

# ── Paths ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states.h5")
VAL_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
OUTPUT_ROOT = Path("/export/scratch/jalegria/experiments")
FOLDER_NAME = "loss_norm_experiment_"

# ── Data ──────────────────────────────────────────────────────────────

SNAPSHOT_INDEX = 79  # last snapshot per simulation (stationary turbulence)
UPSAMPLE_FACTOR = 4

# ── Training hyper-parameters ─────────────────────────────────────────

EPOCHS = 250
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NOISE_STD = 0.01
NUM_WORKERS = 2
USE_AMP = False
GRAD_ACCUM = 8  # effective batch size = BATCH_SIZE × GRAD_ACCUM = 64
EARLY_STOP_PATIENCE = 40
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 15

# ── Best CFNO hyper-parameters (shift=8, best x4 MSE on benchmark) ────

CFNO_PARAMS = dict(
    in_channel=5,
    n_channels=32,
    n_residual_blocks=3,
    n_operator_blocks=2,
    modes=16,
    apply_constraint=False,
    last_layer_kernel=3,
)

# ── EDSR hyper-parameters (activation_f switched to "prelu") ──────────

EDSR_PARAMS = dict(
    input_channels=5,
    n_resblocks=16,
    n_feats=64,
    kernel_size=3,
    scale=4,
    activation_f="prelu",
)

# ── Loss weights ──────────────────────────────────────────────────────

L1_WEIGHT = 1.0
SPECTRAL_WEIGHT = 0.01
SPECTRAL_POOL = 2

# ── Model catalogue (the two best models) ─────────────────────────────

MODELS = [
    {"name": "cfno_shift8", "type": "cfno", "shifting_modes": 8},
    {"name": "edsr", "type": "edsr"},
]

LOSSES = ["mse_l1", "mse_spectral", "l1"]
NORM_OPTIONS = [False, True]


# =====================================================================
# Variable indices (don't hardcode — AGENTS.md)
# =====================================================================


@lru_cache(maxsize=1)
def _get_velocity_indices() -> tuple[int, int, int]:
    """Return (vx, vy, vz) channel indices via jf1uids registered variables."""
    from jf1uids import SimulationConfig, get_registered_variables
    from jf1uids.option_classes.simulation_config import finalize_config

    cfg = finalize_config(SimulationConfig(dimensionality=3), (5, 128, 128, 128))
    rv = get_registered_variables(cfg)
    return rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z


# =====================================================================
# Losses
# =====================================================================


class MSEL1Loss(nn.Module):
    """MSE + weighted L1."""

    def __init__(self, l1_weight: float = L1_WEIGHT):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = self.mse(pred, target)
        l1 = self.l1(pred, target)
        total = mse + self.l1_weight * l1
        return total, {"mse": mse.item(), "l1": l1.item(), "total": total.item()}


class SpectralLoss(nn.Module):
    """
    Velocity-only torch-FFT log-power spectral loss.

    Adapted from
    ``experiments/train_fno2_grid_modes_interp_skip_losses_refine/...py:270-289``:
    rfftn on the 3 velocity channels, power = |F|^2, mean over channels,
    log10, MSE. Optional avg-pool before the FFT to reduce memory.
    """

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

    def __init__(self, vx_idx: int, vy_idx: int, vz_idx: int):
        super().__init__()
        self.mse = nn.MSELoss()
        self.spectral = SpectralLoss(vx_idx, vy_idx, vz_idx)
        self.spectral_weight = SPECTRAL_WEIGHT

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = self.mse(pred, target)
        spec, _ = self.spectral(pred, target)
        total = mse + self.spectral_weight * spec
        return total, {
            "mse": mse.item(),
            "spectral": spec.item(),
            "total": total.item(),
        }


class L1Loss(nn.Module):
    """Pure L1."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = self.l1(pred, target)
        return l1, {"l1": l1.item()}


def _build_loss(loss_name: str) -> nn.Module:
    vx, vy, vz = _get_velocity_indices()
    if loss_name == "mse_l1":
        return MSEL1Loss()
    if loss_name == "mse_spectral":
        return MSESpectralLoss(vx, vy, vz)
    if loss_name == "l1":
        return L1Loss()
    raise ValueError(f"Unknown loss: {loss_name}")


# =====================================================================
# Helpers
# =====================================================================


def _build_run_grid() -> list[dict]:
    runs = []
    for model_config in MODELS:
        for loss_name in LOSSES:
            for use_norm in NORM_OPTIONS:
                tag = f"{model_config['name']}_{loss_name}_{'norm' if use_norm else 'nonorm'}"
                runs.append(
                    {
                        "name": tag,
                        "model": model_config,
                        "loss": loss_name,
                        "use_norm": use_norm,
                    }
                )
    return runs


def _model_params_for(model_config: dict) -> dict:
    """Return the ctor kwargs recorded for a model in the manifest."""
    if model_config["type"] == "cfno":
        return {**CFNO_PARAMS, "shifting_modes": model_config["shifting_modes"]}
    return dict(EDSR_PARAMS)


def _write_manifest(output_dir: Path, norm_stats_path: Path) -> None:
    """Aggregate the run grid into manifest.json for downstream scripts."""
    runs = _build_run_grid()
    models = []
    for run_config in runs:
        model_config = run_config["model"]
        mtype = model_config["type"]
        use_norm = run_config["use_norm"]
        norm_tag = "norm" if use_norm else "nonorm"
        label = (
            f"cfno_{run_config['loss']} ({norm_tag})"
            if mtype == "cfno"
            else f"{run_config['loss']} ({norm_tag})"
        )
        models.append(
            {
                "name": run_config["name"],
                "label": label,
                "model_type": mtype,
                "model_name": model_config["name"],
                "model_params": _model_params_for(model_config),
                "weights": str(output_dir / run_config["name"] / "weights.pt"),
                "apply_positivity_relu": not use_norm,
                "use_norm": use_norm,
                "loss": run_config["loss"],
                "supports_variable_scale": mtype == "cfno",
                "eval_scales": [4],
            }
        )
    models.append(
        {
            "name": "trilinear_baseline",
            "label": "Trilinear",
            "model_type": "trilinear",
            "model_name": "trilinear",
            "model_params": {},
            "weights": None,
            "apply_positivity_relu": False,
            "use_norm": False,
            "loss": "none",
            "supports_variable_scale": True,
            "eval_scales": [4],
        }
    )
    manifest = {
        "experiment": "trying_new_losses_and_norm",
        "output_dir": str(output_dir),
        "benchmark_csv": "experiments/trying_new_losses_and_norm/benchmark_results.csv",
        "benchmark_row_key": "run",
        "benchmark_row_value_field": "name",
        "normalization_stats": str(norm_stats_path),
        "upsample_factor": UPSAMPLE_FACTOR,
        "models": models,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _build_model(model_config: dict, use_norm: bool) -> nn.Module:
    """Instantiate a model; disable the positivity ReLU for norm-on runs."""
    apply_relu = not use_norm
    if model_config["type"] == "cfno":
        return FNO_2(
            **CFNO_PARAMS,
            shifting_modes=model_config["shifting_modes"],
            apply_positivity_relu=apply_relu,
        )
    if model_config["type"] == "edsr":
        return EDSR(**EDSR_PARAMS, apply_positivity_relu=apply_relu)
    raise ValueError(f"Unknown model type: {model_config['type']}")


def _forward(model: nn.Module, lr: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type == "cfno":
        return model(lr, upsample_factor=UPSAMPLE_FACTOR)
    return model(lr)


def _save_artifacts(
    run_dir: Path,
    run_config: dict,
    best_state: dict,
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: float,
    norm_stats_path: Path | None,
) -> None:
    """Persist weights, losses CSV, loss plot, and config JSON."""
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
        f"{run_config['name']} — {run_config['loss']} | norm={run_config['use_norm']}"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    summary = {
        "run": run_config,
        "model_params": (
            {**CFNO_PARAMS, "shifting_modes": run_config["model"]["shifting_modes"]}
            if run_config["model"]["type"] == "cfno"
            else EDSR_PARAMS
        ),
        "apply_positivity_relu": not run_config["use_norm"],
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
        "loss_weights": {
            "l1_weight": L1_WEIGHT,
            "spectral_weight": SPECTRAL_WEIGHT,
            "spectral_pool": SPECTRAL_POOL,
        },
        "normalization": {
            "enabled": run_config["use_norm"],
            "stats_path": str(norm_stats_path) if norm_stats_path else None,
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


def train_one_run(
    run_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    norm_stats_path: Path | None,
) -> float:
    """Train a single (model, loss, norm) configuration."""
    name = run_config["name"]
    mtype = run_config["model"]["type"]
    loss_name = run_config["loss"]
    use_norm = run_config["use_norm"]

    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"  model={mtype}  loss={loss_name}  norm={use_norm}")
    print(f"{'=' * 60}")

    model = _build_model(run_config["model"], use_norm).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
    )
    loss_fn = _build_loss(loss_name)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            hr = batch[0].to(DEVICE, non_blocking=True)
            lr = batch[1].to(DEVICE, non_blocking=True)

            lr = lr + torch.randn_like(lr) * NOISE_STD

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    output = _forward(model, lr, mtype)
                    loss, _ = loss_fn(output, hr)
                    loss = loss / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                output = _forward(model, lr, mtype)
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

        # ── validate ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                hr = batch[0].to(DEVICE, non_blocking=True)
                lr = batch[1].to(DEVICE, non_blocking=True)
                if USE_AMP:
                    with torch.amp.autocast("cuda"):
                        output = _forward(model, lr, mtype)
                        loss, _ = loss_fn(output, hr)
                else:
                    output = _forward(model, lr, mtype)
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

    # ── persist ───────────────────────────────────────────────────
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    run_dir = output_dir / name
    _save_artifacts(
        run_dir,
        run_config,
        best_state,
        train_losses,
        val_losses,
        best_val_loss,
        norm_stats_path,
    )
    print(f"  Saved to {run_dir}")
    print(f"  Best val loss: {best_val_loss:.6f}")

    del model, best_state, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_loss


# =====================================================================
# Main
# =====================================================================


def main():
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    output_dir = OUTPUT_ROOT / f"{FOLDER_NAME}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {output_dir}")
    print(f"Device           : {DEVICE}")
    print(f"Train data       : {TRAIN_H5}")
    print(f"Val data         : {VAL_H5}")
    print(f"Snapshot index   : {SNAPSHOT_INDEX}")
    print(f"Epochs           : {EPOCHS} (patience {EARLY_STOP_PATIENCE})")
    print(f"Optimizer        : AdamW(lr={LEARNING_RATE}, wd={WEIGHT_DECAY}) + ReduceLROnPlateau")
    print(f"Noise std        : {NOISE_STD}")
    print(f"Batch size       : {BATCH_SIZE} (accum {GRAD_ACCUM} -> eff {BATCH_SIZE * GRAD_ACCUM})")
    print(f"AMP              : {USE_AMP}")

    # ── datasets ──────────────────────────────────────────────────
    # Normalization stats are computed once from the TRAIN set and cached to
    # a single npz; the val set reuses the same (train) stats.
    norm_stats_path = output_dir / "normalization_stats.npz"

    print("\nLoading training dataset (norm-on) …")
    train_ds_norm = dataset_sr(
        h5_path=TRAIN_H5,
        snapshot_index=SNAPSHOT_INDEX,
        use_normalizing=True,
        mean_std_path=norm_stats_path,
    )
    print(f"  Training samples (norm): {len(train_ds_norm)}")

    print("Loading validation dataset (norm-on, reusing train stats) …")
    val_ds_norm = dataset_sr(
        h5_path=VAL_H5,
        snapshot_index=SNAPSHOT_INDEX,
        use_normalizing=True,
        means=(train_ds_norm.mean_hr, train_ds_norm.mean_lr),
        stds=(train_ds_norm.std_hr, train_ds_norm.std_lr),
    )
    print(f"  Validation samples (norm): {len(val_ds_norm)}")

    print("Loading training dataset (raw) …")
    train_ds_raw = dataset_sr(h5_path=TRAIN_H5, snapshot_index=SNAPSHOT_INDEX)
    print(f"  Training samples (raw): {len(train_ds_raw)}")

    print("Loading validation dataset (raw) …")
    val_ds_raw = dataset_sr(h5_path=VAL_H5, snapshot_index=SNAPSHOT_INDEX)
    print(f"  Validation samples (raw): {len(val_ds_raw)}")

    def _make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

    loaders = {
        True: (_make_loader(train_ds_norm, True), _make_loader(val_ds_norm, False)),
        False: (_make_loader(train_ds_raw, True), _make_loader(val_ds_raw, False)),
    }

    # ── run grid ──────────────────────────────────────────────────
    runs = _build_run_grid()
    print(f"\nRun grid: {len(runs)} runs")
    for r in runs:
        print(f"  - {r['name']}")

    results = {}
    for run_config in runs:
        use_norm = run_config["use_norm"]
        train_loader, val_loader = loaders[use_norm]
        try:
            best_val = train_one_run(
                run_config,
                train_loader,
                val_loader,
                output_dir,
                norm_stats_path if use_norm else None,
            )
            results[run_config["name"]] = {
                "status": "success",
                "best_val_loss": best_val,
                "loss": run_config["loss"],
                "use_norm": use_norm,
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[run_config["name"]] = {
                "status": "failed",
                "error": str(e),
                "loss": run_config["loss"],
                "use_norm": use_norm,
            }
            gc.collect()
            torch.cuda.empty_cache()

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        if result["status"] == "success":
            print(f"  {name:40s}  val_loss = {result['best_val_loss']:.6f}")
        else:
            print(f"  {name:40s}  FAILED — {result['error']}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _write_manifest(output_dir, norm_stats_path)
    print(f"  Manifest written to {output_dir / 'manifest.json'}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
