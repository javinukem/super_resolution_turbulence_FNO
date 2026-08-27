"""
Grid experiment for FNO_2 variants on turbulence SR.

What this script does (single-file workflow):
1) Trains a grid of FNO_2 models on the same train/val datasets and base
   hyper-parameters as `train_best_models.py`, changing only the requested axes:
   - modes in {16, 24, 32}
   - interpolation_mode in {"nearest", "trilinear"}
   - skip_connection in {False, True}
   - head variant in {"mlp_tail", "conv_refine_tail"}
   - loss combo in {"mse", "mse_vorticity", "mse_spectral"}
   - epochs fixed to 200
2) Evaluates every trained run using benchmark-equivalent metrics
   (`evaluation/benchmark.py` metric classes and evaluate loop).
3) Saves per-run artifacts and aggregate train/eval summaries.
"""

# from autocvd import autocvd
#
# autocvd(num_gpus=1)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import gc
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

# Benchmark-compatible metrics/eval pipeline
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


# ── Paths ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states.h5")
VAL_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
OUTPUT_ROOT = Path("/export/scratch/jalegria/experiments")
FOLDER_NAME = "fno2_grid_modes_interp_skip_losses_refine_"

# ── Data ──────────────────────────────────────────────────────────────
SNAPSHOT_INDEX = 79
UPSAMPLE_FACTOR = 4

# ── Training hyper-parameters (same as train_best_models.py except EPOCHS) ──
EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2
NOISE_STD = 0.01
NUM_WORKERS = 2
USE_AMP = False
GRAD_ACCUM = 30
EARLY_STOP_PATIENCE = 30

# ── Grid axes ─────────────────────────────────────────────────────────
SHIFTING_MODES = 8
GRID_MODES = [16, 24, 32]
GRID_INTERPOLATION = ["nearest", "trilinear"]
GRID_SKIP_CONNECTION = [False, True]
GRID_HEAD_VARIANT = ["mlp_tail", "conv_refine_tail"]
GRID_LOSS_COMBO = ["mse", "mse_vorticity", "mse_spectral"]

# ── Loss weights (small aux terms) ────────────────────────────────────
VORTICITY_WEIGHT = 0.02
SPECTRAL_WEIGHT = 0.01
SPECTRAL_POOL = 2

# ── Baseline FNO_2 params from train_best_models.py ───────────────────
FNO2_BASE_PARAMS = dict(
    in_channel=5,
    n_channels=32,
    n_residual_blocks=3,
    n_operator_blocks=2,
    apply_constraint=False,
    last_layer_kernel=3,
)


def _interpolate_3d(x: torch.Tensor, scale_factor: int, mode: str) -> torch.Tensor:
    if mode in {"nearest", "area", "nearest-exact"}:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)


@dataclass(frozen=True)
class RunConfig:
    modes: int
    interpolation_mode: str
    skip_connection: bool
    head_variant: str
    loss_combo: str

    @property
    def name(self) -> str:
        return (
            f"fno2_m{self.modes}_shift{SHIFTING_MODES}_"
            f"interp-{self.interpolation_mode}_"
            f"skip-{int(self.skip_connection)}_"
            f"head-{self.head_variant}_"
            f"loss-{self.loss_combo}"
        )


class FNO2GridModel(nn.Module):
    """
    Experiment wrapper around FNO_2 with:
    - robust interpolation handling for nearest/trilinear
    - explicit skip connection on upsampled latent
    - tail variants: baseline conv ("mlp_tail") vs Conv3d→GELU→Conv3d refinement
    """

    def __init__(
        self,
        *,
        modes: int,
        interpolation_mode: str,
        skip_connection: bool,
        head_variant: str,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.interpolation_mode = interpolation_mode

        self.core = FNO_2(
            **FNO2_BASE_PARAMS,
            modes=modes,
            shifting_modes=SHIFTING_MODES,
            interpolation_mode=interpolation_mode,
            skip_connection=False,  # handled explicitly in this wrapper
        )

        if head_variant == "conv_refine_tail":
            self.core.tail = nn.Sequential(
                nn.Conv3d(
                    self.core.n_channels, self.core.n_channels, kernel_size=3, padding=1
                ),
                nn.GELU(),
                nn.Conv3d(
                    self.core.n_channels, self.core.in_channel, kernel_size=3, padding=1
                ),
            )

        elif head_variant != "mlp_tail":
            raise ValueError(f"Unsupported head variant: {head_variant}")

    def forward(
        self, x: torch.Tensor, upsample_factor: int = UPSAMPLE_FACTOR
    ) -> torch.Tensor:
        x1 = self.core.conv1(x)
        x2 = x1
        for layer in self.core.res_blocks:
            x2 = layer(x2)
        latent = x1 + x2

        upsampled = _interpolate_3d(
            latent, scale_factor=upsample_factor, mode=self.interpolation_mode
        )

        out = upsampled
        for layer in self.core.fno_blocks:
            out = layer(out)

        if self.skip_connection:
            out = out + upsampled

        out = self.core.tail(out)

        if self.core.apply_constraint:
            out = self.core.constraint(x, out, upsample_factor)

        out[:, 0] = self.core.tail_relu(out[:, 0])
        out[:, 4] = self.core.tail_relu(out[:, 4])
        return out


class WeightedPhysicsLoss(nn.Module):
    """
    MSE + optional physics-informed terms.

    API:
      - combo="mse": only MSE
      - combo="mse_vorticity": MSE + w_vort * L_vorticity
      - combo="mse_spectral": MSE + w_spec * L_spectral
    """

    def __init__(
        self,
        *,
        combo: str,
        vx_idx: int,
        vy_idx: int,
        vz_idx: int,
        w_vorticity: float,
        w_spectral: float,
        spectral_pool: int = 2,
    ):
        super().__init__()
        self.combo = combo
        self.vx_idx = vx_idx
        self.vy_idx = vy_idx
        self.vz_idx = vz_idx
        self.w_vorticity = float(w_vorticity)
        self.w_spectral = float(w_spectral)
        self.spectral_pool = int(spectral_pool)
        if combo not in {"mse", "mse_vorticity", "mse_spectral"}:
            raise ValueError(f"Unsupported loss combo: {combo}")

    @staticmethod
    def _vorticity_magnitude(
        vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor
    ) -> torch.Tensor:
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

    def _vorticity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vx_pred = pred[:, self.vx_idx]
        vy_pred = pred[:, self.vy_idx]
        vz_pred = pred[:, self.vz_idx]
        vx_target = target[:, self.vx_idx]
        vy_target = target[:, self.vy_idx]
        vz_target = target[:, self.vz_idx]

        vort_pred = self._vorticity_magnitude(vx_pred, vy_pred, vz_pred)
        vort_target = self._vorticity_magnitude(vx_target, vy_target, vz_target)
        return F.mse_loss(vort_pred, vort_target)

    def _spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_v = pred[:, [self.vx_idx, self.vy_idx, self.vz_idx]]
        target_v = target[:, [self.vx_idx, self.vy_idx, self.vz_idx]]

        if self.spectral_pool > 1:
            pred_v = F.avg_pool3d(pred_v, kernel_size=self.spectral_pool)
            target_v = F.avg_pool3d(target_v, kernel_size=self.spectral_pool)

        pred_fft = torch.fft.rfftn(pred_v, dim=(-3, -2, -1), norm="ortho")
        target_fft = torch.fft.rfftn(target_v, dim=(-3, -2, -1), norm="ortho")

        pred_power = pred_fft.real.square() + pred_fft.imag.square()
        target_power = target_fft.real.square() + target_fft.imag.square()

        pred_power = pred_power.mean(dim=1)
        target_power = target_power.mean(dim=1)

        pred_log = torch.log10(pred_power + 1e-12)
        target_log = torch.log10(target_power + 1e-12)
        return F.mse_loss(pred_log, target_log)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mse = F.mse_loss(pred, target)
        vort = torch.zeros((), device=pred.device, dtype=pred.dtype)
        spec = torch.zeros((), device=pred.device, dtype=pred.dtype)
        total = mse

        if self.combo == "mse_vorticity":
            vort = self._vorticity_loss(pred, target)
            total = total + self.w_vorticity * vort
        elif self.combo == "mse_spectral":
            spec = self._spectral_loss(pred, target)
            total = total + self.w_spectral * spec

        return total, {"mse": mse, "vorticity": vort, "spectral": spec}


def _build_loss_fn(loss_combo: str, rv: Any) -> WeightedPhysicsLoss:
    return WeightedPhysicsLoss(
        combo=loss_combo,
        vx_idx=rv.velocity_index.x,
        vy_idx=rv.velocity_index.y,
        vz_idx=rv.velocity_index.z,
        w_vorticity=VORTICITY_WEIGHT,
        w_spectral=SPECTRAL_WEIGHT,
        spectral_pool=SPECTRAL_POOL,
    )


def _save_artifacts(
    run_dir: Path,
    run_config: RunConfig,
    best_state: dict[str, torch.Tensor],
    history: dict[str, list[float]],
    best_val_loss: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "weights.pt")

    pd.DataFrame(history).to_csv(run_dir / "losses.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["train_total"], label="Train total")
    ax.plot(history["val_total"], label="Validation total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(run_config.name)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    summary = {
        "run_config": {
            "modes": run_config.modes,
            "shifting_modes": SHIFTING_MODES,
            "interpolation_mode": run_config.interpolation_mode,
            "skip_connection": run_config.skip_connection,
            "head_variant": run_config.head_variant,
            "loss_combo": run_config.loss_combo,
        },
        "model_params_base": FNO2_BASE_PARAMS,
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "noise_std": NOISE_STD,
            "use_amp": USE_AMP,
            "snapshot_index": SNAPSHOT_INDEX,
            "upsample_factor": UPSAMPLE_FACTOR,
        },
        "loss_weights": {
            "vorticity_weight": VORTICITY_WEIGHT,
            "spectral_weight": SPECTRAL_WEIGHT,
            "spectral_pool": SPECTRAL_POOL,
        },
        "results": {
            "best_val_loss": best_val_loss,
            "final_train_total": history["train_total"][-1]
            if history["train_total"]
            else None,
            "final_val_total": history["val_total"][-1]
            if history["val_total"]
            else None,
        },
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _build_metrics() -> list:
    rv = _get_registered_variables_3d()
    vx_idx, vy_idx, vz_idx = (
        rv.velocity_index.x,
        rv.velocity_index.y,
        rv.velocity_index.z,
    )
    return [
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


def _sanity_check_loss_api(rv: Any) -> None:
    x = torch.randn(1, 5, 16, 16, 16, device=DEVICE, dtype=torch.float32)
    y = torch.randn(1, 5, 16, 16, 16, device=DEVICE, dtype=torch.float32)
    for combo in GRID_LOSS_COMBO:
        loss_fn = _build_loss_fn(combo, rv).to(DEVICE)
        total, comps = loss_fn(x, y)
        if not torch.isfinite(total):
            raise RuntimeError(
                f"Loss sanity check failed for combo={combo}: total is not finite"
            )
        for k, v in comps.items():
            if not torch.isfinite(v):
                raise RuntimeError(
                    f"Loss sanity check failed for combo={combo}, component={k}"
                )


def train_one_model(
    run_config: RunConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    metrics: list,
    rv: Any,
) -> dict[str, Any]:
    print(f"\n{'=' * 90}")
    print(f"Training run: {run_config.name}")
    print(f"{'=' * 90}")

    model = FNO2GridModel(
        modes=run_config.modes,
        interpolation_mode=run_config.interpolation_mode,
        skip_connection=run_config.skip_connection,
        head_variant=run_config.head_variant,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = _build_loss_fn(run_config.loss_combo, rv).to(DEVICE)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_total": [],
        "train_mse": [],
        "train_vorticity": [],
        "train_spectral": [],
        "val_total": [],
        "val_mse": [],
        "val_vorticity": [],
        "val_spectral": [],
    }

    best_val_loss = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        train_acc = {"total": 0.0, "mse": 0.0, "vorticity": 0.0, "spectral": 0.0}
        for step, batch in enumerate(train_loader):
            hr = batch[0].to(DEVICE, non_blocking=True)
            lr = batch[1].to(DEVICE, non_blocking=True)
            lr = lr + torch.randn_like(lr) * NOISE_STD

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    output = model(lr, upsample_factor=UPSAMPLE_FACTOR)
                    total_loss, comps = loss_fn(output, hr)
                    loss = total_loss / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                output = model(lr, upsample_factor=UPSAMPLE_FACTOR)
                total_loss, comps = loss_fn(output, hr)
                loss = total_loss / GRAD_ACCUM
                loss.backward()

            train_acc["total"] += float(total_loss.detach().item())
            train_acc["mse"] += float(comps["mse"].detach().item())
            train_acc["vorticity"] += float(comps["vorticity"].detach().item())
            train_acc["spectral"] += float(comps["spectral"].detach().item())

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        n_train = len(train_loader)
        for k in train_acc:
            train_acc[k] /= n_train

        model.eval()
        val_acc = {"total": 0.0, "mse": 0.0, "vorticity": 0.0, "spectral": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                hr = batch[0].to(DEVICE, non_blocking=True)
                lr = batch[1].to(DEVICE, non_blocking=True)
                output = model(lr, upsample_factor=UPSAMPLE_FACTOR)
                total_loss, comps = loss_fn(output, hr)
                val_acc["total"] += float(total_loss.item())
                val_acc["mse"] += float(comps["mse"].item())
                val_acc["vorticity"] += float(comps["vorticity"].item())
                val_acc["spectral"] += float(comps["spectral"].item())

        n_val = len(val_loader)
        for k in val_acc:
            val_acc[k] /= n_val

        history["epoch"].append(epoch + 1)
        history["train_total"].append(train_acc["total"])
        history["train_mse"].append(train_acc["mse"])
        history["train_vorticity"].append(train_acc["vorticity"])
        history["train_spectral"].append(train_acc["spectral"])
        history["val_total"].append(val_acc["total"])
        history["val_mse"].append(val_acc["mse"])
        history["val_vorticity"].append(val_acc["vorticity"])
        history["val_spectral"].append(val_acc["spectral"])

        if val_acc["total"] < best_val_loss:
            best_val_loss = val_acc["total"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"train_total={train_acc['total']:.6f} val_total={val_acc['total']:.6f} | "
            f"best_val={best_val_loss:.6f} | no_improve={epochs_no_improve}/{EARLY_STOP_PATIENCE} | "
            f"{elapsed:.1f}s"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    run_dir = output_dir / run_config.name
    _save_artifacts(run_dir, run_config, best_state, history, best_val_loss)
    print(f"Saved artifacts to {run_dir}")

    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    eval_results = []
    for scale in [4, 2]:
        res = benchmark_evaluate_model(
            model=model,
            val_loader=val_loader,
            metrics=metrics,
            label=run_config.name,
            upsample_factor=scale,
            supports_variable_scale=True,
        )
        eval_results.append(res)

    del model, best_state, optimizer, scaler, loss_fn
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "run_name": run_config.name,
        "status": "success",
        "best_val_loss": best_val_loss,
        "eval_results": eval_results,
    }


def _build_grid() -> list[RunConfig]:
    grid: list[RunConfig] = []
    for modes in GRID_MODES:
        for interpolation_mode in GRID_INTERPOLATION:
            for skip_connection in GRID_SKIP_CONNECTION:
                for head_variant in GRID_HEAD_VARIANT:
                    for loss_combo in GRID_LOSS_COMBO:
                        grid.append(
                            RunConfig(
                                modes=modes,
                                interpolation_mode=interpolation_mode,
                                skip_connection=skip_connection,
                                head_variant=head_variant,
                                loss_combo=loss_combo,
                            )
                        )
    return grid


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUT_ROOT / f"{FOLDER_NAME}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Train data: {TRAIN_H5}")
    print(f"Val data: {VAL_H5}")
    print(f"Epochs: {EPOCHS}")
    print(f"Grid size: {len(_build_grid())}")

    rv = _get_registered_variables_3d()
    _sanity_check_loss_api(rv)
    metrics = _build_metrics()

    print("Loading training dataset …")
    train_ds = dataset_sr(h5_path=TRAIN_H5, snapshot_index=SNAPSHOT_INDEX)
    print(f"Training samples: {len(train_ds)}")

    print("Loading validation dataset …")
    val_ds = dataset_sr(h5_path=VAL_H5, snapshot_index=SNAPSHOT_INDEX)
    print(f"Validation samples: {len(val_ds)}")

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

    all_run_summaries: list[dict[str, Any]] = []
    all_eval_rows: list[dict[str, Any]] = []

    for run_config in _build_grid():
        try:
            summary = train_one_model(
                run_config=run_config,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_dir,
                metrics=metrics,
                rv=rv,
            )
            all_run_summaries.append(
                {
                    "run_name": summary["run_name"],
                    "status": summary["status"],
                    "best_val_loss": summary["best_val_loss"],
                }
            )
            all_eval_rows.extend(summary["eval_results"])
        except Exception as exc:
            print(f"FAILED run {run_config.name}: {exc}")
            traceback.print_exc()
            all_run_summaries.append(
                {
                    "run_name": run_config.name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_run_summaries)
    summary_df.to_csv(output_dir / "training_summary.csv", index=False)

    eval_df = pd.DataFrame(all_eval_rows)
    eval_cols = [
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
    eval_df = eval_df[[c for c in eval_cols if c in eval_df.columns]]
    eval_df.to_csv(output_dir / "benchmark_metrics.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(output_dir),
                "grid_size": len(_build_grid()),
                "successful_runs": int((summary_df["status"] == "success").sum()),
                "failed_runs": int((summary_df["status"] == "failed").sum()),
            },
            f,
            indent=2,
        )

    print(f"\nAll artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
