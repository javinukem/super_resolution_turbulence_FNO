"""Unified experiment trainer for turbulence_sr.

One engine replaces the per-experiment ``train_*.py`` scripts: the shared
training loop (AdamW + ReduceLROnPlateau, gradient accumulation, Gaussian
input noise, early stopping, optional grad clipping / AMP), artifact saving
(``weights.pt``, ``losses.csv``, ``loss_curve.png``, ``config.json``), the
run-group ``manifest.json``, and the benchmark evaluation stage
(``benchmark_metrics.csv`` via ``evaluation.benchmark``).

Each experiment is a declarative YAML config (``experiments/<name>/experiment.yaml``)
describing the run grid; experiment folders keep only the config plus plots
and aggregated results. Bar charts and final-snapshot plots stay in separate
scripts invoked from ``run.sh`` (per AGENTS.md conventions).

Config schema
-------------
.. code-block:: yaml

    experiment: <name>
    output:
      scratch_root: /export/scratch/jalegria/experiments
      folder_prefix: <name>_            # timestamped run-group dirs
      repo_dir: experiments/<name>      # benchmark_metrics.csv copy target
    data:
      train_h5: /abs/full_states.h5
      val_h5: /abs/full_states_val.h5
      snapshot_index: 79
      upsample_factor: 4
      num_workers: 2
      eval_batch_size: 2
    normalization:
      stats_path: experiments/l1_spectral_weighting/normalization_stats.npz
    spectral_pool: 2                     # avg-pool before FFT in SpectralLoss
    trilinear_baseline: false            # append a trilinear manifest entry
    training:                            # experiment defaults, run-overridable
      epochs: 500
      learning_rate: 0.001
      weight_decay: 1.0e-4
      batch_size: 8
      grad_accum: 8
      noise_std: 0.01
      early_stop_patience: 40
      sched_factor: 0.5
      sched_patience: 15
      grad_clip_norm: null               # null = no clipping
      use_amp: false
    runs:                                # the trained run grid
      - name: <run_name>
        label: <chart label>
        model_type: cfno                 # cfno | edsr | ufno
        model_params: {...}              # ctor kwargs (not apply_positivity_relu)
        apply_positivity_relu: false
        use_norm: true
        eval_scales: [4, 2]
        loss: {type: mse_spectral, spectral_weight: 0.00636}
        training: {...}                  # optional per-run overrides
        manifest_extra: {...}            # extra fields merged into the entry
    references:                          # manifest-only entries (not trained)
      - adopt_from:
          scratch_glob: l1_spectral_weighting_*
          select: {weight_name: w_minor}
          patch_model_params: {skip_connection: false}
        rename: cfno_shift8_skip_off_ref
        label: "skip=False (w_minor ref)"

Loss types: ``mse``, ``l1``, ``spectral`` (standalone), ``mse_spectral``,
``mse_l1``, ``mse_spectral_l1``, ``l1_spectral`` — each optionally with
``spectral_weight`` / ``l1_weight``.

Usage
-----
    python -m src.training.experiment experiments/<name>/experiment.yaml
    python -m src.training.experiment experiments/<name>/experiment.yaml \
        --run <run_name>            # train a single run from the grid
    python -m src.training.experiment --manifest <run-group dir>   # eval-only
"""

if __name__ == "__main__":
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
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader.dataloader_3d import dataset_sr
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
    _instantiate,
    build_model,
    discover_latest,
    load_manifest,
    load_norm_stats,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Velocity indices (don't hardcode — AGENTS.md)
# =====================================================================


@lru_cache(maxsize=1)
def get_velocity_indices() -> tuple[int, int, int]:
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

    def __init__(self, vx_idx: int, vy_idx: int, vz_idx: int, pool: int = 2):
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


class _ScalarLoss(nn.Module):
    """Wrap a plain reduction='mean' loss so forward returns (loss, metrics)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = self.base(pred, target)
        return loss, {"loss": loss.item()}


class MSESpectralLoss(nn.Module):
    """MSE + weighted velocity spectral loss."""

    def __init__(self, spectral_weight: float, vx: int, vy: int, vz: int,
                 pool: int = 2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.spectral = SpectralLoss(vx, vy, vz, pool)
        self.spectral_weight = spectral_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = self.mse(pred, target)
        spec, _ = self.spectral(pred, target)
        total = mse + self.spectral_weight * spec
        return total, {
            "mse": mse.item(), "spectral": spec.item(), "total": total.item(),
        }


class L1SpectralLoss(nn.Module):
    """L1 + weighted velocity spectral loss."""

    def __init__(self, spectral_weight: float, vx: int, vy: int, vz: int, pool: int = 2):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.spectral = SpectralLoss(vx, vy, vz, pool)
        self.spectral_weight = spectral_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = self.l1(pred, target)
        spec, _ = self.spectral(pred, target)
        total = l1 + self.spectral_weight * spec
        return total, {"l1": l1.item(), "spectral": spec.item(), "total": total.item()}


class MSEL1Loss(nn.Module):
    """MSE + weighted L1."""

    def __init__(self, l1_weight: float):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = self.mse(pred, target)
        l1 = self.l1(pred, target)
        total = mse + self.l1_weight * l1
        return total, {"mse": mse.item(), "l1": l1.item(), "total": total.item()}


class MSESpectralL1Loss(nn.Module):
    """MSE + weighted spectral + weighted L1."""

    def __init__(self, spectral_weight: float, l1_weight: float,
                 vx: int, vy: int, vz: int, pool: int = 2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.spectral = SpectralLoss(vx, vy, vz, pool)
        self.l1 = nn.L1Loss()
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = self.mse(pred, target)
        spec, _ = self.spectral(pred, target)
        l1 = self.l1(pred, target)
        total = mse + self.spectral_weight * spec + self.l1_weight * l1
        return total, {
            "mse": mse.item(),
            "spectral": spec.item(),
            "l1": l1.item(),
            "total": total.item(),
        }


def build_loss_fn(loss_cfg: dict, vx: int, vy: int, vz: int,
                  pool: int = 2) -> nn.Module:
    """Return the loss module for a run's ``loss`` config block."""
    loss_cfg = dict(loss_cfg or {"type": "mse"})
    t = loss_cfg["type"]
    sw = loss_cfg.get("spectral_weight")
    lw = loss_cfg.get("l1_weight")
    if t == "mse":
        return _ScalarLoss(nn.MSELoss())
    if t == "l1":
        return _ScalarLoss(nn.L1Loss())
    if t == "spectral":
        return SpectralLoss(vx, vy, vz, pool)
    if t == "mse_spectral":
        return MSESpectralLoss(sw, vx, vy, vz, pool)
    if t == "l1_spectral":
        return L1SpectralLoss(sw, vx, vy, vz, pool)
    if t == "mse_l1":
        return MSEL1Loss(lw)
    if t == "mse_spectral_l1":
        return MSESpectralL1Loss(sw, lw, vx, vy, vz, pool)
    raise ValueError(f"Unknown loss type: {t}")


def _loss_tag(loss_cfg: dict) -> str:
    """Short manifest/config.json tag for a loss config."""
    t = (loss_cfg or {}).get("type", "mse")
    return {
        "mse": "mse",
        "l1": "l1",
        "spectral": "spectral",
        "mse_spectral": "mse_spectral",
        "l1_spectral": "l1_spectral",
        "mse_l1": "mse+l1",
        "mse_spectral_l1": "mse+spectral+l1",
    }[t]


# =====================================================================
# Model / forward
# =====================================================================


def build_run_model(run: dict) -> nn.Module:
    """Instantiate a run's model (cfno/edsr/ufno) from its manifest-shaped fields."""
    return _instantiate(
        {
            "model_type": run["model_type"],
            "model_params": run.get("model_params") or {},
            "apply_positivity_relu": run.get("apply_positivity_relu", False),
        }
    )


def _forward(model: nn.Module, lr: torch.Tensor, upsample_factor: int) -> torch.Tensor:
    """cfno/ufno take upsample_factor as a forward arg; EDSR is fixed-scale."""
    from src.model.models_edsr import EDSR

    if isinstance(model, EDSR):
        return model(lr)
    return model(lr, upsample_factor=upsample_factor)


# =====================================================================
# Artifacts
# =====================================================================


def save_run_artifacts(
    run_dir: Path,
    run: dict,
    train_cfg: dict,
    best_state: dict,
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: float,
    config: dict,
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
    ax.set_title(run["name"])
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    loss_cfg = dict(run.get("loss") or {"type": "mse"})
    summary = {
        "run": {
            "name": run["name"],
            "label": run["label"],
            **{
                k: v
                for k, v in run.items()
                if k not in {"name", "label", "model_type", "model_params",
                             "apply_positivity_relu", "use_norm", "eval_scales",
                             "loss", "training", "manifest_extra"}
            },
        },
        "model_params": run.get("model_params") or {},
        "apply_positivity_relu": run.get("apply_positivity_relu", False),
        "training": {
            "optimizer": "AdamW",
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "epochs": train_cfg["epochs"],
            "batch_size": train_cfg["batch_size"],
            "grad_accum": train_cfg["grad_accum"],
            "effective_batch_size": train_cfg["batch_size"] * train_cfg["grad_accum"],
            "early_stop_patience": train_cfg["early_stop_patience"],
            "grad_clip_norm": train_cfg.get("grad_clip_norm"),
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": train_cfg["sched_factor"],
                "patience": train_cfg["sched_patience"],
            },
            "noise_std": train_cfg["noise_std"],
            "use_amp": train_cfg.get("use_amp", False),
            "snapshot_index": config["data"]["snapshot_index"],
        },
        "loss": {
            "type": _loss_tag(loss_cfg),
            **{k: v for k, v in loss_cfg.items() if k != "type"},
            "spectral_pool": config.get("spectral_pool", 2),
        },
        "normalization": {
            "enabled": bool(run.get("use_norm", False)),
            "stats_path": str((ROOT / config["normalization"]["stats_path"]).resolve())
            if config.get("normalization", {}).get("stats_path")
            else None,
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
# Resume helpers
# =====================================================================


def find_existing_run_dir(
    scratch_root: Path, folder_prefix: str, run_name: str
) -> Path | None:
    """Search existing run-group folders for a trained run.

    Each invocation writes to a fresh timestamped ``output_dir``, so a run
    trained previously lives under a different folder. This globs the scratch
    root for any matching ``<folder>/<run_name>/weights.pt``.
    """
    candidates = sorted(scratch_root.glob(f"{folder_prefix}*/{run_name}/weights.pt"))
    return candidates[-1].parent if candidates else None


def already_trained(output_dir: Path, config: dict, run_name: str) -> bool:
    if (output_dir / run_name / "weights.pt").exists():
        return True
    return (
        find_existing_run_dir(
            Path(config["output"]["scratch_root"]),
            config["output"]["folder_prefix"],
            run_name,
        )
        is not None
    )


def resolve_weights_path(output_dir: Path, config: dict, run_name: str) -> Path:
    current = output_dir / run_name / "weights.pt"
    if current.exists():
        return current
    existing = find_existing_run_dir(
        Path(config["output"]["scratch_root"]),
        config["output"]["folder_prefix"],
        run_name,
    )
    if existing is not None:
        return existing / "weights.pt"
    return current


# =====================================================================
# Training loop
# =====================================================================


def train_one_run(
    run: dict,
    train_cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    config: dict,
) -> float:
    name = run["name"]
    upsample_factor = config["data"]["upsample_factor"]

    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"  loss: {run.get('loss') or {'type': 'mse'}}")
    print(f"  lr={train_cfg['learning_rate']}  clip={train_cfg.get('grad_clip_norm')}"
          f"  sched_patience={train_cfg['sched_patience']}")
    print(f"{'=' * 60}")

    model = build_run_model(run).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=train_cfg["sched_factor"],
        patience=train_cfg["sched_patience"],
    )

    vx, vy, vz = get_velocity_indices()
    loss_fn = build_loss_fn(
        run.get("loss"),
        vx, vy, vz,
        config.get("spectral_pool", 2),
    )
    use_amp = bool(train_cfg.get("use_amp", False))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_accum = int(train_cfg["grad_accum"])
    noise_std = train_cfg["noise_std"]
    grad_clip = train_cfg.get("grad_clip_norm")
    early_stop_patience = int(train_cfg["early_stop_patience"])
    epochs = int(train_cfg["epochs"])

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            hr = batch[0].to(DEVICE, non_blocking=True)
            lr = batch[1].to(DEVICE, non_blocking=True)
            lr = lr + torch.randn_like(lr) * noise_std

            if use_amp:
                with torch.amp.autocast("cuda"):
                    output = _forward(model, lr, upsample_factor)
                    loss, _ = loss_fn(output, hr)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            else:
                output = _forward(model, lr, upsample_factor)
                loss, _ = loss_fn(output, hr)
                loss = loss / grad_accum
                loss.backward()

            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip is not None:
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
                output = _forward(model, lr, upsample_factor)
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
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"train={epoch_loss:.6f}  val={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | "
            f"no_improve={epochs_no_improve}/{early_stop_patience} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if epochs_no_improve >= early_stop_patience:
            print(f"  Early stopping triggered at epoch {epoch + 1}.")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    run_dir = output_dir / name
    save_run_artifacts(
        run_dir, run, train_cfg, best_state,
        train_losses, val_losses, best_val_loss, config,
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


def _adopt_reference(ref_cfg: dict, config: dict) -> dict | None:
    """Adopt a manifest entry from a previous run group as a reference."""
    adopt = ref_cfg["adopt_from"]
    try:
        ref_dir = discover_latest(
            Path(config["output"]["scratch_root"]),
            adopt["scratch_glob"],
        )
    except FileNotFoundError as e:
        print(f"  Warning: reference manifest not found ({e}) — skipping")
        return None
    ref_manifest = load_manifest(ref_dir)
    select = adopt.get("select") or {}
    for e in ref_manifest["models"]:
        if all(e.get(k) == v for k, v in select.items()):
            entry = dict(e)
            params = dict(entry.get("model_params") or {})
            params.update(adopt.get("patch_model_params") or {})
            entry["model_params"] = params
            if "rename" in ref_cfg:
                entry["name"] = ref_cfg["rename"]
            if "label" in ref_cfg:
                entry["label"] = ref_cfg["label"]
            entry.update(ref_cfg.get("manifest_extra") or {})
            return entry
    print(f"  Warning: no reference entry matched {select} — skipping")
    return None


def write_manifest(output_dir: Path, config: dict, runs: list[dict]) -> dict:
    models = []
    for run in runs:
        train_cfg = _merged_train_cfg(config, run)
        entry = {
            "name": run["name"],
            "label": run["label"],
            "model_type": run["model_type"],
            "model_params": run.get("model_params") or {},
            "weights": str(resolve_weights_path(output_dir, config, run["name"])),
            "apply_positivity_relu": run.get("apply_positivity_relu", False),
            "use_norm": run.get("use_norm", False),
            "loss": _loss_tag(run.get("loss")),
            "supports_variable_scale": run["model_type"] != "edsr",
            "eval_scales": run.get("eval_scales", [config["data"]["upsample_factor"]]),
            "n_epochs": train_cfg["epochs"],
            "learning_rate": train_cfg["learning_rate"],
            "grad_clip_norm": train_cfg.get("grad_clip_norm"),
            "sched_patience": train_cfg["sched_patience"],
        }
        loss_cfg = run.get("loss") or {}
        if loss_cfg.get("spectral_weight") is not None:
            entry["spectral_weight"] = loss_cfg["spectral_weight"]
        if loss_cfg.get("l1_weight") is not None:
            entry["l1_weight"] = loss_cfg["l1_weight"]
        if loss_cfg.get("weight_name"):
            entry["weight_name"] = loss_cfg["weight_name"]
        entry.update(run.get("manifest_extra") or {})
        models.append(entry)

    for ref_cfg in config.get("references") or []:
        if "adopt_from" in ref_cfg:
            ref = _adopt_reference(ref_cfg, config)
            if ref is not None:
                models.append(ref)
        else:
            models.append(dict(ref_cfg))

    if config.get("trilinear_baseline"):
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

    stats_rel = config.get("normalization", {}).get("stats_path")
    manifest = {
        "experiment": config["experiment"],
        "output_dir": str(output_dir),
        "benchmark_csv": f"{config['output']['repo_dir']}/benchmark_metrics.csv",
        "benchmark_row_key": "run",
        "benchmark_row_value_field": "name",
        "normalization_stats": str((ROOT / stats_rel).resolve()) if stats_rel else None,
        "upsample_factor": config["data"]["upsample_factor"],
        "models": models,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def _merged_train_cfg(config: dict, run: dict) -> dict:
    """Experiment-level training defaults merged with a run's overrides."""
    cfg = dict(config["training"])
    cfg.update(run.get("training") or {})
    return cfg


# =====================================================================
# Evaluation
# =====================================================================


def build_metrics() -> list:
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


def evaluate_all(
    manifest: dict, val_loader_raw: DataLoader, metrics: list, norm_stats
) -> list[dict]:
    rows = []
    for entry in manifest["models"]:
        label = entry["label"]
        run_label = entry["name"]
        try:
            run = build_model(entry, DEVICE, norm_stats)
            for scale in entry.get("eval_scales", [manifest.get("upsample_factor", 4)]):
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


def run_eval_and_plots(manifest: dict, output_dir: Path, config: dict) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    data_cfg = config["data"]
    print("Loading raw validation dataset (for benchmark) …")
    val_ds_raw = dataset_sr(
        h5_path=data_cfg["val_h5"],
        snapshot_index=data_cfg["snapshot_index"],
    )
    print(f"  Raw validation samples: {len(val_ds_raw)}")
    val_loader_raw = DataLoader(
        val_ds_raw,
        batch_size=data_cfg.get("eval_batch_size", 2),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
    )

    metrics = build_metrics()
    norm_stats = load_norm_stats(manifest, DEVICE)

    eval_rows = evaluate_all(manifest, val_loader_raw, metrics, norm_stats)
    eval_df = pd.DataFrame(eval_rows)
    preferred = [
        "run", "model", "upsample_factor", "MSE",
        "Loss_pressure", "Loss_density", "Loss_vx", "Loss_vy", "Loss_vz",
        "Loss_v_norm", "Loss_vorticity", "Spectral_MSE", "Perceptual",
        "PSNR", "SSIM", "time_s",
    ]
    eval_df = eval_df[[c for c in preferred if c in eval_df.columns]]
    scratch_csv = output_dir / "benchmark_metrics.csv"
    eval_df.to_csv(scratch_csv, index=False)
    repo_csv = ROOT / config["output"]["repo_dir"] / "benchmark_metrics.csv"
    repo_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(scratch_csv, repo_csv)
    print(f"  Saved {scratch_csv}")
    print(f"  Copied to {repo_csv}")

    del val_loader_raw, val_ds_raw
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# Main
# =====================================================================


def _load_config(path: Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(path)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        help="path to the experiment YAML (e.g. experiments/<name>/experiment.yaml)",
    )
    parser.add_argument(
        "--run",
        help="train only this run from the config's run grid (name)",
    )
    parser.add_argument(
        "--manifest",
        help="eval-only mode: run-group folder (containing manifest.json) + its config",
    )
    args = parser.parse_args()

    if args.manifest:
        config_path = Path(args.manifest)
        if config_path.is_dir():
            config_path = config_path / "manifest.json"
        manifest = load_manifest(config_path.parent)
        # locate the experiment config for data/eval settings
        cfg_path = ROOT / "experiments" / manifest["experiment"] / "experiment.yaml"
        config = _load_config(cfg_path)
        output_dir = Path(manifest["output_dir"])
        print(f"Eval-only mode. output_dir={output_dir}")
        run_eval_and_plots(manifest, output_dir, config)
        return

    if not args.config:
        parser.error("a config YAML is required (or --manifest for eval-only)")
    config = _load_config(Path(args.config))
    data_cfg = config["data"]

    rv = _get_registered_variables_3d()
    print(
        f"  velocity_index: x={rv.velocity_index.x} "
        f"y={rv.velocity_index.y} z={rv.velocity_index.z}"
    )

    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    output_dir = (
        Path(config["output"]["scratch_root"])
        / f"{config['output']['folder_prefix']}{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tcfg = config["training"]
    print(f"Output directory : {output_dir}")
    print(f"Device           : {DEVICE}")
    print(f"Train data       : {data_cfg['train_h5']}")
    print(f"Val data         : {data_cfg['val_h5']}")
    print(f"Snapshot index   : {data_cfg['snapshot_index']}")
    print(f"Epochs           : {tcfg['epochs']} "
          f"(patience {tcfg['early_stop_patience']})")
    print(f"Optimizer        : AdamW(wd={tcfg['weight_decay']}) + ReduceLROnPlateau")
    print(f"Noise std        : {tcfg['noise_std']}")
    print(
        f"Batch size       : {tcfg['batch_size']} (accum {tcfg['grad_accum']} "
        f"-> eff {tcfg['batch_size'] * tcfg['grad_accum']})"
    )
    print(f"AMP              : {tcfg.get('use_amp', False)}")

    norm_cfg = config.get("normalization") or {}
    use_norm = bool(norm_cfg.get("stats_path"))
    norm_stats_path = ROOT / norm_cfg["stats_path"] if use_norm else None

    print("\nLoading training dataset …")
    train_ds = dataset_sr(
        h5_path=data_cfg["train_h5"],
        snapshot_index=data_cfg["snapshot_index"],
        use_normalizing=use_norm,
        mean_std_path=norm_stats_path if use_norm else None,
    )
    print(f"  Training samples: {len(train_ds)}")

    print("Loading validation dataset …")
    val_ds = dataset_sr(
        h5_path=data_cfg["val_h5"],
        snapshot_index=data_cfg["snapshot_index"],
        use_normalizing=use_norm,
        means=(train_ds.mean_hr, train_ds.mean_lr) if use_norm else None,
        stds=(train_ds.std_hr, train_ds.std_lr) if use_norm else None,
    )
    print(f"  Validation samples: {len(val_ds)}")

    num_workers = data_cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    runs = config["runs"]
    if args.run:
        runs = [r for r in runs if r["name"] == args.run]
        if not runs:
            names = [r["name"] for r in config["runs"]]
            sys.exit(f"run '{args.run}' not in config (have {names})")
    print(f"\nRun grid: {len(runs)} runs")
    for r in runs:
        rt = _merged_train_cfg(config, r)
        print(
            f"  - {r['name']}  lr={rt['learning_rate']}  "
            f"clip={rt.get('grad_clip_norm')}  sched_p={rt['sched_patience']}"
        )

    results = {}
    for run_config in runs:
        if already_trained(output_dir, config, run_config["name"]):
            print(f"  Already trained: {run_config['name']} — skipping training.")
            results[run_config["name"]] = {"status": "skipped"}
            continue
        try:
            best_val = train_one_run(
                run_config,
                _merged_train_cfg(config, run_config),
                train_loader,
                val_loader,
                output_dir,
                config,
            )
            results[run_config["name"]] = {"status": "success", "best_val_loss": best_val}
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[run_config["name"]] = {"status": "failed", "error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        if result["status"] == "success":
            print(f"  {name:55s}  val_loss = {result['best_val_loss']:.6f}")
        else:
            print(f"  {name:55s}  {result['status'].upper()}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump({"output_dir": str(output_dir), "runs": results}, f, indent=2, default=str)

    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    manifest = write_manifest(output_dir, config, config["runs"])
    print(f"\nManifest written to {output_dir / 'manifest.json'}")

    run_eval_and_plots(manifest, output_dir, config)
    print(f"\nAll artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
