# GPU selection (must happen before CUDA init)
from autocvd import autocvd

autocvd(num_gpus=1)
import gc
import json
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
from src.model.models_fno_2 import FNO_2

# ── Paths ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states.h5")
VAL_H5 = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
OUTPUT_ROOT = Path("/export/scratch/jalegria/experiments")

FOLDER_NAME = "best_models_full_data_"
# ── Data ──────────────────────────────────────────────────────────────

SNAPSHOT_INDEX = None  # last snapshot per simulation only
UPSAMPLE_FACTOR = 4

# ── Training hyper-parameters ─────────────────────────────────────────

EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NOISE_STD = 0.01
NUM_WORKERS = 2
USE_AMP = False
GRAD_ACCUM = 8  # effective batch size = BATCH_SIZE × GRAD_ACCUM = 16
EARLY_STOP_PATIENCE = 30

# ── Best CFNO hyper-parameters (from experiment_2 grid search) ────────

CFNO_PARAMS = dict(
    in_channel=5,
    n_channels=32,
    n_residual_blocks=3,
    n_operator_blocks=2,
    modes=16,
    apply_constraint=False,
    last_layer_kernel=3,
)

# ── Best EDSR hyper-parameters (from experiment_1 grid search) ────────

EDSR_PARAMS = dict(
    input_channels=5,
    n_resblocks=16,
    n_feats=64,
    kernel_size=3,
    scale=4,
    activation_f=False,
)

# ── Model catalogue ──────────────────────────────────────────────────

MODELS = [
    {"name": "cfno_shift0", "type": "cfno", "shifting_modes": 0},
    {"name": "cfno_shift4", "type": "cfno", "shifting_modes": 4},
    {"name": "cfno_shift8", "type": "cfno", "shifting_modes": 8},
    {"name": "cfno_shift12", "type": "cfno", "shifting_modes": 12},
    {"name": "cfno_shift16", "type": "cfno", "shifting_modes": 16},
    {"name": "edsr", "type": "edsr"},
]


# =====================================================================
# Helpers
# =====================================================================


def _build_model(config: dict) -> nn.Module:
    """Instantiate a model from its catalogue entry."""
    if config["type"] == "cfno":
        return FNO_2(**CFNO_PARAMS, shifting_modes=config["shifting_modes"])
    elif config["type"] == "edsr":
        return EDSR(**EDSR_PARAMS)
    raise ValueError(f"Unknown model type: {config['type']}")


def _model_params_for(model_config: dict) -> dict:
    """Return the ctor kwargs recorded for a model in the manifest."""
    if model_config["type"] == "cfno":
        return {**CFNO_PARAMS, "shifting_modes": model_config["shifting_modes"]}
    return dict(EDSR_PARAMS)


def _write_manifest(output_dir: Path) -> None:
    """Aggregate the model catalogue into manifest.json for downstream scripts."""
    models = []
    for cfg in MODELS:
        supports_variable_scale = cfg["type"] == "cfno"
        models.append(
            {
                "name": cfg["name"],
                "label": (
                    f"CFNO (shift={cfg['shifting_modes']})"
                    if cfg["type"] == "cfno"
                    else "EDSR (resb16_f64_k3)"
                ),
                "model_type": cfg["type"],
                "model_params": _model_params_for(cfg),
                "weights": str(output_dir / cfg["name"] / "weights.pt"),
                "apply_positivity_relu": True,
                "use_norm": False,
                "loss": None,
                "supports_variable_scale": supports_variable_scale,
                "eval_scales": [4, 2] if supports_variable_scale else [4],
            }
        )
    models.append(
        {
            "name": "trilinear",
            "label": "Bicubic (trilinear)",
            "model_type": "trilinear",
            "model_params": {},
            "weights": None,
            "apply_positivity_relu": False,
            "use_norm": False,
            "loss": None,
            "supports_variable_scale": True,
            "eval_scales": [4, 2],
        }
    )
    manifest = {
        "experiment": "training_best_models",
        "output_dir": str(output_dir),
        "benchmark_csv": (
            "experiments/training_best_models_experiment/"
            "trained_on_last_snapshot/benchmark_results.csv"
        ),
        "benchmark_row_key": "model",
        "benchmark_row_value_field": "label",
        "normalization_stats": None,
        "upsample_factor": UPSAMPLE_FACTOR,
        "models": models,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _forward(model: nn.Module, lr: torch.Tensor, model_type: str) -> torch.Tensor:
    """Run the model forward; CFNO needs ``upsample_factor``."""
    if model_type == "cfno":
        return model(lr, upsample_factor=UPSAMPLE_FACTOR)
    return model(lr)


def _save_artifacts(
    run_dir: Path,
    model_config: dict,
    best_state: dict,
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: float,
) -> None:
    """Persist weights, losses CSV, loss plot, and config JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    torch.save(best_state, run_dir / "weights.pt")

    # Losses CSV
    df = pd.DataFrame(
        {
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    df.to_csv(run_dir / "losses.csv", index=False)

    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(
        f"{model_config['name']} — Training Curve (AdamW, noise_std={NOISE_STD}, accum={GRAD_ACCUM})"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Config / summary JSON
    summary = {
        "model": model_config,
        "model_params": (
            {**CFNO_PARAMS, "shifting_modes": model_config["shifting_modes"]}
            if model_config["type"] == "cfno"
            else EDSR_PARAMS
        ),
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "noise_std": NOISE_STD,
            "use_amp": USE_AMP,
            "snapshot_index": SNAPSHOT_INDEX,
        },
        "results": {
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
        },
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2)


# =====================================================================
# Training loop
# =====================================================================


def train_one_model(
    model_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
) -> float:
    """Train a single model with SGD + Gaussian noise augmentation.

    Returns the best validation loss.
    """
    name = model_config["name"]
    mtype = model_config["type"]
    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"{'=' * 60}")

    model = _build_model(model_config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
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

        # ── train ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            hr = batch[0].to(DEVICE, non_blocking=True)
            lr = batch[1].to(DEVICE, non_blocking=True)

            # Gaussian noise augmentation on the low-resolution input
            lr = lr + torch.randn_like(lr) * NOISE_STD

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    output = _forward(model, lr, mtype)
                    loss = loss_fn(output, hr) / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                output = _forward(model, lr, mtype)
                loss = loss_fn(output, hr) / GRAD_ACCUM
                loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM  # un-scale for logging

            # Step optimizer every GRAD_ACCUM batches (or at end of epoch)
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
                        loss = loss_fn(output, hr)
                else:
                    output = _forward(model, lr, mtype)
                    loss = loss_fn(output, hr)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"train={epoch_loss:.6f}  val={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | no_improve={epochs_no_improve}/{EARLY_STOP_PATIENCE} | {elapsed:.1f}s"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch + 1}.")
            break

    # ── persist ───────────────────────────────────────────────────
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    run_dir = output_dir / name
    _save_artifacts(
        run_dir, model_config, best_state, train_losses, val_losses, best_val_loss
    )
    print(f"  ✅ Saved to {run_dir}")
    print(f"  Best val loss: {best_val_loss:.6f}")

    # ── free GPU memory ───────────────────────────────────────────
    del model, best_state, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_loss


# =====================================================================
# Main
# =====================================================================


def main():
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    folder_name = f"{FOLDER_NAME}{timestamp}"
    # f"best_models_{timestamp}"
    output_dir = OUTPUT_ROOT / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {output_dir}")
    print(f"Device           : {DEVICE}")
    print(f"Train data       : {TRAIN_H5}")
    print(f"Val data         : {VAL_H5}")
    print(f"Snapshot index   : {SNAPSHOT_INDEX}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Optimizer        : AdamW(lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"Noise std        : {NOISE_STD}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"AMP              : {USE_AMP}")

    # ── datasets ──────────────────────────────────────────────────
    print("\nLoading training dataset …")
    train_ds = dataset_sr(h5_path=TRAIN_H5, snapshot_index=SNAPSHOT_INDEX)
    print(f"  Training samples: {len(train_ds)}")

    print("Loading validation dataset …")
    val_ds = dataset_sr(h5_path=VAL_H5, snapshot_index=SNAPSHOT_INDEX)
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

    # ── train each model sequentially ─────────────────────────────
    results = {}
    for model_config in MODELS:
        try:
            best_val = train_one_model(
                model_config, train_loader, val_loader, output_dir
            )
            results[model_config["name"]] = {
                "status": "success",
                "best_val_loss": best_val,
            }
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()
            results[model_config["name"]] = {
                "status": "failed",
                "error": str(e),
            }
            gc.collect()
            torch.cuda.empty_cache()

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        if result["status"] == "success":
            print(f"  {name:20s}  val_loss = {result['best_val_loss']:.6f}")
        else:
            print(f"  {name:20s}  FAILED — {result['error']}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _write_manifest(output_dir)
    print(f"  Manifest written to {output_dir / 'manifest.json'}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
