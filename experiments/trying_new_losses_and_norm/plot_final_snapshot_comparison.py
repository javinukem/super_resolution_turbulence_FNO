"""
Final-snapshot comparison plots for the ``trying_new_losses_and_norm`` experiment.

Runs a single jf1uids simulation with the same parameters as the dataset,
takes the final snapshot as the HR reference, builds the LR by block-averaging,
runs every model listed in the experiment manifest (CFNO shift=8 and EDSR, each
with three losses and two normalization settings), and produces three
comparison figures:

    plot 1 — comparison_cfno_x4.png
        rows: HR | CFNO {mse_l1, mse_spectral, l1} × {nonorm, norm} | interp | LR
        Shows the effect of loss + normalization on CFNO.

    plot 2 — comparison_edsr_x4.png
        rows: HR | EDSR {mse_l1, mse_spectral, l1} × {nonorm, norm} | interp | LR
        Shows the effect of loss + normalization on EDSR.

    plot 3 — comparison_best_per_loss_x4.png
        rows: HR | best CFNO per loss (3) | best EDSR per loss (3) | interp | LR
        "best" = the norm setting with lowest val MSE from benchmark_results.csv.

Norm-on models are wrapped by ``evaluation.manifest.build_model``: LR is
normalized with cached training statistics before inference, and the SR output
is denormalized back to the raw scale.

The final-snapshot HR/LR and all SR predictions are cached to
``comparison_states.npy`` so re-runs skip the (expensive) jf1uids simulation
and model inference.  Use ``--regen`` to force a fresh simulation + inference.

Usage
-----
    python experiments/trying_new_losses_and_norm/plot_final_snapshot_comparison.py
    python experiments/trying_new_losses_and_norm/plot_final_snapshot_comparison.py \
        --regen
    python experiments/trying_new_losses_and_norm/plot_final_snapshot_comparison.py \
        --manifest /path/to/manifest.json
"""

from autocvd import autocvd

autocvd(num_gpus=1)
import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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

from evaluation.manifest import (
    build_model,
    discover_latest,
    load_manifest,
    load_norm_stats,
)

# ── Paths & constants ─────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = ROOT / "experiments" / "trying_new_losses_and_norm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATES_NPY = OUTPUT_DIR / "comparison_states.npy"

SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
DEFAULT_MANIFEST_GLOB = "loss_norm_experiment_*"

HR_NUM_CELLS = 128
BASE_UPSAMPLE_FACTOR = 4
SEED = 1234

CHANNEL_NAMES = ["density", "vx", "vy", "vz", "pressure"]


# ── Plot axes derived from the manifest ───────────────────────────────


def _derive_axes(manifest: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (models, losses, norm_tags) preserving first-seen order.

    Skips the trilinear baseline entry. Used to structure the plot rows.
    """
    models, losses, norms = [], [], []
    for e in manifest["models"]:
        if e["model_type"] == "trilinear":
            continue
        mn = e.get("model_name")
        if mn and mn not in models:
            models.append(mn)
        if e.get("loss") and e["loss"] not in losses:
            losses.append(e["loss"])
        tag = "norm" if e.get("use_norm") else "nonorm"
        if tag not in norms:
            norms.append(tag)
    return models, losses, norms


# ── Simulation (mirrors figures/plot_hr_lr_sr_slices.py) ──────────────


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
        C_cfl=0.4, dt_max=float(cfg_data["dt_max"]), gamma=5 / 3, t_end=t_end,
    )

    rho_0 = 2 * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B
    rho = jnp.ones((num_cells,) * 3) * rho_0.to(code_units.code_density).value
    p = jnp.ones((num_cells,) * 3) * p_0.to(code_units.code_pressure).value

    for _ in range(8):
        u_x = create_turb_field(
            num_cells, 1, cfg_data["turbulence_slope"],
            cfg_data["kmin"], cfg_data["kmax"],
        )
        u_y = create_turb_field(
            num_cells, 1, cfg_data["turbulence_slope"],
            cfg_data["kmin"], cfg_data["kmax"],
        )
        u_z = create_turb_field(
            num_cells, 1, cfg_data["turbulence_slope"],
            cfg_data["kmin"], cfg_data["kmax"],
        )
        rms = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))
        if not jnp.isfinite(rms) or float(rms) == 0.0:
            continue
        wanted_rms = (
            (float(cfg_data["wanted_rms"]) * u.km / u.s)
            .to(code_units.code_velocity).value
        )
        u_x = u_x / rms * wanted_rms
        u_y = u_y / rms * wanted_rms
        u_z = u_z / rms * wanted_rms

        initial_state = construct_primitive_state(
            config=config, registered_variables=reg_vars,
            density=rho, velocity_x=u_x, velocity_y=u_y, velocity_z=u_z, gas_pressure=p,
        )
        config_run = finalize_config(config, initial_state.shape)
        result = time_integration(
            initial_state, config_run, params, helper_data, reg_vars
        )
        snapshot = np.array(result.states[-1], dtype=np.float32)
        if np.isfinite(snapshot).all():
            return snapshot
    raise RuntimeError("Could not generate a finite turbulent HR state.")


# ── Plotting ──────────────────────────────────────────────────────────


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
                slices[row_idx], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
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


# ── Best-per-loss selection from benchmark CSV ────────────────────────


def _best_norm_per_loss(
    benchmark_csv: Path,
    models: list[str],
    losses: list[str],
) -> dict[str, str]:
    """Return {(model, loss): norm_tag} picking the lowest-MSE norm setting.

    Falls back to 'norm' for all if the CSV is missing.
    """
    fallback = {f"{m}_{loss}": "norm" for m in models for loss in losses}
    if not benchmark_csv.exists():
        print("  (benchmark_results.csv missing — defaulting to norm)")
        return fallback
    df = pd.read_csv(benchmark_csv)
    best = {}
    for model in models:
        for loss in losses:
            sub = df[(df["model"] == model) & (df["loss"] == loss)]
            if sub.empty:
                best[f"{model}_{loss}"] = "norm"
                continue
            row = sub.loc[sub["MSE"].idxmin()]
            best[f"{model}_{loss}"] = "norm" if row["use_norm"] else "nonorm"
    return best


# ── State building ────────────────────────────────────────────────────


def _build_states(regen: bool, manifest: dict) -> dict:
    if not regen and STATES_NPY.exists():
        print(f"Loading cached states from {STATES_NPY}")
        return np.load(STATES_NPY, allow_pickle=True).item()

    print("Generating HR state via jf1uids …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr = _downaverage_state(hr, BASE_UPSAMPLE_FACTOR)
    print(f"  HR {hr.shape}  LR {lr.shape}")

    states = {"hr": hr, "lr": lr}

    norm_stats = load_norm_stats(manifest, DEVICE)

    # trilinear baseline (norm-off, no model weights)
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        states["interp"] = (
            F.interpolate(
                lr_t,
                scale_factor=BASE_UPSAMPLE_FACTOR,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0).cpu().numpy()
        )
    del lr_t

    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        name = entry["name"]
        print(f"  Running {name} …")
        run = build_model(entry, DEVICE, norm_stats)
        lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
        sr = run(lr_t, BASE_UPSAMPLE_FACTOR)
        states[name] = sr.squeeze(0).detach().cpu().numpy()
        del run.model, lr_t
        gc.collect()
        torch.cuda.empty_cache()

    np.save(STATES_NPY, states, allow_pickle=True)
    print(f"  Cached states to {STATES_NPY}")
    return states


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force fresh sim + inference.",
    )
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
    print(f"Manifest : {manifest['_manifest_path']}")
    benchmark_csv = manifest["benchmark_csv"]

    models, losses, norm_tags = _derive_axes(manifest)

    torch.manual_seed(SEED)
    states = _build_states(regen=args.regen, manifest=manifest)
    best = _best_norm_per_loss(benchmark_csv, models, losses)

    # ── Plot 1: CFNO all variants (loss × norm) ────────────────────
    cfno_rows = [("HR target", states["hr"])]
    for loss in losses:
        for norm_tag in norm_tags:
            key = f"cfno_shift8_{loss}_{norm_tag}"
            if key in states:
                cfno_rows.append((f"{loss} ({norm_tag})", states[key]))
    cfno_rows.append(("Trilinear", states["interp"]))
    cfno_rows.append(("LR", states["lr"]))
    _plot_state_grid(
        rows=cfno_rows,
        save_path=OUTPUT_DIR / "comparison_cfno_x4.png",
        title="CFNO shift=8 — x4: loss × normalization comparison",
    )

    # ── Plot 2: EDSR all variants (loss × norm) ────────────────────
    edsr_rows = [("HR target", states["hr"])]
    for loss in losses:
        for norm_tag in norm_tags:
            key = f"edsr_{loss}_{norm_tag}"
            if key in states:
                edsr_rows.append((f"{loss} ({norm_tag})", states[key]))
    edsr_rows.append(("Trilinear", states["interp"]))
    edsr_rows.append(("LR", states["lr"]))
    _plot_state_grid(
        rows=edsr_rows,
        save_path=OUTPUT_DIR / "comparison_edsr_x4.png",
        title="EDSR (prelu) — x4: loss × normalization comparison",
    )

    # ── Plot 3: Best per loss (both models, best norm) ─────────────
    best_rows = [("HR target", states["hr"])]
    for model in models:
        for loss in losses:
            key = f"{model}_{loss}_{best[f'{model}_{loss}']}"
            label = f"{model} {loss} ({best[f'{model}_{loss}']})"
            if key in states:
                best_rows.append((label, states[key]))
    best_rows.append(("Trilinear", states["interp"]))
    best_rows.append(("LR", states["lr"]))
    _plot_state_grid(
        rows=best_rows,
        save_path=OUTPUT_DIR / "comparison_best_per_loss_x4.png",
        title="x4: best norm setting per loss (by val MSE)",
    )


if __name__ == "__main__":
    main()
