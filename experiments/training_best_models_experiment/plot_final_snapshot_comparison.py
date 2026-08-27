"""
Final-snapshot turbulence comparison plots for the best-trained models.

Runs a single jf1uids simulation with the same parameters as the dataset
(read from the legacy top-level ``config.yaml`` ``turbulent_sim`` block), takes
the final snapshot as the HR reference, builds the LR by block-averaging, runs
every model listed in the experiment manifest (CFNO at x4 and x2 for every
``shifting_modes`` variant, EDSR at x4, plus a trilinear baseline), and
produces three comparison figures:

    plot 1 — comparison_x4.png
        rows: HR target | CFNO (best val-loss shift) | EDSR | trilinear interp | LR
        5 columns: density, vx, vy, vz, pressure

    plot 2 — comparison_x2.png
        rows: HR target (x2) | CFNO (best val-loss shift) | trilinear interp | LR
        (EDSR has no x2 model)

    plot 3 — comparison_x4_cfno_all.png
        rows: HR target | CFNO shift{0,4,8,12,16} | trilinear interp | LR
        only ``shifting_modes`` differs across the CFNO rows

The final-snapshot HR/LR and all SR predictions are cached to
``comparison_states.npy`` so re-runs skip the (expensive) jf1uids simulation
and model inference.  Use ``--regen`` to force a fresh simulation + inference.

Usage
-----
    python experiments/training_best_models_experiment/plot_final_snapshot_comparison.py
    python experiments/training_best_models_experiment/\
        plot_final_snapshot_comparison.py --regen
    python experiments/training_best_models_experiment/\
        plot_final_snapshot_comparison.py --manifest /path/to/manifest.json
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
)

# ── Paths & constants ─────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR: Path = (
    ROOT
    / "experiments"
    / "training_best_models_experiment"
    / "trained_on_last_snapshot"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATES_NPY = OUTPUT_DIR / "comparison_states.npy"

SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
DEFAULT_MANIFEST_GLOB = "best_models_*"

HR_NUM_CELLS = 128
BASE_UPSAMPLE_FACTOR = 4
SEED = 1234

CHANNEL_NAMES = ["density", "vx", "vy", "vz", "pressure"]


# ── CFNO shifts derived from the manifest ─────────────────────────────


def _cfno_shifts(manifest: dict) -> list[int]:
    """Return sorted shifting_modes values of the manifest's CFNO entries."""
    shifts = [
        e["model_params"]["shifting_modes"]
        for e in manifest["models"]
        if e["model_type"] == "cfno"
    ]
    return sorted(shifts)


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


# ── State building ────────────────────────────────────────────────────


def _run_to_numpy(run, lr: np.ndarray, scale: int) -> np.ndarray:
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
    sr = run(lr_t, scale)
    return sr.squeeze(0).detach().cpu().numpy()


def _target_x2(hr: np.ndarray) -> np.ndarray:
    return (
        F.interpolate(
            torch.from_numpy(hr).unsqueeze(0),
            scale_factor=0.5,
            mode="trilinear",
            align_corners=False,
        )
        .squeeze(0)
        .numpy()
    )


def _build_states(regen: bool, manifest: dict) -> dict:
    if not regen and STATES_NPY.exists():
        print(f"Loading cached states from {STATES_NPY}")
        return np.load(STATES_NPY, allow_pickle=True).item()

    print("Generating HR state via jf1uids …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr = _downaverage_state(hr, BASE_UPSAMPLE_FACTOR)
    hr_x2 = _target_x2(hr)
    print(f"  HR {hr.shape}  LR {lr.shape}  HR_x2 {hr_x2.shape}")

    states = {"hr": hr, "lr": lr, "hr_x2": hr_x2}

    for entry in manifest["models"]:
        mtype = entry["model_type"]
        run = build_model(entry, DEVICE)
        try:
            if mtype == "cfno":
                shift = entry["model_params"]["shifting_modes"]
                print(f"Running CFNO shift={shift} (x4 + x2) …")
                states[f"cfno_shift{shift}_x4"] = _run_to_numpy(run, lr, 4)
                states[f"cfno_shift{shift}_x2"] = _run_to_numpy(run, lr, 2)
            elif mtype == "edsr":
                print("Running EDSR (x4) …")
                states["edsr_x4"] = _run_to_numpy(run, lr, 4)
            elif mtype == "trilinear":
                print("Running trilinear baseline (x4 + x2) …")
                states["interp_x4"] = _run_to_numpy(run, lr, 4)
                states["interp_x2"] = _run_to_numpy(run, lr, 2)
        finally:
            del run.model
            gc.collect()
            torch.cuda.empty_cache()

    np.save(STATES_NPY, states, allow_pickle=True)
    print(f"  Cached states to {STATES_NPY}")
    return states


# ── Best-shift selection from benchmark_results.csv ───────────────────


def _best_cfno_shift(benchmark_csv: Path) -> int:
    if not benchmark_csv.exists():
        print("  (benchmark_results.csv missing — defaulting to shift=8)")
        return 8
    df = pd.read_csv(benchmark_csv)
    df = df[(df["model"].str.startswith("CFNO")) & (df["upsample_factor"] == 4)]
    if df.empty:
        return 8
    best = df.loc[df["MSE"].idxmin()]
    shift = int(best["model"].split("shift=")[1].rstrip(")"))
    print(f"  Best CFNO shift @ x4 by MSE: shift={shift} (MSE={best['MSE']:.5f})")
    return shift


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

    field_cache = [(name, state) for name, state in rows]
    for col_idx, ch in enumerate(range(n_cols)):
        slices = []
        for _, state in field_cache:
            arr = state[ch]
            z_idx = arr.shape[-1] // 2
            slices.append(arr[:, :, z_idx].T)
        finite = [s for s in slices if np.isfinite(s).any()]
        vmin = min(float(np.nanmin(s)) for s in finite)
        vmax = max(float(np.nanmax(s)) for s in finite)
        for row_idx, (row_name, _) in enumerate(field_cache):
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


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force a fresh jf1uids simulation and model inference.",
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
    shifts = _cfno_shifts(manifest)

    torch.manual_seed(SEED)
    states = _build_states(regen=args.regen, manifest=manifest)
    best_shift = _best_cfno_shift(benchmark_csv)

    # ── Plot 1: x4 — HR | CFNO(best) | EDSR | interp | LR ─────────────
    _plot_state_grid(
        rows=[
            ("HR target", states["hr"]),
            (f"CFNO shift={best_shift} (x4)", states[f"cfno_shift{best_shift}_x4"]),
            ("EDSR (x4)", states["edsr_x4"]),
            ("Trilinear (x4)", states["interp_x4"]),
            ("LR", states["lr"]),
        ],
        save_path=OUTPUT_DIR / "comparison_x4.png",
        title=f"Final snapshot — x4 super-resolution (best CFNO shift={best_shift})",
    )

    # ── Plot 2: x2 — HR(x2) | CFNO(best) | interp | LR ────────────────
    _plot_state_grid(
        rows=[
            ("HR target (x2)", states["hr_x2"]),
            (f"CFNO shift={best_shift} (x2)", states[f"cfno_shift{best_shift}_x2"]),
            ("Trilinear (x2)", states["interp_x2"]),
            ("LR", states["lr"]),
        ],
        save_path=OUTPUT_DIR / "comparison_x2.png",
        title=f"Final snapshot — x2 super-resolution (best CFNO shift={best_shift})",
    )

    # ── Plot 3: x4 — HR | all CFNO shifts | interp | LR ───────────────
    cfno_rows = [
        (f"CFNO shift={s}", states[f"cfno_shift{s}_x4"]) for s in shifts
    ]
    _plot_state_grid(
        rows=[
            ("HR target", states["hr"]),
            *cfno_rows,
            ("Trilinear (x4)", states["interp_x4"]),
            ("LR", states["lr"]),
        ],
        save_path=OUTPUT_DIR / "comparison_x4_cfno_all.png",
        title="Final snapshot — x4 super-resolution: all CFNO shifting_modes",
    )


if __name__ == "__main__":
    main()
