"""
Final-snapshot comparison plot for the mse_loss_combinations experiment.

Runs a fresh jf1uids turbulent simulation (seed 1234, 128^3), block-averages
to LR, runs every trained model from BOTH:

  - the ``mse_loss_combinations_*`` manifest (the 3 new combos:
    MSE + light spectral, MSE + light L1, MSE + light spectral + light L1), and
  - the ``comparing_best_models_mse_*`` manifest (the MSE-only reference run,
    which is reused — not retrained — by this experiment)

and produces a final-snapshot state-grid comparison
(HR | MSE+light spectral | MSE+light L1 | MSE+spectral+L1 | MSE only | Trilinear | LR).
States are cached to ``comparison_states.npy`` in the home experiment folder.

This script is the jf1uids-side companion to ``train_mse_loss_combinations.py``:
``run.sh`` invokes it after training+eval. Ported from
``experiments/edsr_norm_skip/plot_final_snapshot.py``.

Usage
-----
    python experiments/mse_loss_combinations/plot_final_snapshot.py
    python experiments/mse_loss_combinations/plot_final_snapshot.py --regen
    python experiments/mse_loss_combinations/plot_final_snapshot.py \\
        --manifest /export/scratch/jalegria/experiments/mse_loss_combinations_...
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
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.manifest import (
    build_model,
    discover_latest,
    load_manifest,
    load_norm_stats,
)

# jf1uids sim imports
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

# ── Paths ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
REPO_EXP_DIR = ROOT / "experiments" / "mse_loss_combinations"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the l1_spectral_weighting normalization stats (must match training).
NORM_STATS_PATH = ROOT / "experiments" / "l1_spectral_weighting" / "normalization_stats.npz"

# ── Final-snapshot sim constants (mirror edsr_norm_skip) ──────────────

HR_NUM_CELLS = 128
SEED = 1234
UPSAMPLE_FACTOR = 4
CHANNEL_NAMES = ["density", "vx", "vy", "vz", "pressure"]


# =====================================================================
# Sim + state-grid helpers (ported from edsr_norm_skip/plot_final_snapshot.py)
# =====================================================================


def _load_turbulent_cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["turbulent_sim"]


def _downaverage_state(state: np.ndarray, downsample_factor: int) -> np.ndarray:
    channels, hx, hy, hz = state.shape
    if hx % downsample_factor or hy % downsample_factor or hz % downsample_factor:
        raise ValueError(
            f"State spatial shape {(hx, hy, hz)} not divisible by "
            f"{downsample_factor}."
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
            density=rho, velocity_x=u_x, velocity_y=u_y, velocity_z=u_z,
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
                slices[row_idx], origin="lower", cmap="viridis",
                vmin=vmin, vmax=vmax,
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


def _run_model(entry: dict, lr_t: torch.Tensor, norm_stats) -> np.ndarray | None:
    """Run one model entry at x4 and return its SR state (or None on OOM)."""
    try:
        run = build_model(entry, DEVICE, norm_stats)
        sr = run(lr_t, UPSAMPLE_FACTOR)
        out = sr.squeeze(0).detach().cpu().numpy()
        del run.model
        return out
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    OOM running {entry['name']} — skipping")
            return None
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _build_states(
    regen: bool,
    manifest: dict,
    mse_only_manifest: dict | None,
    states_npy: Path,
) -> dict:
    if not regen and states_npy.exists():
        print(f"Loading cached states from {states_npy}")
        return np.load(states_npy, allow_pickle=True).item()

    print("Generating HR state via jf1uids …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr = _downaverage_state(hr, UPSAMPLE_FACTOR)
    print(f"  HR {hr.shape}  LR {lr.shape}")

    states = {"hr": hr, "lr": lr}
    # The new-formula manifest carries its own norm stats; the MSE-only
    # reference manifest uses the same l1_spectral_weighting stats, so the
    # same loader works for both.
    norm_stats = load_norm_stats(manifest, DEVICE)

    lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        states["interp"] = (
            F.interpolate(
                lr_t,
                scale_factor=UPSAMPLE_FACTOR,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0).cpu().numpy()
        )
    del lr_t

    # The 3 new combos
    for entry in manifest["models"]:
        name = entry["name"]
        print(f"  Running {name} …")
        lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
        out = _run_model(entry, lr_t, norm_stats)
        if out is not None:
            states[name] = out
        del lr_t

    # The MSE-only reference run (reused, not retrained)
    if mse_only_manifest is not None:
        for entry in mse_only_manifest["models"]:
            name = entry["name"]
            print(f"  Running {name} (MSE-only reference) …")
            lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
            out = _run_model(entry, lr_t, norm_stats)
            if out is not None:
                states[name] = out
            del lr_t

    np.save(states_npy, states, allow_pickle=True)
    print(f"  Cached states to {states_npy}")
    return states


def _plot_final_snapshot(
    states: dict,
    manifest: dict,
    mse_only_manifest: dict | None,
    output_dir: Path,
) -> None:
    rows: list[tuple[str, np.ndarray]] = [("HR target", states["hr"])]
    # the 3 new combos, in the manifest's training order
    for entry in manifest["models"]:
        name = entry["name"]
        if name in states:
            rows.append((entry["label"], states[name]))
    # the reused MSE-only reference run
    if mse_only_manifest is not None:
        for entry in mse_only_manifest["models"]:
            name = entry["name"]
            if name in states:
                # Relabel to make its provenance clear in the grid.
                rows.append((f"{entry['label']} (ref)", states[name]))
    rows.append(("Trilinear", states["interp"]))
    rows.append(("LR", states["lr"]))
    _plot_state_grid(
        rows=rows,
        save_path=output_dir / "final_snapshot_comparison.png",
        title="MSE-based loss combinations (CFNO shift=8, skip=trilinear, x4)",
    )


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
            "Experiment folder (containing manifest.json). Defaults to the "
            "newest mse_loss_combinations_* under scratch."
        ),
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force fresh jf1uids sim + inference (ignore cached states).",
    )
    args = parser.parse_args()

    if args.manifest:
        manifest_dir = Path(args.manifest)
    else:
        manifest_dir = discover_latest(SCRATCH_BASE, "mse_loss_combinations_*")
    manifest = load_manifest(manifest_dir)

    # Cross-reference the MSE-only run from the comparing_best_models_mse
    # experiment (reused — not retrained in this experiment).
    mse_only_manifest = None
    try:
        mse_only_dir = discover_latest(SCRATCH_BASE, "comparing_best_models_mse_*")
        mse_only_manifest = load_manifest(mse_only_dir)
        print(f"MSE-only reference manifest: {mse_only_dir}")
    except FileNotFoundError as e:
        print(
            f"  Warning: comparing_best_models_mse manifest not found ({e}) "
            f"— MSE-only row will be omitted from the state grid."
        )

    # The comparison plot + cached states live in the home experiment folder
    # (not the scratch run dir), per the experiment's layout convention.
    print(f"Manifest: {manifest_dir}")
    print(f"Saving plot to: {REPO_EXP_DIR}")

    torch.manual_seed(SEED)
    states = _build_states(
        args.regen, manifest, mse_only_manifest,
        REPO_EXP_DIR / "comparison_states.npy",
    )
    _plot_final_snapshot(states, manifest, mse_only_manifest, REPO_EXP_DIR)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()