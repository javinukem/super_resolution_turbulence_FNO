"""
Final-snapshot comparison plots for the UFNO (MSE+spectral, 500e clip_p30) run.

Runs a fresh jf1uids turbulent simulation (seed 1234, 128^3), block-averages
to LR, runs the trained UFNO model from the ufno_mse_spectral manifest, and
produces two PNGs in the home experiment folder:

    - ``final_snapshot_comparison.png`` — full state grid
      (HR | U-SFNO | LR), all 5 channels, central z-slice. A red rectangle
      marks the central-quarter region shown in the zoomed plot below.
    - ``zoomed_snapshot_comparison.png`` — zoomed central-quarter region
      (HR | U-SFNO | LR) only, all 5 channels. LR is shown at its native
      (downsampled) resolution for that region; the others share the panel.

States are cached to ``comparison_states.npy`` in the home experiment folder.

This script is the jf1uids-side companion to ``train_ufno_mse_spectral.py``:
``run.sh`` invokes it after training. Ported from
``experiments/ufno_mse_spectral_500e/plot_final_snapshot.py`` (with zoom).

Usage
-----
    python experiments/ufno_mse_spectral/plot_final_snapshot.py
    python experiments/ufno_mse_spectral/plot_final_snapshot.py --regen
    python experiments/ufno_mse_spectral/plot_final_snapshot.py \\
        --manifest /export/scratch/jalegria/experiments/ufno_mse_spectral_...
"""

from autocvd import autocvd

autocvd(num_gpus=1)
import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
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
# NB: the glob must NOT match the older ``ufno_mse_spectral_500_*`` experiment
# folder (ufno_mse_spectral_500e) — the ``??-??_*`` date prefix excludes it.
SCRATCH_GLOB = "ufno_mse_spectral_??-??_*"
REPO_EXP_DIR = ROOT / "experiments" / "ufno_mse_spectral"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── Final-snapshot sim constants (mirror loss_ablation) ──────────────

HR_NUM_CELLS = 128
SEED = 1234
UPSAMPLE_FACTOR = 4
CHANNEL_NAMES = ["density", "vx", "vy", "vz", "pressure"]

# Zoom region: central quarter of the HR slice (32x32 of 128x128),
# corresponding to 8x8 of the 32x32 LR slice.
ZOOM_FRAC = 0.25


# =====================================================================
# Sim + state helpers (ported from loss_ablation / edsr_norm_skip)
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


def _z_slice(arr: np.ndarray) -> np.ndarray:
    """Central z-slice, transposed for imshow (origin=lower)."""
    z_idx = arr.shape[-1] // 2
    return arr[:, :, z_idx].T


# =====================================================================
# Plots
# =====================================================================


def _zoom_rect_xy(slc: np.ndarray) -> tuple[float, float, int, int]:
    """Central zoom rectangle in imshow data coords for a (hy, hx) slice.

    Returns (x0, y0, nx, ny) covering the central ``ZOOM_FRAC`` of each
    axis — to be drawn on top of an ``origin="lower"`` imshow of ``slc``.
    """
    hy, hx = slc.shape
    nx = max(1, int(round(hx * ZOOM_FRAC)))
    ny = max(1, int(round(hy * ZOOM_FRAC)))
    x0 = (hx - nx) // 2
    y0 = (hy - ny) // 2
    return x0, y0, nx, ny


def _plot_state_grid(
    rows: list[tuple[str, np.ndarray]],
    save_path: Path,
    title: str,
    draw_zoom_rect: bool = False,
) -> None:
    n_rows = len(rows)
    n_cols = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx in range(n_cols):
        slices = []
        for _, state in rows:
            slices.append(_z_slice(state[col_idx]))
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
            if draw_zoom_rect:
                x0, y0, nx, ny = _zoom_rect_xy(slices[row_idx])
                rect = mpatches.Rectangle(
                    (x0, y0), nx, ny,
                    fill=False, edgecolor="red", lw=1.5, alpha=0.9,
                )
                ax.add_patch(rect)
            fig.colorbar(img, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def _crop(arr: np.ndarray, frac: float) -> np.ndarray:
    """Crop each spatial dimension of a (C, H, W, D) array to the central
    ``frac`` fraction (rounded to integers centred on the middle)."""
    _, hx, hy, hz = arr.shape
    nx = max(1, int(round(hx * frac)))
    ny = max(1, int(round(hy * frac)))
    nz = max(1, int(round(hz * frac)))
    x0, y0, z0 = (hx - nx) // 2, (hy - ny) // 2, (hz - nz) // 2
    return arr[:, x0:x0 + nx, y0:y0 + ny, z0:z0 + nz]


def _plot_zoom_grid(
    rows: list[tuple[str, np.ndarray]],
    save_path: Path,
    title: str,
) -> None:
    """Zoomed-grid plot. Each row is (label, full (C,H,W,D) state); we crop
    to the central ``ZOOM_FRAC`` and show the central z-slice. HR and the
    SR model share the same zoomed resolution while LR shows its native
    downsampled resolution for that region (so the LR panel is visibly
    blocky). Per-channel vmin/vmax is computed across HR and the SR rows
    only — LR is drawn on its own scale so its blockiness is visible
    without swamping the colourbar."""
    n_rows = len(rows)
    n_cols = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    crops = [(name, _crop(state, ZOOM_FRAC)) for name, state in rows]

    for col_idx in range(n_cols):
        hr_like = [
            _z_slice(crop[col_idx])
            for name, crop in crops
            if name.lower() != "lr"
        ]
        vmin = min(float(np.nanmin(s)) for s in hr_like)
        vmax = max(float(np.nanmax(s)) for s in hr_like)

        for row_idx, (row_name, crop) in enumerate(crops):
            ax = axes[row_idx, col_idx]
            slc = _z_slice(crop[col_idx])
            if row_name.lower() == "lr":
                img = ax.imshow(slc, origin="lower", cmap="viridis")
            else:
                img = ax.imshow(
                    slc, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(CHANNEL_NAMES[col_idx])
            if col_idx == 0:
                ax.set_ylabel(row_name)
                ax.set_xlabel(f"central {int(ZOOM_FRAC * 100)}%")
            fig.colorbar(img, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# =====================================================================
# State building (HR sim + model inference)
# =====================================================================


def _build_states(regen: bool, manifest: dict, states_npy: Path) -> dict:
    if not regen and states_npy.exists():
        print(f"Loading cached states from {states_npy}")
        states = np.load(states_npy, allow_pickle=True).item()
        # Invalidate the cache if any manifest model isn't present — otherwise
        # we'd silently plot only HR/LR (the bug that masked stale caches).
        missing = [
            e["name"]
            for e in manifest["models"]
            if e["model_type"] != "trilinear" and e["name"] not in states
        ]
        if not missing:
            return states
        print(
            f"  Cached states are stale (missing {missing}) — regenerating."
        )

    print("Generating HR state via jf1uids …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr = _downaverage_state(hr, UPSAMPLE_FACTOR)
    print(f"  HR {hr.shape}  LR {lr.shape}")

    states = {"hr": hr, "lr": lr}
    norm_stats = load_norm_stats(manifest, DEVICE)

    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        name = entry["name"]
        print(f"  Running {name} …")
        try:
            run = build_model(entry, DEVICE, norm_stats)
            lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
            sr = run(lr_t, UPSAMPLE_FACTOR)
            states[name] = sr.squeeze(0).detach().cpu().numpy()
            del run.model, lr_t
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM running {name} — skipping")
            else:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    np.save(states_npy, states, allow_pickle=True)
    print(f"  Cached states to {states_npy}")
    return states


# =====================================================================
# Top-level plot assembly
# =====================================================================


def _plot_final_snapshot(states: dict, manifest: dict, output_dir: Path) -> None:
    rows: list[tuple[str, np.ndarray]] = [("HR target", states["hr"])]
    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        name = entry["name"]
        if name in states:
            rows.append(("U-SFNO", states[name]))
            break
    rows.append(("LR", states["lr"]))
    _plot_state_grid(
        rows=rows,
        save_path=output_dir / "final_snapshot_comparison.png",
        title=(
            "U-SFNO (MSE+spectral w_minor, 500e clip_p30): "
            "final-snapshot comparison @ x4"
        ),
        draw_zoom_rect=True,
    )


def _plot_zoomed_snapshot(states: dict, manifest: dict, output_dir: Path) -> None:
    # Order: HR | U-SFNO | LR (model 2nd, LR 3rd).
    rows: list[tuple[str, np.ndarray]] = [("HR", states["hr"])]
    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        name = entry["name"]
        if name in states:
            rows.append(("U-SFNO", states[name]))
            break
    rows.append(("LR", states["lr"]))
    _plot_zoom_grid(
        rows=rows,
        save_path=output_dir / "zoomed_snapshot_comparison.png",
        title=(
            f"U-SFNO (MSE+spectral w_minor, 500e clip_p30): "
            f"central {int(ZOOM_FRAC * 100)}% zoom @ x4 (HR | U-SFNO | LR)"
        ),
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
            f"newest {SCRATCH_GLOB} under scratch."
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
        manifest_dir = discover_latest(SCRATCH_BASE, SCRATCH_GLOB)
    manifest = load_manifest(manifest_dir)
    # The comparison plots + cached states live in the home experiment folder
    # (not the scratch run dir), per the experiment's layout convention.
    print(f"Manifest: {manifest_dir}")
    print(f"Saving plots to: {REPO_EXP_DIR}")

    torch.manual_seed(SEED)
    states = _build_states(args.regen, manifest, REPO_EXP_DIR / "comparison_states.npy")
    _plot_final_snapshot(states, manifest, REPO_EXP_DIR)
    _plot_zoomed_snapshot(states, manifest, REPO_EXP_DIR)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()