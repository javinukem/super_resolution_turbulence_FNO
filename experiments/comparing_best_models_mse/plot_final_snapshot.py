"""
Final-snapshot comparison plots for the comparing_best_models_mse experiment.

Runs a fresh jf1uids turbulent simulation (seed 1234, 128^3), block-averages
to LR at both x4 (32^3) and x2 (64^3), and runs:

  - the CFNO (MSE-only, shift=8, skip=trilinear) from this experiment's
    ``comparing_best_models_mse_*`` manifest — at both x4 and x2
    (supports_variable_scale=True, eval_scales=[4, 2]), and
  - the EDSR "Baseline D" (norm on, skip on) from the ``edsr_norm_skip_*``
    manifest — at x4 only (fixed-scale x4; eval_scales=[4]), and
  - a trilinear interpolation baseline — at both x4 and x2,

and produces TWO final-snapshot state-grid PNGs:

  - ``final_snapshot_comparison_x4.png`` :
      HR | CFNO@x4 | Baseline D@x4 | Trilinear@x4 | LR(32^3)
  - ``final_snapshot_comparison_x2.png`` :
      HR | CFNO@x2 | Trilinear@x2 | LR(64^3)

The three-model lineup (CFNO + Baseline D + Trilinear) mirrors the
``comparing_best_models_mse`` bar-chart preset
(``evaluation/comparing_models_bar_chart.py``); at x2 Baseline D is omitted
because EDSR is fixed-scale x4 and cannot natively do 2x SR (its
``eval_scales`` is [4]).
States are cached to ``comparison_states.npy`` in the home experiment folder.

Usage
-----
    python experiments/comparing_best_models_mse/plot_final_snapshot.py
    python experiments/comparing_best_models_mse/plot_final_snapshot.py --regen
    python experiments/comparing_best_models_mse/plot_final_snapshot.py \\
        --manifest /export/scratch/jalegria/experiments/comparing_best_models_mse_...
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
REPO_EXP_DIR = ROOT / "experiments" / "comparing_best_models_mse"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# Name of the EDSR "Baseline D" entry inside the edsr_norm_skip manifest
# (must match comparing_models_bar_chart.py::_build_comparing_best_models_mse).
EDSR_BASELINE_D_NAME = "edsr_norm_skip_on"
EDSR_BASELINE_D_LABEL = "EDSR"


def _display_label(label: str) -> str:
    """Short display names for the plot: CFNO* -> "SFNO", Baseline D* -> "EDSR"."""
    if label.startswith("CFNO"):
        return "SFNO"
    if label.startswith("Baseline D"):
        return "EDSR"
    return label

# ── Final-snapshot sim constants (mirror edsr_norm_skip) ──────────────

HR_NUM_CELLS = 128
SEED = 1234
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


def _run_model_at(
    entry: dict, lr_t: torch.Tensor, scale: int, norm_stats
) -> np.ndarray | None:
    """Run one model entry at ``scale`` and return its SR state (or None on OOM).

    For fixed-scale entries (EDSR), ``scale`` is forwarded to ``build_model``'s
    callable but ignored by the underlying model — the caller must ensure the
    LR spatial size is compatible with the entry's native upscale (e.g. for
    EDSR x4 we feed a 32^3 LR to obtain a 128^3 SR).
    """
    try:
        run = build_model(entry, DEVICE, norm_stats)
        sr = run(lr_t, scale)
        out = sr.squeeze(0).detach().cpu().numpy()
        del run.model
        return out
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    OOM running {entry['name']} @x{scale} — skipping")
            return None
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _build_states(
    regen: bool,
    cfno_manifest: dict,
    edsr_manifest: dict | None,
    states_npy: Path,
) -> dict:
    if not regen and states_npy.exists():
        print(f"Loading cached states from {states_npy}")
        return np.load(states_npy, allow_pickle=True).item()

    print("Generating HR state via jf1uids …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr4 = _downaverage_state(hr, 4)
    lr2 = _downaverage_state(hr, 2)
    print(f"  HR {hr.shape}  LR(x4) {lr4.shape}  LR(x2) {lr2.shape}")

    states = {"hr": hr, "lr4": lr4, "lr2": lr2}
    # Both manifests reuse the l1_spectral_weighting normalization stats, so a
    # single loader works for both. Prefer the CFNO manifest's stats (it is
    # the experiment's own).
    norm_stats = load_norm_stats(cfno_manifest, DEVICE)

    # Trilinear baselines at each scale.
    for scale, lr in ((4, lr4), (2, lr2)):
        lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            states[f"trilinear_x{scale}"] = (
                F.interpolate(
                    lr_t,
                    scale_factor=scale,
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0).cpu().numpy()
            )
        del lr_t

    # CFNO at x4 and x2 (supports_variable_scale=True, eval_scales=[4, 2]).
    for entry in cfno_manifest["models"]:
        name = entry["name"]
        for scale in (4, 2):
            print(f"  Running {name} @x{scale} …")
            lr = lr4 if scale == 4 else lr2
            lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
            out = _run_model_at(entry, lr_t, scale, norm_stats)
            if out is not None:
                states[f"{name}_x{scale}"] = out
            del lr_t

    # EDSR "Baseline D" at x4 only (fixed-scale; eval_scales=[4]).
    if edsr_manifest is not None:
        for entry in edsr_manifest["models"]:
            if entry.get("name") != EDSR_BASELINE_D_NAME:
                continue
            print(f"  Running {entry['name']} @x4 (Baseline D) …")
            lr_t = torch.from_numpy(lr4).unsqueeze(0).to(DEVICE)
            out = _run_model_at(entry, lr_t, 4, norm_stats)
            if out is not None:
                states[f"{entry['name']}_x4"] = out
            del lr_t

    np.save(states_npy, states, allow_pickle=True)
    print(f"  Cached states to {states_npy}")
    return states


def _plot_final_snapshot(
    states: dict,
    cfno_manifest: dict,
    edsr_manifest: dict | None,
    output_dir: Path,
) -> None:
    # ── x4: HR | SFNO@x4 | EDSR@x4 | Trilinear@x4 | LR ────────────────
    rows_x4: list[tuple[str, np.ndarray]] = [("HR target", states["hr"])]
    for entry in cfno_manifest["models"]:
        key = f"{entry['name']}_x4"
        if key in states:
            rows_x4.append((_display_label(entry["label"]), states[key]))
    if edsr_manifest is not None:
        for entry in edsr_manifest["models"]:
            if entry.get("name") != EDSR_BASELINE_D_NAME:
                continue
            key = f"{entry['name']}_x4"
            if key in states:
                rows_x4.append((EDSR_BASELINE_D_LABEL, states[key]))
    rows_x4.append(("Trilinear", states["trilinear_x4"]))
    rows_x4.append(("LR", states["lr4"]))
    _plot_state_grid(
        rows=rows_x4,
        save_path=output_dir / "final_snapshot_comparison_x4.png",
        title="Comparing best models (MSE only): SFNO vs EDSR vs Trilinear @ x4",
    )

    # ── x2: HR | SFNO@x2 | Trilinear@x2 | LR ────────────────────────
    # EDSR is fixed-scale x4 → omitted at x2, matching its manifest's
    # eval_scales=[4] and the bar-chart preset's per-scale filter.
    rows_x2: list[tuple[str, np.ndarray]] = [("HR target", states["hr"])]
    for entry in cfno_manifest["models"]:
        key = f"{entry['name']}_x2"
        if key in states:
            rows_x2.append((_display_label(entry["label"]), states[key]))
    rows_x2.append(("Trilinear", states["trilinear_x2"]))
    rows_x2.append(("LR", states["lr2"]))
    _plot_state_grid(
        rows=rows_x2,
        save_path=output_dir / "final_snapshot_comparison_x2.png",
        title="Comparing best models (MSE only): SFNO vs Trilinear @ x2",
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
            "Experiment folder (containing manifest.json) for the CFNO run. "
            "Defaults to the newest comparing_best_models_mse_* under scratch."
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
        manifest_dir = discover_latest(SCRATCH_BASE, "comparing_best_models_mse_*")
    cfno_manifest = load_manifest(manifest_dir)

    # Cross-reference the EDSR "Baseline D" run from the edsr_norm_skip
    # experiment (reused — not retrained here).
    edsr_manifest = None
    try:
        edsr_dir = discover_latest(SCRATCH_BASE, "edsr_norm_skip_*")
        edsr_manifest = load_manifest(edsr_dir)
        print(f"Baseline D reference manifest: {edsr_dir}")
    except FileNotFoundError as e:
        print(
            f"  Warning: edsr_norm_skip manifest not found ({e}) "
            f"— Baseline D row will be omitted from the x4 grid."
        )

    # The comparison plot + cached states live in the home experiment folder
    # (not the scratch run dir), per the experiment's layout convention.
    print(f"Manifest: {manifest_dir}")
    print(f"Saving plot to: {REPO_EXP_DIR}")

    torch.manual_seed(SEED)
    states = _build_states(
        args.regen, cfno_manifest, edsr_manifest,
        REPO_EXP_DIR / "comparison_states.npy",
    )
    _plot_final_snapshot(states, cfno_manifest, edsr_manifest, REPO_EXP_DIR)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()