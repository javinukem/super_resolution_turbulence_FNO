"""
Energy-spectrum comparison plots (shared across experiments).

Plots the 1-D total-energy power spectrum P(k) of the SR×4 output of an
experiment's model lineup next to the HR target and the LR input, on one
log-log axes with a ``k^{-2}`` reference slope. Centralizes the former
per-experiment ``plot_spectra.py`` scripts — the experiment is selected via
``--experiment``:

  - ``usfno`` — the UFNO_2 run trained there (best config from
    ``ufno_l1_spectral_unet`` re-trained with MSE+spectral loss under the
    clip_p30 regime) plus the canonical stabilized CFNO trained with MSE +
    light spectral loss, **reused** (not retrained) from the
    ``mse_loss_combinations_*`` manifest:
      - ``U-SFNO``
      - ``SFNO (MSE + Spectral)``

  - ``sfno_loss_combinations_study`` — the MSE-based CFNO loss combos
    (the MSE + spectral + L1 combo is intentionally excluded) plus the
    MSE-only CFNO reused from the ``comparing_best_models_mse_*`` manifest:
      - ``SFNO (MSE)``
      - ``SFNO (MSE+Spectral)``
      - ``SFNO (MSE+L1)``

  - ``comparing_best_models_mse`` — the same lineup as the
    ``comparing_best_models_mse`` bar-chart preset (short display names:
    CFNO* -> "SFNO", Baseline D* -> "EDSR"):
      - ``SFNO``
      - ``EDSR``
      - ``Trilinear``

The HR/LR reference states are the **same** jf1uids-simulated state (seed
1234, 128³) across experiments — loaded from the experiment folder's shared
``comparison_states.npy`` cache when available, and regenerated on the fly
otherwise — so the spectra plot stays consistent with the final-snapshot
state-grid plots.

Usage
-----
    python src/plotting/plot_spectra.py --experiment usfno
    python src/plotting/plot_spectra.py --experiment sfno_loss_combinations_study
    python src/plotting/plot_spectra.py --experiment comparing_best_models_mse --regen
"""

from autocvd import autocvd

autocvd(num_gpus=1)

import argparse
import gc
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.manifest import (
    build_model,
    discover_latest,
    get_model_entry,
    load_manifest,
    load_norm_stats,
)

# The comparing_best_models_mse lineup mirrors its bar-chart preset so the
# spectra plot stays in lock-step with the bar-chart comparison.
from evaluation.comparing_models_bar_chart import _build_comparing_best_models_mse

# jf1uids sim imports (reference-state generation)
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
from jf1uids.fluid_equations.fluid import (
    construct_primitive_state,
    get_absolute_velocity,
    total_energy_from_primitives,
)
from jf1uids.initial_condition_generation.turb import create_turb_field
from jf1uids.option_classes.simulation_config import HLL, FORWARDS, finalize_config
from jf1uids.time_stepping.time_integration import time_integration

import Pk_library as PKL

# ── Config ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
UPSAMPLE_FACTOR = 4
GAMMA = float(Fraction(5, 3))

REPO_EXP_ROOT = ROOT / "experiments"

# ── Final-snapshot sim constants (mirror plot_final_snapshot) ──────────

HR_NUM_CELLS = 128
SEED = 1234

# Scratch run-group globs of each experiment's own manifest.
# NB: the usfno glob must NOT match the older ``ufno_mse_spectral_500_*``
# experiment folder — the ``??-??_*`` date prefix excludes it.
EXPERIMENT_GLOBS = {
    "usfno": "ufno_mse_spectral_??-??_*",
    "sfno_loss_combinations_study": "mse_loss_combinations_*",
    "comparing_best_models_mse": "comparing_best_models_mse_*",
}

# Cross-referenced manifests (models reused, not retrained).
MSE_LOSS_COMBOS_GLOB = "mse_loss_combinations_*"
COMPARING_BEST_MODELS_GLOB = "comparing_best_models_mse_*"

# Display labels (CFNO* -> "SFNO", UFNO* -> "U-SFNO", per the
# comparing_best_models_mse naming convention).
UFNO_LABEL = "U-SFNO"
SFNO_MSE_SPECTRAL_LABEL = "SFNO (MSE + Spectral)"
MSE_LOSS_COMBOS_LOSS_TAG = "mse+spectral"

MSE_ONLY_LABEL = "SFNO (MSE)"
COMBO_LABELS = {
    "mse+spectral": "SFNO (MSE+Spectral)",
    "mse+l1": "SFNO (MSE+L1)",
}

# Fixed colors for the reference states; models get a distinct color.
COLORS = {
    "HR (128³)": "black",
    "LR (32³)": "tab:blue",
}


# =====================================================================
# Sim + state helpers (shared with the final-snapshot plots)
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


# =====================================================================
# Energy spectrum
# =====================================================================


def get_energy_spectrum(primitive_state, config, registered_variables, gamma):
    """1-D total-energy power spectrum of a primitive state via Pk_library."""
    rho = primitive_state[registered_variables.density_index]
    vel = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]
    energy = np.array(
        total_energy_from_primitives(rho, vel, p, gamma), dtype=np.float32
    )
    return PKL.Pk(
        delta=energy, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
    )


def _load_reference_states(
    regen: bool, states_npy: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (HR, LR×4) reference states used by plot_final_snapshot.

    Reuses the ``comparison_states.npy`` cache written by
    ``plot_final_snapshot.py`` when available (and ``--regen`` is not set);
    otherwise regenerates the jf1uids HR state (seed 1234, 128³) and
    block-averages it down to 32³ — identical to plot_final_snapshot's path.
    """
    if not regen and states_npy.exists():
        print(f"Loading cached reference states from {states_npy}")
        cached = np.load(states_npy, allow_pickle=True).item()
        # The former per-experiment scripts used different cache keys
        # ("lr" vs "lr4") — accept both.
        lr = cached["lr"] if "lr" in cached else cached["lr4"]
        return cached["hr"], lr

    print("Generating HR state via jf1uids (seed 1234, 128³) …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr4 = _downaverage_state(hr, UPSAMPLE_FACTOR)
    print(f"  HR {hr.shape}  LR(x4) {lr4.shape}")
    return hr, lr4


# =====================================================================
# Per-experiment model lineups
# =====================================================================


def _display_label(label: str) -> str:
    """Short display names for the plot: CFNO* -> "SFNO", Baseline D* -> "EDSR"."""
    if label.startswith("CFNO"):
        return "SFNO"
    if label.startswith("Baseline D"):
        return "EDSR"
    return label


def _collect_usfno_models(manifest: dict) -> list[tuple[str, dict, dict]]:
    """U-SFNO (this experiment's UFNO run) + SFNO (MSE + Spectral) reused
    from the mse_loss_combinations run-group."""
    models: list[tuple[str, dict, dict]] = []

    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        models.append((UFNO_LABEL, entry, manifest))
        break

    combos_dir = discover_latest(SCRATCH_BASE, MSE_LOSS_COMBOS_GLOB)
    combos_manifest = load_manifest(combos_dir)
    print(f"MSE-loss-combinations manifest: {combos_dir}")
    for entry in combos_manifest["models"]:
        if entry.get("loss") == MSE_LOSS_COMBOS_LOSS_TAG:
            models.append((SFNO_MSE_SPECTRAL_LABEL, entry, combos_manifest))
            break
    else:
        print(
            "  Warning: no 'mse+spectral' entry found in the "
            "mse_loss_combinations manifest — SFNO (MSE + Spectral) will be "
            "omitted from the spectra plot."
        )

    return models


def _collect_sfno_loss_combination_models(
    manifest: dict,
) -> list[tuple[str, dict, dict]]:
    """SFNO (MSE) reused from comparing_best_models_mse + the MSE+Spectral
    and MSE+L1 combos from this experiment's manifest (the MSE + spectral +
    L1 combo is excluded)."""
    models: list[tuple[str, dict, dict]] = []

    mse_only_dir = discover_latest(SCRATCH_BASE, COMPARING_BEST_MODELS_GLOB)
    mse_only_manifest = load_manifest(mse_only_dir)
    print(f"MSE-only manifest: {mse_only_dir}")
    for entry in mse_only_manifest["models"]:
        if entry.get("model_type") in ("cfno", "sfno"):
            models.append((MSE_ONLY_LABEL, entry, mse_only_manifest))
            break

    for entry in manifest["models"]:
        label = COMBO_LABELS.get(entry.get("loss"))
        if label is not None:
            models.append((label, entry, manifest))

    return models


def _collect_comparing_best_models_models(
    manifest: dict | None,
) -> list[tuple[str, dict, dict]]:
    """The comparing_best_models_mse bar-chart preset lineup (SFNO, EDSR,
    Trilinear) — ``manifest`` is unused; the preset does its own discovery
    so the spectra plot stays in lock-step with the bar chart."""
    models: list[tuple[str, dict, dict]] = []
    for s in _build_comparing_best_models_mse():
        entry = get_model_entry(s["manifest"], s["model"])
        models.append((_display_label(s["label"]), entry, s["manifest"]))
    return models


MODEL_COLLECTORS = {
    "usfno": _collect_usfno_models,
    "sfno_loss_combinations_study": _collect_sfno_loss_combination_models,
    "comparing_best_models_mse": _collect_comparing_best_models_models,
}


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=sorted(MODEL_COLLECTORS),
        help="Which experiment's model lineup to plot.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Experiment folder (containing manifest.json). Overrides "
            "auto-discovery of the experiment's own manifest (cross-"
            "referenced manifests are always auto-discovered; ignored for "
            "comparing_best_models_mse, whose lineup comes from the "
            "bar-chart preset)."
        ),
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force fresh jf1uids sim (ignore cached comparison_states.npy).",
    )
    args = parser.parse_args()

    # ── Discover the manifest + model lineup ───────────────────────────
    if args.experiment == "comparing_best_models_mse":
        if args.manifest:
            print(
                "  Note: --manifest is ignored for comparing_best_models_mse "
                "(the lineup comes from the bar-chart preset)."
            )
        manifest = None
    else:
        manifest_dir = (
            Path(args.manifest)
            if args.manifest
            else discover_latest(SCRATCH_BASE, EXPERIMENT_GLOBS[args.experiment])
        )
        manifest = load_manifest(manifest_dir)
        print(f"Manifest: {manifest_dir}")

    models = MODEL_COLLECTORS[args.experiment](manifest)
    print(f"  Models ({len(models)}):")
    for label, entry, _ in models:
        print(f"    - {label}  (model={entry['name']})")

    # ── Output/cache locations (home experiment folder) ────────────────
    repo_exp_dir = REPO_EXP_ROOT / args.experiment
    states_npy = repo_exp_dir / "comparison_states.npy"
    output_path = repo_exp_dir / "spectra_comparison.png"

    # ── Load the same HR/LR reference state as plot_final_snapshot ─────
    hr_state, lr_state = _load_reference_states(args.regen, states_npy)
    lr_device = torch.from_numpy(lr_state).to(DEVICE)

    # ── jf1uids setup (needed for the energy spectrum) ─────────────────
    print("Setting up jf1uids config …")
    config = SimulationConfig(dimensionality=3)
    config = finalize_config(config, hr_state.shape)
    registered_variables = get_registered_variables(config)

    # ── Collect primitive states: references + each model's SR×4 ──────
    states = {
        "HR (128³)": jnp.array(hr_state),
        "LR (32³)": jnp.array(lr_state),
    }

    model_cmap = plt.cm.tab10(np.linspace(0.1, 0.9, max(len(models), 1)))
    for i, (label, entry, entry_manifest) in enumerate(models):
        norm_stats = load_norm_stats(entry_manifest, DEVICE)
        run = build_model(entry, DEVICE, norm_stats)
        with torch.no_grad():
            sr = run(lr_device.unsqueeze(0), upsample_factor=UPSAMPLE_FACTOR)
        sr = torch.squeeze(sr).cpu()
        states[label] = jnp.array(sr.detach().numpy())
        COLORS[label] = model_cmap[i]
        print(f"  {label}: SR shape {tuple(sr.shape)}")
        del run.model
        del run
        torch.cuda.empty_cache()

    # ── Compute + plot spectra ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    energy_spectra = {}

    for label, primitive_state in states.items():
        print(f"Computing energy spectrum for {label} …")
        spectrum = get_energy_spectrum(
            primitive_state=primitive_state,
            config=config,
            registered_variables=registered_variables,
            gamma=GAMMA,
        )
        energy_spectra[label] = spectrum
        ax.plot(
            spectrum.k1D,
            spectrum.Pk1D,
            lw=2,
            label=label,
            color=COLORS.get(label, None),
        )

    # k^{-2} reference slope
    k_ref = energy_spectra["HR (128³)"].k1D
    ax.plot(
        k_ref,
        k_ref**-2,
        linestyle="--",
        color="gray",
        label=r"$k^{-2}$ reference",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P(k)$")
    ax.set_ylim(1e-8, 1e-1)
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {output_path}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
