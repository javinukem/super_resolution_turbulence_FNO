"""
Energy-spectrum comparison for the ``ufno_mse_spectral`` experiment.

Plots the 1-D total-energy power spectrum P(k) of the SR×4 output of two
models next to the HR target and the LR input, on one log-log axes with a
``k^{-2}`` reference slope:

  - ``U-SFNO``               — the UFNO_2 run trained here (best config from
                                ``ufno_l1_spectral_unet`` re-trained with
                                MSE+spectral loss under the clip_p30 regime).
  - ``SFNO (MSE + Spectral)`` — the canonical stabilized CFNO (Baseline B
                                config: shift=8, skip=trilinear) trained with
                                MSE + light spectral (``w_minor``) loss under
                                the same clip_p30 regime, **reused** (not
                                retrained) from the ``mse_loss_combinations_*
                                manifest (entry whose ``loss`` is
                                ``"mse+spectral"``).

Mirrors ``experiments/mse_loss_combinations/plot_spectra.py`` and follows the
``comparing_best_models_mse`` naming convention (CFNO* -> "SFNO",
UFNO* -> "U-SFNO").

The HR/LR reference states are the **same** jf1uids-simulated state (seed
1234, 128³) used by ``plot_final_snapshot.py`` — loaded from the shared
``comparison_states.npy`` cache when available, and regenerated on the fly
otherwise — so the spectra plot stays consistent with the final-snapshot
state-grid plot.

Usage
-----
    python experiments/ufno_mse_spectral/plot_spectra.py
    python experiments/ufno_mse_spectral/plot_spectra.py --regen
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.manifest import (
    build_model,
    discover_latest,
    load_manifest,
    load_norm_stats,
)

# Reuse the jf1uids sim + block-average helpers from plot_final_snapshot so the
# spectra and the state-grid plots share the exact same HR/LR reference state.
from experiments.ufno_mse_spectral.plot_final_snapshot import (
    _downaverage_state,
    _generate_hr_state,
    HR_NUM_CELLS,
    SEED,
)

# jf1uids data structures
from jf1uids import SimulationConfig
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids.fluid_equations.fluid import (
    get_absolute_velocity,
    total_energy_from_primitives,
)

import jax.numpy as jnp
import Pk_library as PKL

# ── Config ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
# NB: the glob must NOT match the older ``ufno_mse_spectral_500_*`` experiment
# folder (ufno_mse_spectral_500e) — the ``??-??_*`` date prefix excludes it.
SCRATCH_GLOB = "ufno_mse_spectral_??-??_*"
MSE_LOSS_COMBOS_GLOB = "mse_loss_combinations_*"
UPSAMPLE_FACTOR = 4
GAMMA = float(Fraction(5, 3))

REPO_EXP_DIR = ROOT / "experiments" / "ufno_mse_spectral"
REPO_EXP_DIR.mkdir(parents=True, exist_ok=True)
STATES_NPY = REPO_EXP_DIR / "comparison_states.npy"
OUTPUT_PATH = REPO_EXP_DIR / "spectra_comparison.png"

# Display labels for the two plotted models (CFNO* -> "SFNO", UFNO* -> "U-SFNO",
# per the comparing_best_models_mse naming convention).
UFNO_LABEL = "U-SFNO"
SFNO_MSE_SPECTRAL_LABEL = "SFNO (MSE + Spectral)"
MSE_LOSS_COMBOS_LOSS_TAG = "mse+spectral"

# Fixed colors for the reference states; models get a distinct color.
COLORS = {
    "HR (128³)": "black",
    "LR (32³)": "tab:blue",
}


def get_energy_spectrum(primitive_state, config, registered_variables, gamma):
    """1-D total-energy power spectrum of a primitive state via Pk_library."""
    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]
    energy = np.array(
        total_energy_from_primitives(rho, u, p, gamma), dtype=np.float32
    )
    return PKL.Pk(
        delta=energy, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
    )


def _load_reference_states(regen: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the (HR, LR×4) reference states used by plot_final_snapshot.

    Reuses the ``comparison_states.npy`` cache written by
    ``plot_final_snapshot.py`` when available (and ``--regen`` is not set);
    otherwise regenerates the jf1uids HR state (seed 1234, 128³) and
    block-averages it down to 32³ — identical to plot_final_snapshot's path.
    """
    if not regen and STATES_NPY.exists():
        print(f"Loading cached reference states from {STATES_NPY}")
        cached = np.load(STATES_NPY, allow_pickle=True).item()
        return cached["hr"], cached["lr"]

    print("Generating HR state via jf1uids (seed 1234, 128³) …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr4 = _downaverage_state(hr, 4)
    print(f"  HR {hr.shape}  LR(x4) {lr4.shape}")
    return hr, lr4


def _collect_model_entries(manifest: dict) -> list[tuple[str, dict, dict]]:
    """Return ``(display_label, entry, manifest)`` for the two plotted models.

      - ``U-SFNO``               : the UFNO run from this experiment's manifest
                                    (the first non-trilinear entry).
      - ``SFNO (MSE + Spectral)`` : the CFNO ``mse+spectral`` run, **reused**
                                    (not retrained) from the newest
                                    ``mse_loss_combinations_*`` manifest.

    Both manifests share the same ``l1_spectral_weighting`` normalization
    stats, so a single ``load_norm_stats`` result works for both entries.
    """
    out: list[tuple[str, dict, dict]] = []

    # UFNO run from this experiment.
    for entry in manifest["models"]:
        if entry["model_type"] == "trilinear":
            continue
        out.append((UFNO_LABEL, entry, manifest))
        break

    # CFNO (MSE + light spectral) reused from mse_loss_combinations.
    combos_dir = discover_latest(SCRATCH_BASE, MSE_LOSS_COMBOS_GLOB)
    combos_manifest = load_manifest(combos_dir)
    print(f"MSE-loss-combinations manifest: {combos_dir}")
    for entry in combos_manifest["models"]:
        if entry.get("loss") == MSE_LOSS_COMBOS_LOSS_TAG:
            out.append((SFNO_MSE_SPECTRAL_LABEL, entry, combos_manifest))
            break
    else:
        print(
            "  Warning: no 'mse+spectral' entry found in the "
            "mse_loss_combinations manifest — SFNO (MSE + Spectral) will be "
            "omitted from the spectra plot."
        )

    return out


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
        help="Force fresh jf1uids sim (ignore cached comparison_states.npy).",
    )
    args = parser.parse_args()

    # ── Discover the manifest ──────────────────────────────────────────
    manifest_dir = Path(args.manifest) if args.manifest else discover_latest(
        SCRATCH_BASE, SCRATCH_GLOB
    )
    manifest = load_manifest(manifest_dir)
    print(f"Manifest: {manifest_dir}")

    models = _collect_model_entries(manifest)
    print(f"  Models ({len(models)}):")
    for label, entry, _ in models:
        print(f"    - {label}  (model={entry['name']})")

    # ── Load the same HR/LR reference state as plot_final_snapshot ─────
    hr_state, lr_state = _load_reference_states(args.regen)
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

    # Both manifests reuse the same l1_spectral_weighting normalization
    # stats, so a single loader works for both models.
    norm_stats = load_norm_stats(manifest, DEVICE)

    model_cmap = plt.cm.tab10(np.linspace(0.1, 0.9, max(len(models), 1)))
    for i, (label, entry, entry_manifest) in enumerate(models):
        norm_stats_i = load_norm_stats(entry_manifest, DEVICE) or norm_stats
        run = build_model(entry, DEVICE, norm_stats_i)
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"\nSaved {OUTPUT_PATH}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()