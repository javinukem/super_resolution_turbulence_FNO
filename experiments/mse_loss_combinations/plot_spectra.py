"""
Energy-spectrum comparison for the ``mse_loss_combinations`` experiment.

Mirrors ``experiments/comparing_best_models_mse/plot_spectra.py`` but, instead
of the ``comparing_best_models_mse`` preset, plots the 1-D energy spectrum
P(k) of the SR×4 output of three MSE-based CFNO runs:

  - ``SFNO (MSE)``          — the MSE-only CFNO, reused (not retrained) from
                              the ``comparing_best_models_mse_*`` manifest
  - ``SFNO (MSE+Spectral)`` — from the ``mse_loss_combinations_*`` manifest
  - ``SFNO (MSE+L1)``       — from the ``mse_loss_combinations_*`` manifest

The MSE + spectral + L1 combo is intentionally excluded. References (HR
target, LR input) and a k^{-2} slope are plotted alongside on one log-log
axes.

The HR/LR reference states are the **same** jf1uids-simulated state (seed
1234, 128³) used by ``plot_final_snapshot.py`` — loaded from the shared
``comparison_states.npy`` cache when available, and regenerated on the fly
otherwise — so the spectra plot stays consistent with the final-snapshot
state-grid plot.

Usage
-----
    python experiments/mse_loss_combinations/plot_spectra.py
    python experiments/mse_loss_combinations/plot_spectra.py --regen
"""

from autocvd import autocvd

autocvd(num_gpus=1)

import argparse
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
from experiments.mse_loss_combinations.plot_final_snapshot import (
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
UPSAMPLE_FACTOR = 4
GAMMA = float(Fraction(5, 3))

REPO_EXP_DIR = ROOT / "experiments" / "mse_loss_combinations"
STATES_NPY = REPO_EXP_DIR / "comparison_states.npy"
OUTPUT_PATH = REPO_EXP_DIR / "spectra_comparison.png"

# Display labels per loss tag (the MSE + spectral + L1 combo is excluded).
COMBO_LABELS = {
    "mse+spectral": "SFNO (MSE+Spectral)",
    "mse+l1": "SFNO (MSE+L1)",
}
MSE_ONLY_LABEL = "SFNO (MSE)"

# Fixed colors for the reference states; models get a colormap slice.
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


def _collect_model_entries() -> list[tuple[str, dict, dict]]:
    """Return ``(display_label, entry, manifest)`` for the 3 plotted models.

    The MSE-only CFNO comes from the ``comparing_best_models_mse_*`` manifest
    (reused, not retrained); the MSE+Spectral and MSE+L1 combos come from the
    ``mse_loss_combinations_*`` manifest. The MSE+Spectral+L1 combo is
    excluded.
    """
    combos_dir = discover_latest(SCRATCH_BASE, "mse_loss_combinations_*")
    combos_manifest = load_manifest(combos_dir)
    print(f"Combos manifest: {combos_dir}")

    mse_only_dir = discover_latest(SCRATCH_BASE, "comparing_best_models_mse_*")
    mse_only_manifest = load_manifest(mse_only_dir)
    print(f"MSE-only manifest: {mse_only_dir}")

    models: list[tuple[str, dict, dict]] = []

    # MSE-only reference (from comparing_best_models_mse) — plotted first.
    for entry in mse_only_manifest["models"]:
        if entry.get("model_type") == "cfno":
            models.append((MSE_ONLY_LABEL, entry, mse_only_manifest))
            break

    # The two selected combos (MSE+Spectral, MSE+L1), in manifest order.
    for entry in combos_manifest["models"]:
        label = COMBO_LABELS.get(entry.get("loss"))
        if label is not None:
            models.append((label, entry, combos_manifest))

    return models, combos_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force fresh jf1uids sim (ignore cached comparison_states.npy).",
    )
    args = parser.parse_args()

    # ── Discover the 3 models (MSE-only ref + MSE+Spectral + MSE+L1) ──
    print("Discovering manifests …")
    models, combos_manifest = _collect_model_entries()
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
    # stats, so a single loader works for all three models.
    norm_stats = load_norm_stats(combos_manifest, DEVICE)

    model_cmap = plt.cm.tab10(np.linspace(0.1, 0.9, max(len(models), 1)))
    for i, (label, entry, _) in enumerate(models):
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
