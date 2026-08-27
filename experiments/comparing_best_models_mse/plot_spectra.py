"""
Energy-spectrum comparison for the ``comparing_best_models_mse`` preset.

Mirrors ``figures/spectra.py`` but, instead of a single hardcoded model, plots
the 1-D energy spectrum P(k) of the SR×4 output of every model in the
``comparing_best_models_mse`` preset (the MSE-only CFNO, the best MSE-only
EDSR ``edsr_norm_skip_on``, and the trilinear interpolation baseline) next to
the HR target and the LR input, all on one log-log axes with a k^{-2}
reference slope.

The HR/LR reference states are the **same** jf1uids-simulated state (seed
1234, 128³) used by ``plot_final_snapshot.py`` — loaded from the shared
``comparison_states.npy`` cache when available, and regenerated on the fly
otherwise — so the spectra plot stays consistent with the final-snapshot
state-grid plot.

Model discovery, weights loading and LR/SR normalization are delegated to the
shared manifest machinery (``evaluation.manifest`` +
``evaluation.comparing_models_bar_chart`` preset builder) so this script stays
in lock-step with the bar-chart comparison — no duplicated model knowledge.

Usage
-----
    python experiments/comparing_best_models_mse/plot_spectra.py
    python experiments/comparing_best_models_mse/plot_spectra.py --regen
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

from evaluation.manifest import build_model, get_model_entry, load_norm_stats
from evaluation.comparing_models_bar_chart import (
    _best_models_manifest,
    _build_comparing_best_models_mse,
)

# Reuse the jf1uids sim + block-average helpers from plot_final_snapshot so the
# spectra and the state-grid plots share the exact same HR/LR reference state.
from experiments.comparing_best_models_mse.plot_final_snapshot import (
    _display_label,
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
UPSAMPLE_FACTOR = 4
GAMMA = float(Fraction(5, 3))

REPO_EXP_DIR = ROOT / "experiments" / "comparing_best_models_mse"
STATES_NPY = REPO_EXP_DIR / "comparison_states.npy"
OUTPUT_PATH = REPO_EXP_DIR / "spectra_comparison.png"

# Fixed colors for the reference states; preset models get a colormap slice.
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
        return cached["hr"], cached["lr4"]

    print("Generating HR state via jf1uids (seed 1234, 128³) …")
    hr = _generate_hr_state(HR_NUM_CELLS, SEED)
    lr4 = _downaverage_state(hr, 4)
    print(f"  HR {hr.shape}  LR(x4) {lr4.shape}")
    return hr, lr4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force fresh jf1uids sim (ignore cached comparison_states.npy).",
    )
    args = parser.parse_args()

    # ── Discover the preset's series (manifest + entry name + label) ──
    print("Discovering manifests …")
    bm = _best_models_manifest()
    series = _build_comparing_best_models_mse(bm, None, None)
    print(f"  Series ({len(series)}):")
    for s in series:
        print(f"    - {_display_label(s['label'])}  (model={s['model']})")

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

    model_cmap = plt.cm.tab10(np.linspace(0.1, 0.9, max(len(series), 1)))
    for i, s in enumerate(series):
        label = _display_label(s["label"])
        manifest = s["manifest"]
        entry = get_model_entry(manifest, s["model"])
        norm_stats = load_norm_stats(manifest, DEVICE)
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