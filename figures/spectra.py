import sys
import os
import yaml
from pathlib import Path

sys.path.append(os.path.abspath("."))  # This brings 'src' into the path
figure_config = yaml.safe_load(
    open(Path(__file__).resolve().parents[1] / "configs/figures/spectra.yaml", "r")
)

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from autocvd import autocvd

autocvd(num_gpus=1)

from src.dataloader.dataloader_3d import dataset_sr
from src.utils.pipeline import load_model_from_folder
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# jf1uids data structures
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes import WindConfig
from jf1uids.option_classes.simulation_config import BACKWARDS, OSHER, FORWARDS

# jf1uids setup functions
from jf1uids import get_helper_data
from jf1uids.fluid_equations.fluid import (
    construct_primitive_state,
    get_absolute_velocity,
    total_energy_from_primitives,
)

from jf1uids.fluid_equations import total_quantities as tq
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

from astropy import units as u
from fractions import Fraction

import Pk_library as PKL
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_folder = figure_config["paths"]["model_folder"]
model, _ = load_model_from_folder(model_folder=model_folder, device=device)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dsfno_model = DSFNO(
#     in_channel=5,
#     modes=config_file["dsfno"]["modes"],
#     n_channels=config_file["dsfno"]["n_channels"],
#     n_residual_blocks=config_file["dsfno"]["n_residual_blocks"],
#     n_operator_blocks=config_file["dsfno"]["n_operator_blocks"],
#     apply_constraint=config_file["dsfno"]["apply_constraint"],
# ).to(device)
# m16_nc32_res3_op2_ac0_lk3lcexp (best performing model so far)
dataset = dataset_sr()
hr_state, lr_state, energy, mass = dataset[0]
# high_res = hr_state.to(device)
low_res = lr_state.to(device)
superresolved_4 = torch.squeeze(model(torch.unsqueeze(low_res,0), 4))
superresolved_2 = torch.squeeze(model(torch.unsqueeze(low_res,0), 2))

low_res = low_res.cpu()
superresolved_4 = superresolved_4.cpu()
superresolved_2 = superresolved_2.cpu()

print("👷 Setting up simulation...")
# simulation settings
gamma = float(Fraction(figure_config["turbulent_sim"]["gamma"]))
config = SimulationConfig(
    # runtime_debugging=config_file["turbulent_sim"]["runtime_debug"],
    # first_order_fallback=config_file["turbulent_sim"]["first_order_fb"],
    # progress_bar=config_file["turbulent_sim"]["progress_bar"],
    dimensionality=3,
    # num_ghost_cells=config_file["turbulent_sim"]["num_ghost_cells"],
    # box_size=config_file["turbulent_sim"]["box_size"],
    # num_cells=config_file["turbulent_sim"]["num_cells"],
    # fixed_timestep=config_file["turbulent_sim"]["fixed_timestep"],
    # differentiation_mode=FORWARDS,
    # return_snapshots=config_file["turbulent_sim"]["return_snapshots"],
    # num_snapshots=config_file["turbulent_sim"]["num_snapshots"],
)
config = finalize_config(config, hr_state.shape)
helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)


def get_energy_spectrum(primitive_state, config, registered_variables, gamma):
    """Calculate the total energy from the primitive state."""
    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]
    energy = np.array(total_energy_from_primitives(rho, u, p, gamma), dtype=np.float32)
    pk_energy = PKL.Pk(
        delta=energy, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
    )
    return pk_energy


# --- Compute and plot energy spectra for all states recursively ---

# Collect all primitive states in a dictionary for convenience
states = {
    "HR (128³)": jnp.array(hr_state.numpy()),
    "LR (32³)": jnp.array(low_res.numpy()),
    "SR ×4": jnp.array(superresolved_4.detach().numpy()),
    "SR ×2": jnp.array(superresolved_2.detach().numpy()),
}

# Define colors or styles (optional)
colors = {
    "HR (128³)": "black",
    "LR (32³)": "tab:blue",
    "SR ×4": "tab:green",
    "SR ×2": "tab:orange",
}

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Dictionary to store spectra for later analysis if needed
energy_spectra = {}

# Loop through all states
for label, primitive_state in states.items():
    print(f"🔍 Computing energy spectrum for {label}...")
    spectrum = get_energy_spectrum(
        primitive_state=primitive_state,
        config=config,
        registered_variables=registered_variables,
        gamma=gamma,
    )
    energy_spectra[label] = spectrum

    # Plot each spectrum
    ax.plot(
        spectrum.k1D,
        spectrum.Pk1D,
        lw=2,
        label=label,
        color=colors.get(label, None),
    )

# Add reference slope (k^-2)
k_ref = energy_spectra["HR (128³)"].k1D
ax.plot(
    k_ref,
    k_ref**-2,
    linestyle="--",
    color="gray",
    label=r"$k^{-2}$ reference",
)

# Format plot
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k)$")
ax.set_ylim(1e-8, 1e-1)
ax.legend()
ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(
    figure_config["paths"]["output_path"],
    dpi=200,
)
plt.show()
# ax[1].scatter(range(0, len(kinetic_e_spectrum)), kinetic_e_spectrum)
# ax[1].set_yscale("log")
# ax[1].set_xscale("log")
# ax[1].set_title("Kinetic energy")

# ax[2].scatter(range(0, len(mass_spectrum)), mass_spectrum)
# ax[2].set_yscale("log")
# ax[2].set_xscale("log")
# ax[2].set_title("mass spectrum")
