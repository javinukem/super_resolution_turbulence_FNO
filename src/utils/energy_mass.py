import jax
import jax.numpy as jnp
import numpy as np
# timing
from timeit import default_timer as timer

# astronomix data structures
from astronomix import SimulationConfig
from astronomix.option_classes.simulation_config import BACKWARDS, OSHER, FORWARDS

# astronomix setup functions
from astronomix import get_helper_data
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from fractions import Fraction
from astronomix._fluid_equations.total_quantities import get_absolute_velocity

import sys
import os
import yaml
sys.path.append(os.path.abspath(".."))  # This brings 'src' into the path
config_file = yaml.safe_load(open('../config.yaml', 'r'))

def initialize_config(state_shape: list = (5, 128, 128, 128)):
    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = config_file["turbulent_sim"]["runtime_debug"],
        first_order_fallback = config_file["turbulent_sim"]["first_order_fb"],
        progress_bar = config_file["turbulent_sim"]["progress_bar"],
        dimensionality = config_file["turbulent_sim"]["dimensionality"],
        num_ghost_cells = config_file["turbulent_sim"]["num_ghost_cells"],
        box_size = config_file["turbulent_sim"]["box_size"], 
        num_cells = config_file["turbulent_sim"]["num_cells"],
        fixed_timestep = config_file["turbulent_sim"]["fixed_timestep"],
        differentiation_mode = FORWARDS,
        return_snapshots = config_file["turbulent_sim"]["return_snapshots"],
        num_snapshots = config_file["turbulent_sim"]["num_snapshots"]
    )
    config = finalize_config(config, state_shape=state_shape)
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    return config, helper_data, registered_variables


def internal_energy_spectrum(state, gamma, config, registered_variables):
    num_ghost_cells = config.num_ghost_cells
    p = state[registered_variables.pressure_index]

    if config.cosmic_ray_config.cosmic_rays:
        gamma_cr = 4/3
        p = p - state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    internal_energy = p / (gamma - 1)

    # Remove ghost cells if necessary
    if config.dimensionality == 1:
        internal_energy = internal_energy[num_ghost_cells:-num_ghost_cells]
    else:
        slices = tuple(slice(num_ghost_cells, -num_ghost_cells) for _ in range(config.dimensionality))
        internal_energy = internal_energy[slices]

    internal_energy_np = np.array(internal_energy)

    fft_energy = np.fft.fftn(internal_energy_np)
    fft_energy_shifted = np.fft.fftshift(fft_energy)
    power_spectrum = np.abs(fft_energy_shifted)**2

    return power_spectrum

def calculate_kinetic_energy(state, helper_data, config, registered_variables):
    num_ghost_cells = config.num_ghost_cells

    rho = state[registered_variables.density_index]
    u = get_absolute_velocity(state, config, registered_variables)

    kinetic_energy = 0.5 * rho * u ** 2

    if config.dimensionality == 1:
        return jnp.sum(kinetic_energy[num_ghost_cells:-num_ghost_cells] * helper_data.cell_volumes[num_ghost_cells:-num_ghost_cells])
    else:
        return jnp.sum(kinetic_energy * config.grid_spacing**config.dimensionality)

def kinetic_energy_spectrum(state, helper_data, config, registered_variables):
    num_ghost_cells = config.num_ghost_cells
    rho = state[registered_variables.density_index]


    u = get_absolute_velocity(state, config, registered_variables)

    kinetic_energy = 0.5 * rho * u ** 2

    if config.dimensionality == 1:
        kinetic_energy =  kinetic_energy[num_ghost_cells:-num_ghost_cells] * helper_data.cell_volumes[num_ghost_cells:-num_ghost_cells]
    else:
         kinetic_energy =  kinetic_energy * config.grid_spacing**config.dimensionality

    kinetic_energy_np = np.array(kinetic_energy)

    fft_energy = np.fft.fftn(kinetic_energy_np)
    fft_energy_shifted = np.fft.fftshift(fft_energy)
    power_spectrum = np.abs(fft_energy_shifted)**2

    return power_spectrum

def mass_spectrum(
    state,
    helper_data,
    config
):
    num_ghost_cells = config.num_ghost_cells

    if config.dimensionality == 1:
        mass = state[0, num_ghost_cells:-num_ghost_cells] * helper_data.cell_volumes[num_ghost_cells:-num_ghost_cells]
    else:
        slice_off_ghost_cells = (0,) + (slice(num_ghost_cells, -num_ghost_cells),) * config.dimensionality
        # note that here the box size is assumed to be the box size without the ghost cells
        mass = state[slice_off_ghost_cells] * config.box_size**config.dimensionality
    
    mass_np = np.array(mass)

    fft_mass = np.fft.fftn(mass_np)
    fft_mass_shifted = np.fft.fftshift(fft_mass)
    power_spectrum = np.abs(fft_mass_shifted)**2

    return power_spectrum

def radial_spectrum(power_spectrum):
    shape = power_spectrum.shape
    center = [s // 2 for s in shape]
    
    # Create coordinate grids
    z, y, x = np.indices(shape)
    k = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    k = k.astype(int)

    # Bin average
    k_max = int(np.max(k))
    spectrum = np.zeros(k_max + 1)
    counts = np.zeros(k_max + 1)
    
    for i in range(k_max + 1):
        mask = (k == i)
        spectrum[i] = power_spectrum[mask].sum()
        counts[i] = mask.sum()

    return spectrum / np.maximum(counts, 1)

