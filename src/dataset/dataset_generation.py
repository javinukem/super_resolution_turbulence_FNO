# IMPORTANT: check gpustat --watch before
# running scripts on the cluster
# BETTER NOT USE NOTEBOOKS AT ALL,
# THEY BLOCK THE GPU MEMORY IF NOT
# RESET PROPERLY
import yaml
from pathlib import Path

# legacy flat config.yaml at the repo root (two levels up from this file)
config_file = yaml.safe_load(
    open(Path(__file__).resolve().parents[2] / "config.yaml", "r")
)
import os

# # ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# # =======================

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix data structures
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    OSHER,
    FORWARDS,
)

# astronomix setup functions
from astronomix import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import (
    SnapshotSettings,
    finalize_config,
)

# turbulent ic setup
from astronomix.initial_condition_generation.turbulent_ic_generator import (
    create_turb_field,
)

# main simulation function
from astronomix import time_integration

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c

import h5py

from scipy.ndimage import convolve


def convolve_lr(
    images: np.array, stride: int = 4, kernel: np.array = np.ones((3, 3, 3)) / 27
) -> np.array:
    n_images = images.shape[0]
    n_channels = images.shape[1]
    convolved_size = images.shape[2] // stride
    convolved_images = np.empty(
        (n_images, n_channels, convolved_size, convolved_size, convolved_size)
    )
    for batch_n in range(n_images):
        for channel in range(n_channels):
            convolved_images[batch_n, channel] = convolve(
                images[batch_n, channel], kernel, mode="reflect", cval=0.0
            )[::stride, ::stride, ::stride]
    return convolved_images


# setup simulation config
config = SimulationConfig(
    runtime_debugging=config_file["turbulent_sim"]["runtime_debug"],
    first_order_fallback=config_file["turbulent_sim"]["first_order_fb"],
    progress_bar=config_file["turbulent_sim"]["progress_bar"],
    dimensionality=config_file["turbulent_sim"]["dimensionality"],
    num_ghost_cells=config_file["turbulent_sim"]["num_ghost_cells"],
    box_size=config_file["turbulent_sim"]["box_size"],
    num_cells=config_file["turbulent_sim"]["num_cells"],
    fixed_timestep=config_file["turbulent_sim"]["fixed_timestep"],
    differentiation_mode=FORWARDS,
    return_snapshots=config_file["turbulent_sim"]["return_snapshots"],
    num_snapshots=config_file["turbulent_sim"]["num_snapshots"],
    snapshot_settings=SnapshotSettings(
        return_states=True, return_total_mass=True, return_total_energy=True
    ),
)

registered_variables = get_registered_variables(config)

# setup the unit system
code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.4

# set the final time of the simulation
t_final = 1.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

# simulation settings
gamma = 5 / 3

# turbulence
wanted_rms = 50 * u.km / u.s

dt_max = 0.1

# set the simulation parameters
params = SimulationParams(
    C_cfl=C_CFL,
    dt_max=dt_max,
    gamma=gamma,
    t_end=t_end,
)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho = (
    jnp.ones((config.num_cells, config.num_cells, config.num_cells))
    * rho_0.to(code_units.code_density).value
)

# turbulence parameters
turbulence_slope = config_file["turbulent_sim"]["turbulence_slope"]
kmin = config_file["turbulent_sim"]["kmin"]
kmax = config_file["turbulent_sim"]["kmax"]

i = 0
p = (
    jnp.ones((config.num_cells, config.num_cells, config.num_cells))
    * p_0.to(code_units.code_pressure).value
)
save_path = "./data/jalegria/full_states_h5"
os.makedirs(save_path, exist_ok=True)

h5_path = os.path.join(save_path, "full_states.h5")
h5f = h5py.File(h5_path, "w")

max_sims = config_file["turbulent_sim"]["max_sims"]
hr_shape = (
    max_sims * config.num_snapshots,
    config.dimensionality + 2,
    config.num_cells,
    config.num_cells,
    config.num_cells,
)
lr_shape = (
    max_sims * config.num_snapshots,
    config.dimensionality + 2,
    config.num_cells // config_file["data"]["upsample_factor"],
    config.num_cells // config_file["data"]["upsample_factor"],
    config.num_cells // config_file["data"]["upsample_factor"],
)

hr_states_dataset = h5f.create_dataset("hr_states", shape=hr_shape, dtype="float32")
lr_states_dataset = h5f.create_dataset("lr_states", shape=lr_shape, dtype="float32")
energy_dataset = h5f.create_dataset(
    "first_snapshot_energy", shape=(max_sims * config.num_snapshots), dtype="float32"
)
mass_dataset = h5f.create_dataset(
    "first_snapshot_mass", shape=(max_sims * config.num_snapshots), dtype="float32"
)

snapshot_counter = 0
rng_key = jax.random.PRNGKey(0)
while i < max_sims:
    rng_key, key_x, key_y, key_z = jax.random.split(rng_key, 4)
    u_x = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key_x)
    u_y = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key_y)
    u_z = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key_z)

    # scale the turbulence to the desired rms velocity
    rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))
    if not jnp.isfinite(rms_vel) or rms_vel == 0.0:
        print("Skipping iteration due to bad rms_vel:", rms_vel)
        continue
    u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value
    # construct primitive state
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u_x,
        velocity_y=u_y,
        velocity_z=u_z,
        gas_pressure=p,
    )

    config = finalize_config(config, initial_state.shape)
    result = time_integration(
        initial_state, config, params, registered_variables
    )

    if np.all(result.states[-1] == 0):
        continue

    lr_batch = convolve_lr(
        np.array(result.states), config_file["data"]["upsample_factor"]
    )
    for snapshot_idx, snapshot in enumerate(result.states):
        hr_states_dataset[snapshot_counter] = np.array(snapshot, dtype="float32")
        lr_states_dataset[snapshot_counter] = lr_batch[snapshot_idx]
        energy_dataset[snapshot_counter] = result.total_energy[0]
        mass_dataset[snapshot_counter] = result.total_mass[0]
        snapshot_counter += 1
    i += 1
    if i % 10 == 0:
        print(i)
    del result
    del initial_state
h5f.close()
