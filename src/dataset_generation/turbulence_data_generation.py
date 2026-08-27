# IMPORTANT: check gpustat --watch before 
# running scripts on the cluster
# BETTER NOT USE NOTEBOOKS AT ALL,
# THEY BLOCK THE GPU MEMORY IF NOT
# RESET PROPERLY

from pathlib import Path


import yaml
config = yaml.safe_load(open(Path(__file__).resolve().parents[2] /'config.yaml', 'r'))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
# you may also use
# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================
# in regular python scripts

# numerics
import jax.numpy as jnp
import numpy as np
# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# jf1uids data structures
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes import WindConfig
from jf1uids.option_classes.simulation_config import BACKWARDS, OSHER, FORWARDS

# jf1uids setup functions
from jf1uids import get_helper_data
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

# turbulent ic setup
from jf1uids.initial_condition_generation.turb import create_turb_field

# main simulation function
from jf1uids import time_integration

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c

import h5py

from pathlib import Path

save_path = Path(__file__).resolve().parents[2] / "data/final_states_h5"

print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0

# resolution
num_cells = 128

# turbulence
turbulence = False
wanted_rms = 50 * u.km / u.s

fixed_timestep = False
dt_max = 0.1

# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    first_order_fallback = False,
    progress_bar = False,
    dimensionality = 3,
    num_ghost_cells = 2,
    box_size = box_size, 
    num_cells = num_cells,
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
    return_snapshots = True,
    num_snapshots = 80,
)

helper_data = get_helper_data(config)
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

# set the simulation parameters
params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = dt_max,
    gamma = gamma,
    t_end = t_end,
)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value

# turbulence parameters
turbulence_slope = -2
kmin = 2
kmax = 64

i = 0
a = num_cells // 2 - 10
b = num_cells // 2 + 10
p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value
os.makedirs(save_path, exist_ok=True)

max_snapshots = 500  # total number you expect to save
shape = (max_snapshots, config.dimensionality + 2, config.num_cells, config.num_cells, config.num_cells)  # include channels if needed

h5_path = os.path.join(save_path, "final_states.h5")
h5f = h5py.File(h5_path, "w")
dataset = h5f.create_dataset("states", shape=shape, dtype='float32')


while i < max_snapshots:
    u_x = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax)
    u_y = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax)
    u_z = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax)

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
        config = config,
        registered_variables=registered_variables,
        density = rho,
        velocity_x = u_x,
        velocity_y = u_y,
        velocity_z = u_z,
        gas_pressure = p
    )

    config = finalize_config(config, initial_state.shape)
    result = time_integration(initial_state, config, params, helper_data, registered_variables)
    final_state = result.states[-1]
    if np.all(final_state == 0):
        continue
    #np.savez_compressed(f"{save_path}/state_{i:03d}.npz", final_state=final_state)
    dataset[i] = final_state.astype('float16') 
    i += 1
    if i % 10 == 0:
        print(i)
    del result
    del initial_state
h5f.close()