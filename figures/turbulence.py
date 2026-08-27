import os

# # ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# jf1uids.jf1uids data structures
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes import WindConfig
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    OSHER,
    FORWARDS,
    HLL,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

# jf1uids.jf1uids setup functions
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

import random

from animate import animate_states

print("👷 Setting up simulation...")

# simulation settings
adiabatic_index = 5 / 3
box_size = 1.0
num_cells = 128
wanted_rms = 50 * u.km / u.s
fixed_timestep = False
dt_max = 0.1
mhd = True

# setup simulation config
config = SimulationConfig(
    runtime_debugging=False,
    first_order_fallback=False,
    progress_bar=True,
    dimensionality=3,
    num_ghost_cells=2,
    box_size=box_size,
    num_cells=num_cells,
    mhd=mhd,
    fixed_timestep=fixed_timestep,
    differentiation_mode=FORWARDS,
    riemann_solver=HLL,
    limiter=0,
    return_snapshots=True,
    num_snapshots=80,
    # boundary_settings=BoundarySettings(
    #    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    #    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    #    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    # ),
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# setup the unit system
code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.4  # Courant-Friedrichs-Lewy number
t_final = 1.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

# set the simulation parameters
params = SimulationParams(
    C_cfl=C_CFL,
    dt_max=dt_max,
    gamma=adiabatic_index,
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
turbulence_slope = -2
kmin = 2
kmax = 50

p = (
    jnp.ones((config.num_cells, config.num_cells, config.num_cells))
    * p_0.to(code_units.code_pressure).value
)

u_x = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax)
u_y = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax)
u_z = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax)

# scale the turbulence to the desired rms velocity
rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))

u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value

r = helper_data.r


if mhd:
    grid_spacing = config.box_size / config.num_cells
    x = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )
    y = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )
    z = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    """
    rho = jnp.ones_like(X)
    P = jnp.ones_like(X) * 0.1
    r_inj = 0.1 * box_size
    p_inj = 10.0
    P = jnp.where(r**2 < r_inj**2, p_inj, P)
    """

    B_0 = 1 / np.sqrt(2)
    B_x = jnp.zeros_like(X)
    B_y = jnp.zeros_like(X)
    B_z = B_0 * jnp.ones_like(X)
    initial_magnetic_field = jnp.stack([B_x, B_y, B_z], axis=0)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u_x,
        velocity_y=u_y,
        velocity_z=u_z,
        gas_pressure=p,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
    )
else:
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u_x,
        velocity_y=u_y,
        velocity_z=u_z,
        gas_pressure=p,
    )

print(registered_variables)
config = finalize_config(config, initial_state.shape)
result = time_integration(
    initial_state, config, params, helper_data, registered_variables
)

z_level = num_cells // 2

animate_states(result, "turbulent_magnetic", z_level)
