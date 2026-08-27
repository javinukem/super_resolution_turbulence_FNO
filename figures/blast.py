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
from matplotlib import animation

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

grid_spacing = config.box_size / config.num_cells
x = jnp.linspace(grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells)
y = jnp.linspace(grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells)
z = jnp.linspace(grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells)

X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

r = helper_data.r

# Initialize state
rho = jnp.ones_like(X)
P = jnp.ones_like(X) * 0.1
r_inj = 0.1 * box_size
p_inj = 10.0
P = jnp.where(r**2 < r_inj**2, p_inj, P)

u_x = jnp.zeros_like(X)
u_y = jnp.zeros_like(X)
u_z = jnp.zeros_like(X)

B_0 = 1 / np.sqrt(2)
B_x = B_0 * jnp.ones_like(X)
B_y = B_0 * jnp.ones_like(X)
B_z = jnp.zeros_like(X)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    velocity_z=u_z,
    gas_pressure=P,
    magnetic_field_x=B_x,
    magnetic_field_y=B_y,
    magnetic_field_z=B_z,
)

config = finalize_config(config, initial_state.shape)
result = time_integration(
    initial_state, config, params, helper_data, registered_variables
)


# PLOTTING
states = result.states
# z_level = random.randint(0, num_cells - 1)
z_level = num_cells // 2

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Plot 1: Scalar field (e.g., density)
cax0 = axs[0].imshow(
    states[0, 0, :, :, z_level].T,
    origin="lower",
    norm=plt.Normalize(vmin=0, vmax=1),
)
fig.colorbar(cax0, ax=axs[0])
axs[0].set_title("Density")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

# Plot 2: Vector magnitude (e.g., velocity magnitude)
cax1 = axs[1].imshow(
    jnp.sqrt(
        states[0, 1, :, :, z_level] ** 2
        + states[0, 2, :, :, z_level] ** 2
        + states[0, 3, :, :, z_level] ** 2
    ).T,
    origin="lower",
    norm=plt.Normalize(vmin=0, vmax=1),
)
fig.colorbar(cax1, ax=axs[1])
axs[1].set_title("Velocity Magnitude")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

# Plot 3: Optional third field, e.g., pressure or similar (reusing animate_vector logic)
cax2 = axs[2].imshow(
    states[0, 4, :, :, z_level].T,
    origin="lower",
    norm=plt.Normalize(vmin=0, vmax=1),
)
fig.colorbar(cax2, ax=axs[2])
axs[2].set_title("Pressure")
axs[2].set_xlabel("x")
axs[2].set_ylabel("y")

cax3 = axs[3].imshow(
    jnp.sqrt(
        states[0, 5, :, :, z_level] ** 2
        + states[0, 6, :, :, z_level] ** 2
        + states[0, 7, :, :, z_level] ** 2
    ).T,
    origin="lower",
    norm=plt.Normalize(vmin=0, vmax=1),
)

fig.colorbar(cax3, ax=axs[3])
axs[3].set_title("Magnetic field")
axs[3].set_xlabel("x")
axs[3].set_ylabel("y")


# Update function for all three plots
def animate_all(i):
    cax0.set_array(states[i, 0, :, :, z_level].T)
    cax1.set_array(
        jnp.sqrt(
            states[i, 1, :, :, z_level] ** 2
            + states[i, 2, :, :, z_level] ** 2
            + states[i, 3, :, :, z_level] ** 2
        ).T
    )
    cax2.set_array(states[i, 4, :, :, z_level].T)
    cax3.set_array(
        jnp.sqrt(
            states[i, 5, :, :, z_level] ** 2
            + states[i, 6, :, :, z_level] ** 2
            + states[i, 7, :, :, z_level] ** 2
        ).T
    )
    return cax0, cax1, cax2


ani = animation.FuncAnimation(fig, animate_all, frames=states.shape[0], interval=50)

ani.save("turbulence_sr/figures/turb_all_blast.gif")
plt.show()
