"""
Inspect all fields of the jf1uids SimulationConfig, SimulationParams,
and helper_data as used by turbulence_data_generation_all_states.py.

Reconstructs the exact config from config.yaml values + the hardcoded
settings in the generation script, then prints every field (including
defaults not explicitly set in the script).

Run:  conda run -n pytorch python data/inspect_simulation_config.py
"""

from autocvd import autocvd

autocvd(num_gpus=1)

import yaml
from pathlib import Path

config_file = yaml.safe_load(
    open(Path(__file__).resolve().parents[1] / "config.yaml", "r")
)

from jf1uids import SimulationConfig, SimulationParams, get_helper_data, get_registered_variables
from jf1uids.option_classes.simulation_config import FORWARDS, finalize_config

# ── Reconstruct the exact config from turbulence_data_generation_all_states.py ──
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
)

print("=" * 70)
print("SimulationConfig — all fields")
print("=" * 70)

if hasattr(config, "_fields"):
    # NamedTuple
    for f in config._fields:
        print(f"  {f:30s} = {getattr(config, f)!r}")
elif hasattr(config, "__dataclass_fields__"):
    # Dataclass
    for f in config.__dataclass_fields__:
        print(f"  {f:30s} = {getattr(config, f)!r}")
else:
    # Fallback: print all non-callable attrs
    for attr in sorted(dir(config)):
        if not attr.startswith("_") and not callable(getattr(config, attr)):
            print(f"  {attr:30s} = {getattr(config, attr)!r}")

print()

# ── SimulationParams ──
gamma = 5 / 3
C_CFL = 0.4
dt_max = 0.1
t_end = 1e4  # placeholder (real value needs code_units; just show structure)

params = SimulationParams(
    C_cfl=C_CFL,
    dt_max=dt_max,
    gamma=gamma,
    t_end=t_end,
)

print("=" * 70)
print("SimulationParams — all fields")
print("=" * 70)

if hasattr(params, "_fields"):
    for f in params._fields:
        print(f"  {f:30s} = {getattr(params, f)!r}")
elif hasattr(params, "__dataclass_fields__"):
    for f in params.__dataclass_fields__:
        print(f"  {f:30s} = {getattr(params, f)!r}")
else:
    for attr in sorted(dir(params)):
        if not attr.startswith("_") and not callable(getattr(params, attr)):
            print(f"  {attr:30s} = {getattr(params, attr)!r}")

print()

# ── helper_data ──
helper_data = get_helper_data(config)

print("=" * 70)
print("helper_data — structure")
print("=" * 70)
print(f"  type: {type(helper_data)}")

if hasattr(helper_data, "_fields"):
    for f in helper_data._fields:
        val = getattr(helper_data, f)
        if hasattr(val, "shape"):
            print(f"  {f:30s} = {type(val).__name__} shape={val.shape} dtype={val.dtype}")
        else:
            print(f"  {f:30s} = {val!r}")
elif hasattr(helper_data, "__dataclass_fields__"):
    for f in helper_data.__dataclass_fields__:
        val = getattr(helper_data, f)
        if hasattr(val, "shape"):
            print(f"  {f:30s} = {type(val).__name__} shape={val.shape} dtype={val.dtype}")
        else:
            print(f"  {f:30s} = {val!r}")
else:
    for attr in sorted(dir(helper_data)):
        if not attr.startswith("_") and not callable(getattr(helper_data, attr)):
            val = getattr(helper_data, attr)
            if hasattr(val, "shape"):
                print(f"  {attr:30s} = {type(val).__name__} shape={val.shape}")
            else:
                print(f"  {attr:30s} = {val!r}")

print()

# ── registered_variables ──
rv = get_registered_variables(config)

# ── finalize_config (as the generation script does at line 189) ──
# finalize_config computes grid_spacing correctly (box_size/num_cells).
# Before finalize_config, grid_spacing is a placeholder from get_helper_data.
import jax.numpy as jnp
import numpy as np

rho_shape = (5, config.num_cells, config.num_cells, config.num_cells)
finalized_config = finalize_config(config, rho_shape)

print("=" * 70)
print("SimulationConfig AFTER finalize_config — changed fields")
print("=" * 70)
for f in config._fields:
    before = getattr(config, f)
    after = getattr(finalized_config, f)
    if before != after:
        print(f"  {f:30s} = {after!r}   (was {before!r})")
print()
print(f"  grid_spacing after finalize_config = {finalized_config.grid_spacing}")
print(f"  1/num_cells                         = {1.0/config.num_cells}")

print("=" * 70)
print("registered_variables — all fields")
print("=" * 70)
if hasattr(rv, "_fields"):
    for f in rv._fields:
        print(f"  {f:30s} = {getattr(rv, f)!r}")
elif hasattr(rv, "__dataclass_fields__"):
    for f in rv.__dataclass_fields__:
        print(f"  {f:30s} = {getattr(rv, f)!r}")
else:
    for attr in sorted(dir(rv)):
        if not attr.startswith("_") and not callable(getattr(rv, attr)):
            print(f"  {attr:30s} = {getattr(rv, attr)!r}")
