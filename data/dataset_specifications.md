# Dataset Specifications — `training_best_models_experiment`

This document is the canonical specification of the dataset consumed by
`experiments/training_best_models_experiment/train_best_models.py`: where it
lives, how it was split, how the dataloader reads it, and how it was originally
generated with **jf1uids** (the JAX Euler solver). It consolidates information
from `train_best_models.py`, `src/dataloader/dataloader_3d.py`,
`data/split_dataset.py`, the legacy top-level `config.yaml`, and the generation
script `src/dataset_generation/turbulence_data_generation_all_states.py`.

> Note on conventions: per `AGENTS.md`, the channel order of every state is the
> jf1uids primitive-state order. In 3-D this is 5 channels
> `[density, vx, vy, vz, pressure]`. Density (ch 0) and pressure (last) are
> positivity-constrained.

---

## 1. Overview

| Item | Value |
|------|-------|
| Experiment | `experiments/training_best_models_experiment/` |
| Training entrypoint | `experiments/training_best_models_experiment/train_best_models.py` |
| Dataset format | HDF5, 3-D compressible-turbulence states |
| Solver used to generate data | **jf1uids** (JAX) — Euler equations, run via `src/dataset_generation/turbulence_data_generation_all_states.py` |
| ML side | PyTorch only (`dataset_sr` dataloader, FNO_2 / EDSR models) |
| Grid (HR) | 128³ cells, 5 channels |
| Grid (LR) | 32³ cells, 5 channels (downsample factor 4) |
| Total generated | 500 simulations × 80 snapshots = 40 000 samples |
| Split scheme | **Simulation-level** contiguous 80/20 split |
| Train file | `/export/scratch/jalegria/full_states_h5/full_states.h5` |
| Validation file | `/export/scratch/jalegria/full_states_h5/full_states_val.h5` |
| Source (pre-split) file | `/export/data/jalegria/full_states_h5/full_states.h5` |

---

## 2. HDF5 format

### Keys, shapes, dtype

All four datasets are `float32`. Confirmed by reading the actual files on disk
(not just from the generation script):

```python
# /export/data/jalegria/full_states_h5/full_states.h5  (source, pre-split)
first_snapshot_energy  (40000,)                float32
first_snapshot_mass    (40000,)                float32
hr_states              (40000, 5, 128, 128, 128) float32
lr_states              (40000, 5, 32, 32, 32)    float32

# /export/scratch/jalegria/full_states_h5/full_states.h5  (train split)
first_snapshot_energy  (32000,)                float32
first_snapshot_mass    (32000,)                float32
hr_states              (32000, 5, 128, 128, 128) float32
lr_states              (32000, 5, 32, 32, 32)    float32

# /export/scratch/jalegria/full_states_h5/full_states_val.h5  (val split)
first_snapshot_energy  (8000,)                 float32
first_snapshot_mass    (8000,)                 float32
hr_states              (8000, 5, 128, 128, 128)  float32
lr_states              (8000, 5, 32, 32, 32)     float32
```

### File paths and on-disk sizes

| Path | Role | Size |
|------|------|------|
| `/export/data/jalegria/full_states_h5/full_states.h5` | Source full dataset (500 sims × 80 snapshots) | 1.6 TB |
| `/export/scratch/jalegria/full_states_h5/full_states.h5` | Train split (sims 0–399, 32 000 samples) | 1.3 TB |
| `/export/scratch/jalegria/full_states_h5/full_states_val.h5` | Val split (sims 400–499, 8 000 samples) | 318 GB |

### Sample layout

Samples are stored simulation-major: index `i * 80 + s` is the `s`-th snapshot
of simulation `i` (0 ≤ i < num_sims, 0 ≤ s < 80). This is the layout assumed by
`dataset_sr` (it hardcodes `snapshots_per_sim = 80`) and by `split_dataset.py`.

### Key semantics

- `hr_states` — high-resolution primitive state, 128³ × 5 channels.
- `lr_states` — low-resolution primitive state, 32³ × 5 channels, obtained by
  convolving the HR state with a 3³ box kernel (mean filter) of weight 1/27 and
  striding by 4 (`scipy.ndimage.convolve` then `[::4, ::4, ::4]`).
- `first_snapshot_energy` — total energy of the **first** snapshot of the
  simulation (scalar per sample; identical across the 80 rows of one sim).
- `first_snapshot_mass` — total mass of the first snapshot (scalar per sample;
  identical across the 80 rows of one sim).

---

## 3. Dataset split (`data/split_dataset.py`)

`data/split_dataset.py` performs a **simulation-level, contiguous, deterministic
80/20 split** of the source HDF5 into dedicated train/val HDF5 files. There is
**no random shuffling and no RNG seed** — simulations 0–399 go to training and
400–499 go to validation. (The `data/rng_seeds_160_40.pkl` file in the same
folder is **not** used by this script; it belongs to the *sample-level* random
split used by the Hydra pipeline, a different scheme — see `AGENTS.md`.)

### Hardcoded parameters

```python
SRC_DIR          = /export/data/jalegria/full_states_h5
OUT_DIR          = /export/scratch/jalegria/full_states_h5
ORIGINAL         = SRC_DIR / "full_states.h5"
TRAIN_OUT        = OUT_DIR / "full_states.h5"
VAL_OUT          = OUT_DIR / "full_states_val.h5"
TOTAL_SIMS       = 500
SNAPSHOTS_PER_SIM = 80
TRAIN_SIMS       = 400      # 80 %
CHUNK_SIZE       = 80       # copy one simulation at a time
```

There are no CLI (argparse) arguments; everything is hardcoded.

### What it does

1. Verifies `ORIGINAL` exists and that `f["hr_states"].shape[0] == 40000`
   (asserts `500 × 80 == 40000`).
2. Refuses to overwrite if both `TRAIN_OUT` and `VAL_OUT` already exist
   (prints a message and exits).
3. Writes `TRAIN_OUT` by copying rows `[0, 32000)` of **every key** from the
   source, preserving dtype and shape (with axis-0 reduced to 32 000).
4. Writes `VAL_OUT` by copying rows `[32000, 40000)` of every key (axis-0
   reduced to 8 000).
5. Copies are done one simulation at a time (`CHUNK_SIZE = 80` rows) to bound
   memory. No preprocessing / normalization is applied — the data is copied
   verbatim.

> The docstring of `split_dataset.py` mentions renaming the original file to
> `full_states_original.h5`; the actual code does **not** perform that rename —
> the original file remains at `/export/data/.../full_states.h5`.

### Sample-level vs simulation-level

This is a **simulation-level** split (whole simulations are held out). The
Hydra training pipeline (`src/utils/pipeline.py::build_train_test_loader`) uses
a different, **sample-level** `random_split` over snapshots. The two schemes
are not equivalent; treat the difference as a known limitation (documented in
`experiments/experiment_summary.md`). The `training_best_models_experiment`
uses the simulation-level files produced by `split_dataset.py`.

---

## 4. Dataloader (`src/dataloader/dataloader_3d.py::dataset_sr`)

Imported in `train_best_models.py` as:

```python
from src.dataloader.dataloader_3d import dataset_sr
```

### `__init__` signature and defaults

```python
class dataset_sr(Dataset):
    def __init__(
        self,
        h5_path = Path(__file__).resolve().parents[6]
                  / "data/jalegria/full_states_h5/full_states.h5",
        snapshot_index: Optional[int] = None,
        max_samples = None,
        use_printing = False,
        use_normalizing = False,
        only_normalize_lr = False,
        means = None,
        stds = None,
    ):
```

- **Default `h5_path`**: resolves to
  `/export/data/jalegria/full_states_h5/full_states.h5`
  (`parents[6]` of `src/dataloader/dataloader_3d.py` is `/export`). The
  experiment overrides this explicitly with the `/export/scratch/...` paths.
- The HDF5 file is opened once (`h5py.File(..., "r")`) and kept open for the
  life of the object; `__del__` closes it. Data is read lazily — nothing is
  loaded into memory at construction time except the index list.

### `snapshot_index` semantics

`dataset_sr` hardcodes `self.snapshots_per_sim = 80` and computes
`total_sims = len(hr_states) // 80`.

- **`snapshot_index = None`** (default) → use **all** snapshots:
  `indices = list(range(total_snapshots))` (e.g. 32 000 for the train file,
  8 000 for the val file).
- **`snapshot_index = k`** (an int, 0 ≤ k < 80) → pick snapshot `k` of **each**
  simulation: `indices = [i * 80 + k for i in range(total_sims)]`
  (e.g. 400 for the train file, 100 for the val file).
- **`snapshot_index = 79`** → the *last* statistically-stationary snapshot of
  each simulation. This is the value the experiment summary claims to use
  (see §7 for the discrepancy with the actual script).

`max_samples` truncates `indices` from the front if set.

### `__getitem__` return tuple

Returns a 4-tuple `(hr, lr, energy, mass)`:

| Position | Name    | Type                  | Shape (3-D)            |
|----------|---------|-----------------------|------------------------|
| 0        | `hr`    | `torch.float32` tensor| `(5, 128, 128, 128)`   |
| 1        | `lr`    | `torch.float32` tensor| `(5, 32, 32, 32)`      |
| 2        | `energy`| scalar (HDF5 read)    | `()`                   |
| 3        | `mass`  | scalar (HDF5 read)    | `()`                   |

When `use_normalizing=True`, HR and/or LR are standardized with per-channel
mean/std (`(x - mean[:, None, None, None]) / std[:, None, None, None]`); means
and stds are either passed in or computed over the active indices via
`utils.mean_std.compute_mean_std_dataset`. The experiment does **not** enable
normalization.

### Expected HDF5 keys

`hr_states`, `lr_states`, `first_snapshot_energy`, `first_snapshot_mass`
(opened in `__init__` as `self.hr_states`, `self.lr_states`, `self.energy`,
`self.mass`).

---

## 5. Dataset generation (jf1uids)

The data was generated by
`src/dataset_generation/turbulence_data_generation_all_states.py`, which reads
the **legacy top-level `config.yaml`** (not the Hydra tree). This is the script
that produces an HDF5 file with exactly the four keys used above (3-D, 5
channels).

> **Known legacy bug:** the script opens the config with
> `open("home/jalegria/Thesis/turbulence_sr/config.yaml", "r")` — missing the
> leading `/` — and writes to a relative `./data/jalegria/full_states_h5`.
> These are broken unless run with a specific CWD workaround (per `AGENTS.md`).
> The presence of the file at `/export/data/...` indicates it was run with such
> a workaround (or an earlier version of the script).

### Parameters from `config.yaml`

#### `turbulent_sim:` block (the physics/simulation parameters)

| Parameter            | Value   | Meaning |
|----------------------|---------|---------|
| `runtime_debug`      | `False` | jf1uids runtime debugging |
| `first_order_fb`     | `False` | first-order fallback disabled |
| `progress_bar`       | `False` | no progress bar |
| `dimensionality`     | `3`     | 3-D simulation |
| `num_ghost_cells`    | `2`     | ghost-cell halo thickness |
| `box_size`           | `1.0`   | simulation box size (code units) |
| `num_cells`          | `128`   | HR grid: 128³ |
| `fixed_timestep`     | `False` | adaptive (CFL) time stepping |
| `return_snapshots`   | `True`  | return the time series of states |
| `num_snapshots`      | `100`   | requested snapshots per sim (see discrepancy §8) |
| `turbulence_slope`   | `-2`    | Kolmogorov-like power spectrum slope |
| `kmin`               | `2`     | minimum forcing wavenumber |
| `kmax`               | `64`    | maximum forcing wavenumber |
| `max_sims`           | `500`   | number of simulations |
| `gamma`              | `5/3`   | adiabatic index (ideal monatomic gas) |
| `dt_max`             | `0.1`   | maximum allowed timestep |
| `wanted_rms`         | `50`    | target RMS velocity (km/s; applied as `50 km/s` in script) |

#### `data:` block

| Parameter          | Value | Meaning |
|--------------------|-------|---------|
| `upsample_factor`  | `4`   | HR→LR downsample (stride) factor → 128³ → 32³ |

#### Full `SimulationConfig` (all fields, including defaults)

The table below lists **every** field of the `SimulationConfig` namedtuple as
instantiated by `turbulence_data_generation_all_states.py` (lines 68–80), including
fields left at their jf1uids defaults. Values were obtained by running
`data/inspect_simulation_config.py`, which reconstructs the exact config from
`config.yaml` + the hardcoded settings in the generation script and prints all
fields via reflection.

| Field                         | Value                                   | Set by         | Meaning |
|-------------------------------|-----------------------------------------|----------------|---------|
| `runtime_debugging`           | `False`                                 | `config.yaml`  | jf1uids runtime debugging |
| `progress_bar`                | `False`                                 | `config.yaml`  | no progress bar |
| `dimensionality`              | `3`                                     | `config.yaml`  | 3-D simulation |
| `geometry`                    | `0`                                     | default        | Cartesian geometry (jf1uids enum) |
| `mhd`                         | `False`                                 | default        | magnetohydrodynamics disabled (pure hydro) |
| `self_gravity`                | `False`                                 | default        | self-gravity disabled |
| `box_size`                    | `1.0`                                   | `config.yaml`  | simulation box size (code units) |
| `num_cells`                   | `128`                                   | `config.yaml`  | HR grid: 128³ |
| `reconstruction_order`        | `1`                                     | default        | first-order spatial reconstruction |
| `limiter`                     | `0`                                     | default        | slope limiter index (jf1uids enum) |
| `riemann_solver`              | `1`                                     | default        | Riemann solver index (jf1uids enum; 1 = HLL/HLLC family) |
| `num_ghost_cells`             | `2`                                     | `config.yaml`  | ghost-cell halo thickness |
| `grid_spacing`                | `0.007874015748031496` (= 1/127)        | derived (`finalize_config`) | cell size in code units; computed by `finalize_config` as `box_size / (num_cells - 1)` — before `finalize_config` this field holds a placeholder (`1/399`), so it must be read **after** finalization |
| `boundary_settings`           | `None` → populated by `finalize_config` | default → derived | `None` before finalization; after `finalize_config`: `BoundarySettings(x=BoundarySettings1D(left_boundary=0, right_boundary=0), y=..., z=...)` — boundary enum `0` on all faces (jf1uids default, periodic-style) |
| `fixed_timestep`              | `False`                                 | `config.yaml`  | adaptive (CFL) time stepping |
| `exact_end_time`              | `False`                                 | default        | do not force integration to land exactly on `t_end` |
| `source_term_aware_timestep`  | `False`                                 | default        | no source-term-aware dt reduction |
| `num_timesteps`               | `1000`                                  | default        | max number of timesteps (safety cap; unused since `fixed_timestep=False`) |
| `differentiation_mode`        | `0` (`FORWARDS`)                        | script         | forwards-in-time integration |
| `num_checkpoints`             | `100`                                   | default        | number of checkpoint slots |
| `return_snapshots`            | `True`                                  | `config.yaml`  | return the time series of states |
| `activate_snapshot_callback`  | `False`                                 | default        | no per-snapshot callback |
| `num_snapshots`               | `100`                                   | `config.yaml`  | requested snapshots per sim (see discrepancy §8 — on-disk data has 80) |
| `first_order_fallback`        | `False`                                 | `config.yaml`  | first-order fallback disabled |
| `wind_config`                 | `WindConfig(stellar_wind=False, num_injection_cells=10, wind_injection_scheme=1, trace_wind_density=False)` | default | stellar wind disabled |
| `cosmic_ray_config`           | `CosmicRayConfig(cosmic_rays=False, diffusive_shock_acceleration=False)` | default | cosmic rays disabled |

> **Note on `grid_spacing`:** `finalize_config` (called at line 189 of the
> generation script, after the initial state is constructed) computes
> `grid_spacing = box_size / (num_cells - 1) = 1.0 / 127 ≈ 0.007874` (code
> units). Before `finalize_config` runs, this field holds a placeholder value
> (`1/399`) and must not be used. The physical cell size used for
> Reynolds-number analysis is `3 pc / 128 = 0.0234 pc = 7.23×10¹⁶ cm`
> (see `data/reynolds_number.md`), i.e. the naive `box_size/num_cells` —
> jf1uids uses `num_cells - 1` intervals between 128 cell centers.
>
> **`boundary_settings`** is `None` until `finalize_config` populates it;
> after finalization it is `BoundarySettings` with `left_boundary=0,
> right_boundary=0` on all three axes (jf1uids default boundary enum).

#### Full `SimulationParams` (all fields, including defaults)

| Field                         | Value          | Set by         | Meaning |
|-------------------------------|----------------|----------------|---------|
| `C_cfl`                       | `0.4`          | script         | CFL coefficient |
| `gravitational_constant`      | `1.0`          | default        | gravitational constant (code units; unused since `self_gravity=False`) |
| `gamma`                       | `1.6667` (5/3) | script         | adiabatic index (ideal monatomic gas) |
| `dt_max`                      | `0.1`          | script         | maximum allowed timestep (code units) |
| `t_end`                       | `10000.0`      | script         | final time (code units; = 10⁴ yr converted via `CodeUnits`) |
| `wind_params`                 | `WindParams(wind_mass_loss_rate=0.0, wind_final_velocity=0.0, pressure_floor=100000.0)` | default | stellar wind parameters (unused since `stellar_wind=False`) |
| `cosmic_ray_params`           | `CosmicRayParams(diffusive_shock_acceleration_start_time=0.0, diffusive_shock_acceleration_efficiency=0.1)` | default | cosmic-ray DSA parameters (unused since `cosmic_rays=False`) |

#### `helper_data` (generated by `get_helper_data(config)`)

| Field                  | Type        | Shape              | Meaning |
|------------------------|-------------|--------------------|---------|
| `geometric_centers`    | `ArrayImpl` | `(128, 128, 128, 3)` | geometric cell-center coordinates |
| `volumetric_centers`   | `ArrayImpl` | `(128, 128, 128, 3)` | volumetric cell-center coordinates |
| `r`                    | `ArrayImpl` | `(128, 128, 128)`    | radial distance from origin per cell |
| `r_hat_alpha`          | `None`      | —                  | (unused for Cartesian) |
| `cell_volumes`         | `None`      | —                  | (unused for uniform Cartesian grid) |
| `inner_cell_boundaries`| `None`      | —                  | (unused for uniform Cartesian grid) |
| `outer_cell_boundaries`| `None`      | —                  | (unused for uniform Cartesian grid) |

#### `registered_variables` (generated by `get_registered_variables(config)`)

| Field                  | Value                | Meaning |
|------------------------|----------------------|---------|
| `num_vars`             | `5`                  | total primitive variables |
| `density_index`        | `0`                  | density is channel 0 |
| `velocity_index`       | `StaticIntVector(x=1, y=2, z=3)` | velocity channels 1, 2, 3 |
| `magnetic_index`       | `-1`                 | no magnetic field (`mhd=False`) |
| `pressure_index`       | `4`                  | pressure is channel 4 |
| `wind_density_index`   | `-1`                 | no wind tracer (`stellar_wind=False`) |
| `wind_density_active`  | `False`              | wind tracer inactive |
| `cosmic_ray_n_index`   | `-1`                 | no cosmic-ray number density (`cosmic_rays=False`) |
| `cosmic_ray_n_active`  | `False`              | cosmic-ray field inactive |

This confirms the channel order `[density, vx, vy, vz, pressure]` (5 channels)
documented in `AGENTS.md`, with no MHD, wind, or cosmic-ray tracers.

### Physics and unit system (from the generation script, lines 86–112)

```python
code_length   = 3 * u.parsec
code_mass     = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units    = CodeUnits(code_length, code_mass, code_velocity)

C_CFL       = 0.4
t_final     = 1.0e4 * u.yr            # 10 000 years
t_end       = t_final.to(code_units.code_time).value
gamma       = 5/3
wanted_rms  = 50 * u.km / u.s
dt_max      = 0.1

params = SimulationParams(C_cfl=C_CFL, dt_max=dt_max, gamma=gamma, t_end=t_end)
```

So the physical box is **3 pc** on a side, mass unit **1 M☉**, velocity unit
**100 km/s**, integrated for **10⁴ yr** with CFL coefficient **0.4** and a
maximum timestep of **0.1** (code units). `differentiation_mode = FORWARDS`.

### Homogeneous background state (lines 114–132)

```python
rho_0 = 2 * c.m_p / u.cm**3                          # 2 proton masses per cm³
p_0  = 3e4 * u.K / u.cm**3 * c.k_B                   # n k_B T with n in cm⁻³, T=3e4 K
```

Density and pressure are initialized as uniform fields (filled with `rho_0`
and `p_0` converted to code units); the turbulence is injected only through the
velocity field (see §6).

### Boundary conditions

Not explicitly set in the script beyond `num_ghost_cells = 2`; the downsampling
kernel uses `scipy.ndimage.convolve(..., mode="reflect", cval=0.0)`. Boundary
conditions for the jf1uids integration itself come from `SimulationConfig` /
`get_helper_data` defaults (periodic-style halo handling internal to jf1uids).

kitty +kitten ssh --help
LR is produced by convolving each channel with a 3³ mean kernel (weight 1/27)
and then striding by `upsample_factor = 4`, taking every 4th cell. Output shape
per channel: `128 // 4 = 32`.

### Time stepping

Adaptive CFL (`fixed_timestep = False`) with `C_CFL = 0.4`, `dt_max = 0.1`,
integrated to `t_end = 1e4 yr` (in code-time units). Snapshots are returned
evenly over the integration (`return_snapshots = True`).

---

## 6. Initial state creation

For each simulation (loop at lines 165–209 of
`turbulence_data_generation_all_states.py`):

1. **Three independent turbulent velocity fields** are drawn, one per component,
   by calling jf1uids' `create_turb_field`:
   ```python
   u_x = create_turb_field(num_cells=128, seed=1, slope=-2, kmin=2, kmax=64)
   u_y = create_turb_field(num_cells=128, seed=1, slope=-2, kmin=2, kmax=64)
   u_z = create_turb_field(num_cells=128, seed=1, slope=-2, kmin=2, kmax=64)
   ```
   `create_turb_field` builds a divergence-free-ish velocity realization with a
   power spectrum `P(k) ∝ k^{slope}` (here slope `-2`, i.e. Kolmogorov-like)
   over the wavenumber band `[kmin, kmax] = [2, 64]`.

   > **Realization seed:** the literal `1` is passed as the `seed` argument on
   > every call. jf1uids' `create_turb_field` is expected to derive a different
   > realization per call from internal state / a per-call RNG draw; the
   > dataset clearly has distinct fields per simulation (otherwise all sims
   > would be identical). The exact per-simulation seeding mechanism is
   > internal to jf1uids and not controlled by this script — i.e. each
   > simulation gets a different random realization, but **how** the seed
   > varies is not visible in the generation script.

2. **Scale to target RMS velocity.** The raw field is normalized then rescaled
   so the volume RMS of `sqrt(u_x² + u_y² + u_z²)` equals `wanted_rms` (50 km/s
   in code-velocity units):
   ```python
   rms_vel = sqrt(mean(u_x**2 + u_y**2 + u_z**2))
   if not finite(rms_vel) or rms_vel == 0: skip and retry
   u_x = u_x / rms_vel * wanted_rms_code
   u_y = u_y / rms_vel * wanted_rms_code
   u_z = u_z / rms_vel * wanted_rms_code
   ```
   Sims with non-finite or zero `rms_vel` are skipped (not counted toward
   `max_sims`).

3. **Build the primitive state** with
   `jf1uids.fluid_equations.fluid.construct_primitive_state`:
   ```python
   initial_state = construct_primitive_state(
       config=config, registered_variables=registered_variables,
       density=rho,            # uniform 2 m_p/cm³
       velocity_x=u_x,
       velocity_y=u_y,
       velocity_z=u_z,
       gas_pressure=p,         # uniform, n k_B T with T = 3e4 K
   )
   ```

4. **Finalize the config and integrate:**
   ```python
   config = finalize_config(config, initial_state.shape)
   result = time_integration(initial_state, config, params, helper_data, registered_variables)
   ```

5. **Discard failed sims.** If `result.states[-1]` is all zeros the simulation
   is skipped (not counted toward `max_sims`).

6. **Downsample and store.** `convolve_lr(np.array(result.states), 4)` produces
   the LR snapshots; each of the returned HR snapshots and its LR counterpart
   are written at row `snapshot_counter`, and `first_snapshot_energy` /
   `first_snapshot_mass` are filled with `result.total_energy[0]` and
   `result.total_mass[0]` (the *first* snapshot's totals — hence the key name).

---

## 7. Training-time data usage (`train_best_models.py`)

### Constants (top of file, lines 30–53)

```python
DEVICE          = cuda if available else cpu
TRAIN_H5        = Path("/export/scratch/jalegria/full_states_h5/full_states.h5")
VAL_H5          = Path("/export/scratch/jalegria/full_states_h5/full_states_val.h5")
OUTPUT_ROOT     = Path("/export/scratch/jalegria/experiments")

SNAPSHOT_INDEX  = None        # <-- see discrepancy §8
UPSAMPLE_FACTOR = 4

EPOCHS              = 100
LEARNING_RATE       = 0.001
WEIGHT_DECAY        = 1e-4
BATCH_SIZE          = 8
NOISE_STD           = 0.01
NUM_WORKERS         = 2
USE_AMP             = False
GRAD_ACCUM          = 8
EARLY_STOP_PATIENCE = 30
```

### Dataset / dataloader instantiation (lines 337–359)

```python
train_ds = dataset_sr(h5_path=TRAIN_H5, snapshot_index=SNAPSHOT_INDEX)
val_ds   = dataset_sr(h5_path=VAL_H5,   snapshot_index=SNAPSHOT_INDEX)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True)
```

No normalization, no `max_samples`, no custom means/stds.

### Data augmentation

Gaussian noise on the **LR input only**, applied per-batch inside the training
loop (line 231):

```python
lr = lr + torch.randn_like(lr) * NOISE_STD     # NOISE_STD = 0.01
```

No augmentation on HR targets; no noise at validation time.

### Effective batch size

Optimizer steps occur every `GRAD_ACCUM = 8` mini-batches (or at the end of
the epoch). With `BATCH_SIZE = 8` and `GRAD_ACCUM = 8`, the effective batch
size is **64**. (The inline comment on line 52 says `= 16`; that comment is
stale/incorrect — see §8.)

### What each model receives

- CFNO variants: `_forward(model, lr, "cfno")` calls `model(lr, upsample_factor=4)`.
- EDSR: `_forward(model, lr, "edsr")` calls `model(lr)`.

Both consume `lr` of shape `(B, 5, 32, 32, 32)` and produce `(B, 5, 128, 128, 128)`.

---

## 8. Discrepancies and gaps

These are worth flagging because they affect reproducibility.

1. **`SNAPSHOT_INDEX` in the script does not match the experiment summary.**
   `train_best_models.py` sets `SNAPSHOT_INDEX = None` (line 40) with a
   comment `# last snapshot per simulation only`. But `None` means **all
   snapshots**, not the last one. `experiment_summary.md` states
   `snapshot_index=79` (the last snapshot) and "400 / 100 samples". With the
   script as written, the train set is **32 000** samples and the val set is
   **8 000** samples (all 80 snapshots per sim). Either the summary is stale
   or the script was edited after the documented runs. To actually use only
   the last snapshot, set `SNAPSHOT_INDEX = 79`.

2. **Other mismatches between `train_best_models.py` and `experiment_summary.md`:**
   | Setting | `train_best_models.py` | `experiment_summary.md` |
   |---------|------------------------|--------------------------|
   | Learning rate | `0.001` | `0.0002` |
   | Batch size | `8` | `2` |
   | AMP | `False` | "Enabled" |
   | Snapshot | `None` (all) | `79` (last only) |
   | Effective batch | `8 × 8 = 64` | n/a |

   The summary appears to describe an earlier configuration; trust the script
   for the current run.

3. **Stale `GRAD_ACCUM` comment.** Line 52: `# effective batch size = BATCH_SIZE × GRAD_ACCUM = 16`
   but `8 × 8 = 64`.

4. **`config.yaml` `num_snapshots: 100` vs actual 80 on disk.** The generation
   script sizes the HDF5 datasets at `max_sims * config.num_snapshots`
   = `500 × 100 = 50 000` rows, but the actual source file has
   `hr_states.shape = (40000, ...)` — i.e. **80 snapshots per sim**. The
   dataloader (`snapshots_per_sim = 80`) and `split_dataset.py`
   (`SNAPSHOTS_PER_SIM = 80`, asserts 40 000) both assume 80. Either the data
   was generated with `num_snapshots = 80` and `config.yaml` was later bumped
   to 100, or jf1uids returned 80 snapshots despite the request for 100. In
   any case the on-disk truth is **80 snapshots/sim**, and `config.yaml`'s
   `100` does not describe the existing dataset.

5. **Generation script is broken as committed.** `open("home/jalegria/...")`
   (missing leading `/`) and the relative `./data/jalegria/full_states_h5`
   save path mean the script cannot reproduce the dataset as-is; it requires a
   CWD workaround or an older version. Treat the generation scripts as
   legacy (per `AGENTS.md`).

6. **`rng_seeds_160_40.pkl` is unrelated to this experiment.** It belongs to
   the sample-level random split in the Hydra pipeline, not to
   `split_dataset.py` (which is deterministic and unseeded).

7. **Per-simulation RNG seeding is internal to jf1uids.** The generation
   script passes the literal seed `1` to every `create_turb_field` call; the
   fact that the 500 simulations are distinct realizations comes from
   jf1uids-internal state, not from anything controlled by the script. The
   exact mechanism is not visible here.

---

## 9. Quick reproduction recipe

```bash
# 1. (Already done) Generate the source dataset with jf1uids — legacy script,
#    needs CWD workaround:
#    python src/dataset_generation/turbulence_data_generation_all_states.py
#    → produces /export/data/jalegria/full_states_h5/full_states.h5 (40 000 samples)

# 2. Produce the simulation-level 80/20 split:
python data/split_dataset.py
#    → /export/scratch/jalegria/full_states_h5/full_states.h5      (32 000)
#    → /export/scratch/jalegria/full_states_h5/full_states_val.h5  (8 000)

# 3. Train the best-models suite:
python experiments/training_best_models_experiment/train_best_models.py
```
