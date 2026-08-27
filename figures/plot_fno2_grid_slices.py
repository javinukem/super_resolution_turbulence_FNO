"""
Plot HR/LR/SR slices for each model from the FNO2 grid experiment.

Similar to plot_hr_lr_sr_slices.py but tailored for the grid experiment
in experiments/train_fno2_grid_modes_interp_skip_losses_refine/.
"""

from autocvd import autocvd

autocvd(num_gpus=1)

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import json

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.models_fno_2 import FNO_2

from astropy import units as u
import astropy.constants as c
import jax.numpy as jnp
from jf1uids import CodeUnits, SimulationConfig, SimulationParams
from jf1uids import get_helper_data, get_registered_variables
from jf1uids.time_stepping.time_integration import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids.initial_condition_generation.turb import create_turb_field
from jf1uids.option_classes.simulation_config import FORWARDS, HLL, finalize_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the grid experiment output
GRID_EXPERIMENT_ROOT = Path(
    "/export/scratch/jalegria/experiments/train_fno2_grid_modes_interp_skip_losses_refine"
)

HR_NUM_CELLS = 128
BASE_UPSAMPLE_FACTOR = 4
SEED = 1234
SHIFTING_MODES = 8


# ── FNO2GridModel (same as in the training script) ────────────────────


def _interpolate_3d(x: torch.Tensor, scale_factor: int, mode: str) -> torch.Tensor:
    if mode in {"nearest", "area", "nearest-exact"}:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)


# Baseline FNO2 params
FNO2_BASE_PARAMS = dict(
    in_channel=5,
    n_channels=32,
    n_residual_blocks=3,
    n_operator_blocks=2,
    apply_constraint=False,
    last_layer_kernel=3,
)


class FNO2GridModel(nn.Module):
    """
    Experiment wrapper around FNO_2 with:
    - robust interpolation handling for nearest/trilinear
    - explicit skip connection on upsampled latent
    - tail variants: baseline conv ("mlp_tail") vs Conv3d→GELU→Conv3d refinement
    """

    def __init__(
        self,
        *,
        modes: int,
        interpolation_mode: str,
        skip_connection: bool,
        head_variant: str,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.interpolation_mode = interpolation_mode

        self.core = FNO_2(
            **FNO2_BASE_PARAMS,
            modes=modes,
            shifting_modes=SHIFTING_MODES,
            interpolation_mode=interpolation_mode,
            skip_connection=False,
        )

        if head_variant == "conv_refine_tail":
            self.core.tail = nn.Sequential(
                nn.Conv3d(
                    self.core.n_channels, self.core.n_channels, kernel_size=3, padding=1
                ),
                nn.GELU(),
                nn.Conv3d(
                    self.core.n_channels, self.core.in_channel, kernel_size=3, padding=1
                ),
            )
        elif head_variant != "mlp_tail":
            raise ValueError(f"Unsupported head variant: {head_variant}")

    def forward(
        self, x: torch.Tensor, upsample_factor: int = BASE_UPSAMPLE_FACTOR
    ) -> torch.Tensor:
        x1 = self.core.conv1(x)
        x2 = x1
        for layer in self.core.res_blocks:
            x2 = layer(x2)
        latent = x1 + x2

        upsampled = _interpolate_3d(
            latent, scale_factor=upsample_factor, mode=self.interpolation_mode
        )

        out = upsampled
        for layer in self.core.fno_blocks:
            out = layer(out)

        if self.skip_connection:
            out = out + upsampled

        out = self.core.tail(out)

        if self.core.apply_constraint:
            out = self.core.constraint(x, out, upsample_factor)

        out[:, 0] = self.core.tail_relu(out[:, 0])
        out[:, 4] = self.core.tail_relu(out[:, 4])
        return out


# ── Model loading ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    name: str
    folder_name: str
    modes: int
    interpolation_mode: str
    skip_connection: bool
    head_variant: str


def _load_model(run_dir: Path, spec: ModelSpec) -> nn.Module:
    """Load a FNO2GridModel from a run directory."""
    weights = run_dir / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = FNO2GridModel(
        modes=spec.modes,
        interpolation_mode=spec.interpolation_mode,
        skip_connection=spec.skip_connection,
        head_variant=spec.head_variant,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def _parse_model_name(folder_name: str) -> Optional[ModelSpec]:
    """Parse a model folder name into a ModelSpec."""
    # e.g. fno2_m16_shift8_interp-nearest_skip-0_head-mlp_tail_loss-mse
    try:
        parts = folder_name.split("_")
        modes = int(parts[1][1:])  # m16 -> 16

        interp_idx = next(i for i, p in enumerate(parts) if p.startswith("interp-"))
        interpolation_mode = parts[interp_idx].split("-")[1]

        skip_idx = next(i for i, p in enumerate(parts) if p.startswith("skip-"))
        skip_connection = parts[skip_idx].split("-")[1] == "1"

        head_idx = next(i for i, p in enumerate(parts) if p.startswith("head-"))
        # head could be "head-mlp_tail" or "head-conv_refine_tail"
        head_parts = []
        for i in range(head_idx, len(parts)):
            if parts[i].startswith("loss-"):
                break
            head_parts.append(parts[i])
        head_variant = "_".join(head_parts).replace("head-", "")

        return ModelSpec(
            name=folder_name,
            folder_name=folder_name,
            modes=modes,
            interpolation_mode=interpolation_mode,
            skip_connection=skip_connection,
            head_variant=head_variant,
        )
    except Exception as e:
        print(f"Failed to parse {folder_name}: {e}")
        return None


def _discover_models(experiment_dir: Path) -> list[tuple[Path, ModelSpec]]:
    """Discover all model folders in the experiment directory."""
    models = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.startswith("fno2_"):
            spec = _parse_model_name(item.name)
            if spec is not None and (item / "weights.pt").exists():
                models.append((item, spec))
    return sorted(models, key=lambda x: x[1].name)


def _resolve_experiment_dir() -> Path:
    """Find the latest experiment directory."""
    if not GRID_EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(f"Experiment root not found: {GRID_EXPERIMENT_ROOT}")

    candidates = sorted(
        [d for d in GRID_EXPERIMENT_ROOT.iterdir() if d.is_dir()],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No experiment directories found in {GRID_EXPERIMENT_ROOT}"
        )
    return candidates[0]


# ── Simulation helpers ────────────────────────────────────────────────


def _load_cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _downaverage_state(state: np.ndarray, downsample_factor: int) -> np.ndarray:
    channels, hx, hy, hz = state.shape
    if (
        hx % downsample_factor != 0
        or hy % downsample_factor != 0
        or hz % downsample_factor != 0
    ):
        raise ValueError(
            f"State spatial shape {(hx, hy, hz)} is not divisible by factor {downsample_factor}."
        )
    lx, ly, lz = (
        hx // downsample_factor,
        hy // downsample_factor,
        hz // downsample_factor,
    )
    reshaped = state.reshape(
        channels,
        lx,
        downsample_factor,
        ly,
        downsample_factor,
        lz,
        downsample_factor,
    )
    return reshaped.mean(axis=(2, 4, 6))


def _generate_hr_states(n_states: int, num_cells: int, seed: int) -> list[np.ndarray]:
    cfg_data = _load_cfg()["turbulent_sim"]
    np.random.seed(seed)

    config = SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=False,
        progress_bar=False,
        dimensionality=3,
        num_ghost_cells=int(cfg_data["num_ghost_cells"]),
        box_size=float(cfg_data["box_size"]),
        num_cells=num_cells,
        fixed_timestep=False,
        differentiation_mode=FORWARDS,
        riemann_solver=HLL,
        mhd=False,
        return_snapshots=True,
        num_snapshots=2,
    )
    helper_data = get_helper_data(config)
    reg_vars = get_registered_variables(config)

    code_units = CodeUnits(3 * u.parsec, 1 * u.M_sun, 100 * u.km / u.s)
    t_end = (1.0 * 1e4 * u.yr).to(code_units.code_time).value
    params = SimulationParams(
        C_cfl=0.4,
        dt_max=float(cfg_data["dt_max"]),
        gamma=5 / 3,
        t_end=t_end,
    )

    rho_0 = 2 * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B
    rho = (
        jnp.ones((num_cells, num_cells, num_cells))
        * rho_0.to(code_units.code_density).value
    )
    p = (
        jnp.ones((num_cells, num_cells, num_cells))
        * p_0.to(code_units.code_pressure).value
    )

    states: list[np.ndarray] = []
    attempts = 0
    while len(states) < n_states and attempts < n_states * 8:
        attempts += 1
        u_x = create_turb_field(
            num_cells,
            1,
            cfg_data["turbulence_slope"],
            cfg_data["kmin"],
            cfg_data["kmax"],
        )
        u_y = create_turb_field(
            num_cells,
            1,
            cfg_data["turbulence_slope"],
            cfg_data["kmin"],
            cfg_data["kmax"],
        )
        u_z = create_turb_field(
            num_cells,
            1,
            cfg_data["turbulence_slope"],
            cfg_data["kmin"],
            cfg_data["kmax"],
        )

        rms = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))
        if not jnp.isfinite(rms) or float(rms) == 0.0:
            continue
        wanted_rms = (
            (float(cfg_data["wanted_rms"]) * u.km / u.s)
            .to(code_units.code_velocity)
            .value
        )
        u_x = u_x / rms * wanted_rms
        u_y = u_y / rms * wanted_rms
        u_z = u_z / rms * wanted_rms

        initial_state = construct_primitive_state(
            config=config,
            registered_variables=reg_vars,
            density=rho,
            velocity_x=u_x,
            velocity_y=u_y,
            velocity_z=u_z,
            gas_pressure=p,
        )
        config_run = finalize_config(config, initial_state.shape)
        result = time_integration(
            initial_state, config_run, params, helper_data, reg_vars
        )
        snapshot = np.array(result.states[-1], dtype=np.float32)
        if not np.isfinite(snapshot).all():
            continue
        states.append(snapshot)
    if len(states) != n_states:
        raise RuntimeError(
            f"Could not generate {n_states} finite turbulent HR states, got {len(states)}."
        )
    return states


# ── Field extraction and plotting ─────────────────────────────────────


def _extract_fields(state: np.ndarray) -> dict[str, np.ndarray]:
    rho = state[0]
    vx, vy, vz = state[1], state[2], state[3]
    pressure = state[4]
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    dvx_dz = np.gradient(vx, axis=0)
    dvx_dy = np.gradient(vx, axis=1)
    dvy_dz = np.gradient(vy, axis=0)
    dvy_dx = np.gradient(vy, axis=2)
    dvz_dy = np.gradient(vz, axis=1)
    dvz_dx = np.gradient(vz, axis=2)
    omega_x = dvz_dy - dvy_dz
    omega_y = dvx_dz - dvz_dx
    omega_z = dvy_dx - dvx_dy
    vort = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-12)
    return {"rho": rho, "p": pressure, "v": vmag, "vorticity": vort}


def _run_model(model: nn.Module, lr: np.ndarray, scale: int) -> np.ndarray:
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sr = model(lr_t, upsample_factor=scale)
    return sr.squeeze(0).detach().cpu().numpy()


def _plot_triplet_rows(
    lr_state: np.ndarray,
    hr_state: np.ndarray,
    sr_state: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    cols = ["rho", "p", "v", "vorticity"]
    rows = [("LR", lr_state), ("HR", hr_state), ("SR", sr_state)]
    field_cache = {
        "LR": _extract_fields(lr_state),
        "HR": _extract_fields(hr_state),
        "SR": _extract_fields(sr_state),
    }

    for row_idx, (row_name, state) in enumerate(rows):
        for col_idx, field in enumerate(cols):
            fields = [
                field_cache["LR"][field],
                field_cache["HR"][field],
                field_cache["SR"][field],
            ]
            vmin = min(float(np.nanmin(f)) for f in fields if np.isfinite(f).any())
            vmax = max(float(np.nanmax(f)) for f in fields if np.isfinite(f).any())
            z_idx = state.shape[-1] // 2
            img = axes[row_idx, col_idx].imshow(
                field_cache[row_name][field][:, :, z_idx].T,
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(field)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(row_name)
            fig.colorbar(img, ax=axes[row_idx, col_idx], fraction=0.045, pad=0.02)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(SEED)

    experiment_dir = _resolve_experiment_dir()
    print(f"Using experiment directory: {experiment_dir}")

    models = _discover_models(experiment_dir)
    print(f"Found {len(models)} model(s)")

    if not models:
        print("No models found. Exiting.")
        return

    print("Generating HR state...")
    hr_state = _generate_hr_states(n_states=1, num_cells=HR_NUM_CELLS, seed=SEED)[0]
    print(f"HR state shape: {hr_state.shape}")

    for model_dir, spec in models:
        print(f"\nProcessing: {spec.name}")

        try:
            model = _load_model(model_dir, spec)
        except FileNotFoundError as exc:
            print(f"  Skipping: {exc}")
            continue

        # Test scales: 4 (native) and 2
        scales = [4, 2]
        for scale in scales:
            hr_target = hr_state
            if scale == 2:
                hr_target = (
                    F.interpolate(
                        torch.from_numpy(hr_state).unsqueeze(0),
                        scale_factor=0.5,
                        mode="trilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .numpy()
                )

            lr_state = _downaverage_state(
                hr_state, downsample_factor=BASE_UPSAMPLE_FACTOR
            )
            sr_state = _run_model(model, lr_state, scale=scale)

            out_name = f"hr_lr_sr_slices_x{scale}.png"
            out_path = model_dir / out_name

            # Create a shorter title
            short_title = f"m{spec.modes} | {spec.interpolation_mode} | skip={int(spec.skip_connection)} | {spec.head_variant} | x{scale}"

            _plot_triplet_rows(
                lr_state=lr_state,
                hr_state=hr_target,
                sr_state=sr_state,
                save_path=out_path,
                title=short_title,
            )
            print(f"  Saved: {out_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
