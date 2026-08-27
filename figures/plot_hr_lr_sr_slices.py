import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# from autocvd import autocvd

# autocvd(num_gpus=1)
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.models_edsr import EDSR
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
BEST_MODELS_DIR: Path = Path(
    "/export/scratch/jalegria/experiments/train_fno2_grid_modes_interp_skip_losses_refine"
)
HR_NUM_CELLS = 128
BASE_UPSAMPLE_FACTOR = 4
SEED = 1234


@dataclass(frozen=True)
class ModelSpec:
    name: str
    supports_variable_scale: bool
    folder_name: str
    loader: Callable[[Path], torch.nn.Module]


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


def _resolve_best_models_dir() -> Path:
    if BEST_MODELS_DIR.exists():
        return BEST_MODELS_DIR
    base = Path("/export/scratch/jalegria/experiments")
    candidates = sorted(base.glob("best_models_*"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No best_models_* directory found under {base}.")
    return candidates[0]


def _load_cfno(run_dir: Path, shift: int) -> torch.nn.Module:
    weights = run_dir / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = FNO_2(
        in_channel=5,
        n_channels=32,
        n_residual_blocks=3,
        n_operator_blocks=2,
        modes=16,
        shifting_modes=shift,
        apply_constraint=False,
        last_layer_kernel=3,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def _load_edsr(run_dir: Path) -> torch.nn.Module:
    weights = run_dir / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = EDSR(
        input_channels=5,
        n_resblocks=16,
        n_feats=64,
        kernel_size=3,
        scale=4,
        activation_f=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def _build_model_specs(models_root: Path) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="cfno_shift0",
            supports_variable_scale=True,
            folder_name="cfno_shift0",
            loader=lambda _: _load_cfno(models_root / "cfno_shift0", 0),
        ),
        ModelSpec(
            name="cfno_shift4",
            supports_variable_scale=True,
            folder_name="cfno_shift4",
            loader=lambda _: _load_cfno(models_root / "cfno_shift4", 4),
        ),
        ModelSpec(
            name="cfno_shift8",
            supports_variable_scale=True,
            folder_name="cfno_shift8",
            loader=lambda _: _load_cfno(models_root / "cfno_shift8", 8),
        ),
        ModelSpec(
            name="cfno_shift12",
            supports_variable_scale=True,
            folder_name="cfno_shift12",
            loader=lambda _: _load_cfno(models_root / "cfno_shift12", 12),
        ),
        ModelSpec(
            name="cfno_shift16",
            supports_variable_scale=True,
            folder_name="cfno_shift16",
            loader=lambda _: _load_cfno(models_root / "cfno_shift16", 16),
        ),
        ModelSpec(
            name="edsr",
            supports_variable_scale=False,
            folder_name="edsr",
            loader=lambda _: _load_edsr(models_root / "edsr"),
        ),
    ]


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


def _run_model(
    model: torch.nn.Module, lr: np.ndarray, scale: int, variable_scale: bool
) -> np.ndarray:
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if variable_scale:
            sr = model(lr_t, upsample_factor=scale)
        else:
            if scale != BASE_UPSAMPLE_FACTOR:
                raise ValueError("Fixed-scale model requested for non-native scale.")
            sr = model(lr_t)
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

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    models_root = _resolve_best_models_dir()
    model_specs = _build_model_specs(models_root)

    hr_state = _generate_hr_states(n_states=1, num_cells=HR_NUM_CELLS, seed=SEED)[0]

    for spec in model_specs:
        model_dir = models_root / spec.folder_name
        if not model_dir.exists():
            print(f"Skipping {spec.name}: missing model folder {model_dir}")
            continue
        try:
            model = spec.loader(model_dir)
        except FileNotFoundError as exc:
            print(f"Skipping {spec.name}: {exc}")
            continue

        scales = [4, 2] if spec.supports_variable_scale else [4]
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
            sr_state = _run_model(
                model,
                lr_state,
                scale=scale,
                variable_scale=spec.supports_variable_scale,
            )

            out_name = f"hr_lr_sr_slices_x{scale}.png"
            out_path = model_dir / out_name
            _plot_triplet_rows(
                lr_state=lr_state,
                hr_state=hr_target,
                sr_state=sr_state,
                save_path=out_path,
                title=f"{spec.name} | x{scale} | rows: LR / HR / SR",
            )
            print(f"Saved {out_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
