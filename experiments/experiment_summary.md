# Experiments Overview

This directory contains all hyperparameter search and ablation experiments for the
**Turbulence Super-Resolution** project. The goal is to learn a 4× upsampling operator
from low-resolution turbulence simulation snapshots to high-resolution ones, using
neural-operator and CNN architectures.

> **Training is consolidated**: every retained experiment trains through the
> unified engine `src/training/experiment.py`, configured by the
> `experiment.yaml` in each folder (`python -m src.training.experiment
> experiments/<name>/experiment.yaml`). Historical references below to
> per-experiment `train_*.py` scripts correspond to that experiment's
> `experiment.yaml` run grid. Only the 4 model-zoo-backed experiment folders
> remain (`comparing_best_models_mse`, `mse_loss_combinations`,
> `edsr_norm_skip`, `ufno_mse_spectral`); all other experiment folders
> (incl. legacy frozen scripts under `cfno_2/`, `edsr/`, etc.) were removed —
> their record lives in git history and in this file / `MODELS.md`.

## Project Context

- **Task**: Super-resolution (4×) of compressible turbulence simulations
- **Data**: Generated from a custom Euler-equation solver with Kolmogorov-spectrum forcing
  - **3D data**: 128³ grid, 5 channels (velocity_x, velocity_y, velocity_z, density, pressure)
  - **2D data**: 4-channel slices
- **Loss**: MSE (L2)
- **Optimizer**: Adam (lr = 0.0002 typically)

---

## Experiment Folders

### [`cfno_2/`](cfno_2/experiment_summary.md) — Custom FNO v2 (3D)

The most extensive experiment set (6 sub-experiments). Explores a hybrid **Fourier Neural
Operator + CNN residual block** architecture for 3D turbulence. Progressed from initial
grid search → extended grid search with positivity constraints → gradient accumulation /
normalization ablation → Optuna Bayesian optimization → Hydra-based refactored pipeline.

- **Best result**: val_loss = **0.00125** (Experiment 2)
- **Best config**: modes=16, n_channels=32, res_blocks=3, op_blocks=2, constraint=exp

### [`edsr/`](edsr/experiment_summary.md) — Enhanced Deep Super-Resolution (3D)

EDSR adapted from 2D image SR to 3D turbulence with `Conv3d` + `PixelShuffle3d`. Two
sub-experiments: grid search and Optuna optimization.

- **Best result**: val_loss = **0.138** (Experiment 1)
- **Key finding**: extremely sensitive to hyperparameters — only 1/12 grid configs converged

### [`fno_2d_grid_search/`](fno_2d_grid_search/experiment_summary.md) — FNO_2 on 2D Data

Grid search for the FNO_2 model on 2D turbulence slices (4 channels). 10 configurations
tested with modes=8.

- **Best result**: val_loss = **0.00799**
- **Key finding**: only 2 operator blocks is stable; ≥3 causes NaN or divergence

### [`l1_spectral_skip_connection/`](l1_spectral_skip_connection/) — Skip-connection ablation (3D)

Single-script end-to-end experiment that activates the FNO_2 `skip_connection`
branch (which is off by default) and tests two `skip_connection_interpolation_mode`
settings — `nearest` and `trilinear` — with the skip-branch mode tied to the main
`interpolation_mode`. Training mirrors `l1_spectral_weighting/train_l1_spectral.py`
(CFNO shift=8, normalized, L1 + spectral loss at the calibrated `w_minor` weight,
AdamW + ReduceLROnPlateau, 250 epochs / patience 40, snapshot 79, x4). The script
embeds benchmark evaluation (the two skip runs plus the prior best `w_minor` model
as a skip=False reference and a trilinear baseline, at x4 and x2), bar charts, and
a jf1uids final-snapshot state-grid comparison in a single `python` invocation
(`bash run.sh`).

- **Prerequisite patch**: `src/model/models_fno_2.py` was patched — the skip branch
  previously re-interpolated the already-HR `out` (shape mismatch); it now
  interpolates the LR `latent`. `nearest` mode (which rejects `align_corners`) is
  also fixed via the new `_upsample_3d` helper.
- **Output**: `/export/scratch/jalegria/experiments/l1_spectral_skip_connection_<ts>/`
  with per-run `weights.pt`, `losses.csv`, `loss_curve.png`, `config.json`, plus
  `benchmark_metrics.csv`, two bar-chart PNGs, `final_snapshot_comparison.png`,
  `comparison_states.npy`, `manifest.json`, `training_summary.csv`, `summary.json`.
- **Bar-chart preset**: `python evaluation/comparing_models_bar_chart.py --preset l1_spectral_skip`.

### [`ufno_l1_spectral_unet/`](ufno_l1_spectral_unet/) — U-FNO U-Net depth sweep (3D)

Single-script end-to-end experiment that trains the new **`UFNO_2`** architecture
(which augments each FNO_2 operator block with a parallel 3-D U-Net path, per Wen
et al. / the RT-JAX port https://github.com/RuneRost/RT-JAX/blob/main/architectures/ufno_3d.py)
under the exact training/loss/normalization regime of the `l1_spectral_skip_connection`
reference run (`cfno_shift8_skip_trilinear`: CFNO shift=8, normalized, L1 + spectral
`w_minor`, AdamW + ReduceLROnPlateau, 250 epochs / patience 40, snapshot 79, x4/x2 eval).
The sweep is a 2×2 grid over `n_operator_blocks ∈ {1, 2}` and `dropout_rate ∈ {0.0, 0.1}`
— `ufno_shift8_op1_drop0`, `ufno_shift8_op1_drop01`, `ufno_shift8_op2_drop0`,
`ufno_shift8_op2_drop01`. The manifest also carries the prior best `w_minor` CFNO
(`skip=False`) reference and a trilinear baseline, so UFNO is benchmarked head-to-head
against the vanilla FNO_2 operator stack.

- **Prerequisite patch**: `evaluation/manifest.py` gained a `model_type == "ufno"`
  branch in `_instantiate` (`from src.model.models_ufno_2 import UFNO_2`).
- **Output routing inversion**: `benchmark_metrics.csv`, the two bar-chart PNGs, and
  `final_snapshot_comparison.png` are written DIRECTLY into this repo experiment dir,
  NOT into the scratch `output_dir`. Weights / losses / config / manifest / summary /
  `comparison_states.npy` keep going to `/export/scratch/jalegria/experiments/
  ufno_l1_spectral_unet_<ts>/`.
- **Bar-chart preset**: `python evaluation/comparing_models_bar_chart.py --preset ufno`.

### `grid_search_dsfno.py` — DSFNO Grid Search Script

Standalone Hydra-based grid search script for the **DSFNO (Downscaling FNO)** model
(3D variant from Yang et al.). Searches over modes, n_channels, n_residual_blocks, and
n_operator_blocks (81 combinations). Results are saved to `runs/dsfno_grid/`.

- **Config**: `configs/experiments/grid_search_dsfno.yaml`
- **Model source**: Yang, Q. et al. *"Fourier Neural Operators for Arbitrary Resolution
  Climate Data Downscaling"*. <https://github.com/qy707/DSFNO>

---

## Cross-Model Comparison

| Model       | Data | Best Val Loss | Converge Rate | Notes                              |
|-------------|------|---------------|---------------|------------------------------------|
| **CFNO_2**  | 3D   | **0.00125**   | 43/44 (98%)   | Best overall; hybrid FNO+ResBlock  |
| FNO_2 (2D)  | 2D   | 0.00799       | 3/10 (30%)    | Unstable with deep operator stacks |
| EDSR        | 3D   | 0.138         | 1/12 (8%)     | Very unstable for 3D turbulence    |

### Key Takeaways

1. **FNO-based models vastly outperform pure CNN (EDSR)** for turbulence SR — ~110× lower loss
2. **Positivity constraints on density/pressure** (relu/exp on last layer) are critical for
   physical consistency and improved performance
3. **Moderate model sizes are optimal** — 32 channels, 16 Fourier modes, 2–3 operator blocks
4. **Deeper operator stacks (≥3) are unstable** across all FNO variants
5. **Gradient accumulation and normalization** improve training stability (CFNO_2 Experiment 3)

### Data Usage Notes

- **Most experiments used all 80 snapshots** per simulation (the full time series), which
  means temporal snapshots from the same simulation appear in training — there is no
  simulation-level train/test split.
- **Exception**: CFNO Experiment 5 used `snapshot_index=79` (last snapshot only), giving
  at most 500 unique samples (one per simulation).
- The `max_samples` parameter was used to cap data size (typically 2500–5000 samples).
- No dedicated validation set existed during these experiments; train/test split was done
  randomly on the sample level (not simulation level).

---

## References

- Li, Z. et al. (2021). *"Fourier Neural Operator for Parametric Partial Differential
  Equations"*. ICLR 2021. <https://openreview.net/pdf?id=c8P9NQVtmnO>
- Yang, Q. et al. *"Fourier Neural Operators for Arbitrary Resolution Climate Data
  Downscaling"*. <https://github.com/qy707/DSFNO>
- Lim, B. et al. (2017). *"Enhanced Deep Residual Networks for Single Image
  Super-Resolution"*. CVPR 2017. <https://github.com/sanghyun-son/EDSR-PyTorch>
- SoftmaxConstraint layer: <https://arxiv.org/pdf/2208.05424>
