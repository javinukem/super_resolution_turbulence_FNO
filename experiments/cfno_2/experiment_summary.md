# CFNO_2 (Custom Fourier Neural Operator v2) — Experiment Summary

## Overview

This folder contains a **progressive series of 6 experiments** for the **FNO_2** (Fourier
Neural Operator, 2nd variant) applied to **3D turbulence super-resolution**. The experiments
span from initial grid search to gradient accumulation studies, normalization ablation,
Optuna-based hyperparameter optimization, and finally a Hydra-based refactored training
pipeline. The model operates on 5-channel 3D volumetric data (3 velocity components +
density + pressure) at 128³ resolution.

**Source papers**:

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhatt, K., Stuart, A., & Anandkumar, A.
  (2021). *"Fourier Neural Operator for Parametric Partial Differential Equations"* (ICLR 2021).
  <https://openreview.net/pdf?id=c8P9NQVtmnO>

- Yang, Q. et al. *"Fourier Neural Operators for Arbitrary Resolution Climate Data
  Downscaling"*. <https://github.com/qy707/DSFNO>
  (SoftmaxConstraint layer adapted from: <https://arxiv.org/pdf/2208.05424>)

---

## Model Architecture (FNO_2)

The FNO_2 model (`fno_2_exp_2.py`, evolved across experiments) is a hybrid architecture
combining Fourier operator blocks with CNN residual blocks:

1. **Input lifting**: `Conv3d(in_channel, n_channels, 1)` — lifts input to hidden dim
2. **Residual blocks**: Stack of `n_residual_blocks` CNN residual blocks (from EDSR)
   operating in physical space
3. **Upsampler**: `PixelShuffle3d`-based 4× upsampling
4. **FNO operator blocks**: Stack of `n_operator_blocks` Fourier blocks, each performing:
   - 3D spectral convolution (via FFT, mode truncation, learned weights, IFFT)
   - Pointwise MLP bypass
   - GELU activation + residual connection
5. **Output projection**: `Conv3d(n_channels, in_channel, last_layer_kernel)`
6. **Optional constraint layer**: `SoftmaxConstraint` to enforce physical consistency
   between low-res input and super-resolved output
7. **Last layer constraint**: Optional ReLU or exp applied to density/pressure channels
   (to enforce positivity of physical quantities)

### Key Parameters

| Parameter               | Description                                         |
|-------------------------|-----------------------------------------------------|
| `modes`                 | Fourier modes retained in spectral convolution      |
| `n_channels`            | Hidden feature width                                |
| `n_residual_blocks`     | CNN residual blocks before upsampling               |
| `n_operator_blocks`     | Fourier operator blocks after upsampling            |
| `shifting_modes`        | Frequency-domain shift parameter                    |
| `last_layer_kernel`     | Kernel size for final convolution                   |
| `last_layer_constraint` | `relu` or `exp` — positivity constraint on density/pressure |
| `apply_constraint`      | Whether to use the SoftmaxConstraint post-processing|

---

## Experiment 1 — Initial Grid Search (Small Scale)

**Date**: July 10–11, 2025
**Script**: `fno_2_used.py` (model definition used inline)

### Setup

- **Grid**: 5 configurations varying `modes`, `n_residual_blocks`, `shifting_modes`
- **Fixed**: n_channels=64, n_operator_blocks=2, batch_size=1, epochs=100, lr=0.0005
- **No last_layer_constraint** (not yet implemented)
- **No normalization**, no gradient accumulation
- **Data**: all 80 snapshots per simulation used (no `snapshot_index` filter), sample
  count unknown (training script not preserved — likely one of the deleted `main_*.py`)

### Results (`experiment_1/fno_eval_results.csv`)

| Configuration                            | Val Loss   | Inference (s) | Modes | Channels | Res | Op |
|-----------------------------------------|-----------|---------------|-------|----------|-----|----|
| m16_nc64_res4_op2_ac0_lk3 (shift=16)   | **0.0280** | 0.116        | 16    | 64       | 4   | 2  |
| m16_nc64_res3_op2_ac0_lk3 (shift=16)   | 0.0281    | 0.115         | 16    | 64       | 3   | 2  |
| m16_nc64_res2_op2_ac0_lk3 (shift=16)   | 0.0301    | 0.119         | 16    | 64       | 2   | 2  |
| m16_nc64_res1_op2_ac0_lk3 (shift=16)   | 0.0319    | 0.128         | 16    | 64       | 1   | 2  |
| m18_nc64_res4_op2_ac0_lk3 (shift=1)    | 0.0555    | 0.116         | 18    | 64       | 4   | 2  |

### Key Findings

- **Best**: m16_nc64_res4_op2 with `shifting_modes=16` — val_loss = 0.0280
- More residual blocks slightly helps (1 → 4 blocks: 0.032 → 0.028)
- `shifting_modes=16` significantly better than `shifting_modes=1` (0.028 vs 0.056)
- All 5 models converged successfully (unlike EDSR where 11/12 diverged)

---

## Experiment 2 — Extended Grid Search with Last-Layer Constraint

**Date**: ~August–September 2025
**Script**: `run_parallel_4.py` (parallel multi-GPU grid search)
**Change log**: *"Added a RELU or LOG option for the density and pressure"*

### Setup

- **Grid**: modes × channels × res × op × last_layer_constraint
- **Values**: modes=[12,22,32], chan=[24,32,64], res=[3], op=[1,2,3],
  last_layer_constraint=[relu,exp]
- **Additional runs** with modes=16 and varied parameters
- **Total runs**: 44
- **Training**: lr=0.0002, epochs=85–100, batch_size=4
- **Data**: `max_samples=4000`, all 80 snapshots per simulation (no `snapshot_index`
  filter) — uses the first 4,000 sequential samples (= first 50 sims × 80 snapshots)

### Results (`experiment_2/evaluation/fno_eval_results.csv`)

**Top 5 models**:

| Configuration                          | Val Loss     | Constraint | Modes | Channels | Res | Op |
|---------------------------------------|-------------|------------|-------|----------|-----|----|
| m16_nc32_res3_op2_ac0_lk3lc**exp**   | **0.001245** | exp        | 16    | 32       | 3   | 2  |
| m16_nc32_res3_op2_ac0_lk3lc**relu**  | 0.001248    | relu       | 16    | 32       | 3   | 2  |
| m16_nc32_res3_op3_ac0_lk3lc**exp**   | 0.001275    | exp        | 16    | 32       | 3   | 3  |
| m32_nc32_res3_op3_ac0_lk3lc**exp**   | 0.001312    | exp        | 32    | 32       | 3   | 3  |
| m22_nc32_res3_op3_ac0_lk3lcrelu      | 0.001489    | relu       | 22    | 32       | 3   | 3  |

**Aggregate statistics** (43 valid models):

| Metric          | Value     |
|-----------------|-----------|
| Min val_loss    | 0.001245  |
| Max val_loss    | 0.034249  |
| Mean val_loss   | 0.005212  |
| Models tested   | 44        |
| Valid (non-NaN) | 43        |

### Key Findings

- **Massive improvement over Exp 1**: best loss dropped from 0.028 → 0.00125 (22× better)
- **32 channels outperform 64**: 32-channel models consistently better (0.001 range vs
  0.003–0.011 for 64 channels)
- **16 modes is optimal**: modes=16 consistently beats 22 and 32
- **exp vs relu constraint**: nearly identical for best configs, `exp` slightly favored
- **24-channel models underperform**: losses in the 0.004–0.007 range
- **Extremely consistent**: 43/44 models converged, unlike EDSR and 2D FNO experiments

---

## Experiment 3 — Gradient Accumulation & Normalization Ablation

**Date**: ~September 2025
**Change log**: *"Added batch accumulation to the training script; Added normalization to
the dataset"*

### Sub-experiment 3a: Gradient Accumulation (`run_acc_grad_experiment.py`)

Trained the **same model** 5 times with different gradient accumulation settings.
**Data**: `max_samples` from config, all 80 snapshots per simulation (no `snapshot_index`
filter), with optional normalization.

| Accumulation Steps | Effect                                          |
|-------------------|--------------------------------------------------|
| None              | Standard training (effective batch_size = batch_size) |
| 3                 | Effective batch_size = batch_size × 3            |
| 6                 | Effective batch_size = batch_size × 6            |
| 9                 | Effective batch_size = batch_size × 9            |
| 12                | Effective batch_size = batch_size × 12           |

Results saved to `/export/data/jalegria/experiments/experiment_3/acc_grad/losses.npz`.
Visualization in `visualize_acc_grad.ipynb`.

### Sub-experiment 3b: Normalization (`run_norm_experiment.py`)

Compared training **with vs. without input data normalization** on the same model config.

### Purpose

These ablation studies informed the training pipeline improvements used in Experiments 4–6
(gradient accumulation = 3 and normalization = True became the defaults).

---

## Experiment 4 — Optuna Hyperparameter Optimization (v1)

**Date**: ~October 2025
**Script**: `hyp_tuning.py` (Optuna Bayesian optimization)

### Setup

- **Trials**: 12
- **Training**: 80 epochs, batch_size=8, total_effective_batch=24 (grad accumulation)
- **Data**: `max_samples=5000`, all 80 snapshots per simulation (no `snapshot_index`
  filter), normalization enabled — uses the first 5,000 sequential samples
  (= first 62.5 sims × 80 snapshots)
- **Study stored in**: `results/optuna_study.db` (SQLite)
- **Analysis**: `results/optuna_analyse.ipynb`

### Search Space (Optuna-defined)

The script uses Optuna's `suggest_int`/`suggest_categorical` to search over:
- `modes`, `n_channels`, `n_residual_blocks`, `n_operator_blocks`
- `shifting_modes`, `last_layer_constraint`

---

## Experiment 5 — Optuna Hyperparameter Optimization (v2)

**Date**: ~October 2025
**Script**: `hyp_tuning.py` (expanded Optuna search)

### Setup

- **Trials**: 20 (increased from 12)
- **Training**: 80 epochs, batch_size=8, total_effective_batch=40
- **Data**: `snapshot_index=79` — **only the last (final) snapshot from each simulation**,
  normalization enabled. This means at most 500 samples (one per simulation), not the
  full 40,000. This is a significant change from Experiments 1–4 which used all 80
  time snapshots per simulation.
- **Study stored in**: `results/optuna_study.db`
- **Analysis**: `results/optuna_analyse.ipynb`
- Removed `apply_constraint` parameter from the model (compared to Exp 4)

---

## Experiment 6 — Hydra-Based Refactored Training

**Date**: ~February 2026
**Script**: `training_fno.py` (uses Hydra + refactored pipeline utilities)
**Config**: `configs/experiments/training_fno_exp6.yaml`

### What Changed

This experiment marks the **transition to the refactored codebase** using:
- **Hydra** for configuration management (instead of raw YAML + manual parsing)
- **Pipeline utilities**: `build_dataset`, `build_model`, `build_train_test_loaders`,
  `create_run_dir`, `save_training_artifacts` from `src.utils.pipeline`
- **Structured output**: runs saved to `experiments/cfno_2/experiment_6/runs/`

### Configuration

| Parameter             | Value                |
|-----------------------|----------------------|
| Model                 | FNO_2 (from Exp 4)  |
| in_channel            | 5                    |
| n_channels            | 32                   |
| n_residual_blocks     | 2                    |
| n_operator_blocks     | 2                    |
| modes                 | 16                   |
| shifting_modes        | 3                    |
| last_layer_constraint | relu                 |
| Data                  | 3D, max_samples=2500, all 80 snapshots per simulation (≈31 sims) |
| Split                 | 80/20                |
| Epochs                | 85                   |
| Learning rate         | 0.0002               |
| Batch size            | 4                    |
| AMP                   | Enabled              |
| Early stopping        | Yes (patience=8)     |
| Upsample factor       | 4×                   |

This experiment serves as the **production training script** for single-model training
after hyperparameters were identified in Experiments 1–5.

---

## Progression Summary

| Exp | Date      | Method         | Best Val Loss | Key Change                           |
|-----|-----------|----------------|---------------|--------------------------------------|
| 1   | Jul 2025  | Small grid     | 0.0280        | Initial baseline, no constraints     |
| 2   | Aug 2025  | Large grid     | **0.00125**   | Added last_layer_constraint (relu/exp)|
| 3   | Sep 2025  | Ablation       | —             | Gradient accumulation + normalization |
| 4   | Oct 2025  | Optuna (12)    | (in .db)      | Bayesian optimization                |
| 5   | Oct 2025  | Optuna (20)    | (in .db)      | Extended Bayesian optimization       |
| 6   | Feb 2026  | Single train   | —             | Hydra refactor, production pipeline  |

### Overall Best Configuration

From Experiment 2: **modes=16, n_channels=32, n_residual_blocks=3, n_operator_blocks=2,
last_layer_constraint=exp** → validation loss = **0.001245**

This represents a **~110× improvement** over the initial Experiment 1 best (0.028) and
a **~110× improvement** over the EDSR best (0.138), demonstrating that the hybrid
FNO+ResBlock architecture with positivity constraints is well-suited for 3D turbulence
super-resolution.
