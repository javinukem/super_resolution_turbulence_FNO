# EDSR Experiments Summary

## Overview

This folder contains experiments for **EDSR (Enhanced Deep Super-Resolution)** applied to
**3D turbulence super-resolution**. The EDSR architecture, originally designed for 2D image
super-resolution, was adapted here to use 3D convolutions for volumetric turbulence data
(5-channel input: 3 velocity components + density + pressure).

**Source paper**: Lim, B., Son, S., Kim, H., Nah, S., & Lee, K.M. (2017).
*"Enhanced Deep Residual Networks for Single Image Super-Resolution"* (CVPR 2017).
Code reference: <https://github.com/sanghyun-son/EDSR-PyTorch>

---

## Model Architecture

The EDSR model (`models_edsr.py`) consists of:

- **Input convolution**: `Conv3d(in_channels, n_feats, kernel_size)`
- **Residual blocks**: Stack of `n_resblocks` residual blocks, each containing two `Conv3d`
  layers with ReLU activation and a skip connection
- **Upsampler**: `PixelShuffle3d`-based upsampling (factor 4×) — the 3D analog of sub-pixel
  convolution from Shi et al. (2016)
- **Output convolution**: `Conv3d(n_feats, in_channels, kernel_size)` to produce the final
  high-resolution output

All convolutions use `padding="same"` to preserve spatial dimensions within blocks.

---

## Experiment 1 — Grid Search

**Date**: ~August–September 2025
**Script**: `run_parallel_4.py` (multi-GPU parallel grid search, up to 5 GPUs)

### Hyperparameter Grid

| Parameter          | Values Tested   |
|--------------------|-----------------|
| `n_resblocks`      | 16, 32, 64      |
| `n_feats`          | 64, 128         |
| `kernel_size`      | 3, 5            |
| `activation_f`     | False (no act.)  |

**Total combinations**: 12

### Training Setup

- **Data**: max_samples=4000, all 80 snapshots per simulation (= first 50 sims × 80),
  128³ resolution, 5 channels
- **Split**: 80% train / 20% test
- **Learning rate**: 0.0005
- **Batch size**: 1 (limited by 3D data memory)
- **Epochs**: 100
- **Loss**: MSE
- **Upsample factor**: 4×

### Results (`experiment_1/evaluation/eval_results.csv`)

| Configuration                  | Val Loss          | Inference Time (s) | Status       |
|-------------------------------|-------------------|---------------------|--------------|
| resb16_feats64_ks3            | **0.1376**        | 0.110               | ✅ Converged  |
| resb64_feats128_ks3           | 7.47 × 10⁷       | 0.650               | ❌ Diverged   |
| resb16_feats128_ks5           | 3.48 × 10⁹       | 0.932               | ❌ Diverged   |
| resb16_feats64_ks5            | 3.34 × 10⁸       | 0.258               | ❌ Diverged   |
| resb64_feats64_ks3            | 5.06 × 10⁸       | 0.183               | ❌ Diverged   |
| resb32_feats64_ks3            | 1.97 × 10¹³      | 0.132               | ❌ Diverged   |
| resb16_feats128_ks3           | 3.51 × 10¹³      | 0.378               | ❌ Diverged   |
| resb32_feats128_ks3           | 6.33 × 10¹³      | 0.465               | ❌ Diverged   |
| resb32_feats64_ks5            | 1.04 × 10¹⁶      | 0.360               | ❌ Diverged   |
| resb32_feats128_ks5           | 2.78 × 10²⁰      | 1.318               | ❌ Diverged   |
| resb64_feats64_ks5            | 1.35 × 10¹¹      | 0.557               | ❌ Diverged   |
| resb64_feats128_ks5           | 7.02 × 10²⁹      | 2.079               | ❌ Diverged   |

### Key Findings

- **Only 1 out of 12 configurations converged** — `resb16_feats64_ks3` with val_loss = 0.1376
- **Kernel size 5 universally failed**: every `ks=5` configuration diverged
- **Larger models are unstable**: more residual blocks and features led to numerical explosion
- **Smallest model won**: 16 res-blocks, 64 features, kernel 3 was the most stable
- **Inference speed**: the best model is also the fastest at 0.110s/sample

---

## Experiment 3 — Optuna Bayesian Hyperparameter Optimization

**Date**: ~October 2025
**Script**: `hyp_tuning.py` (Optuna-based Bayesian optimization)

### Search Space

| Parameter          | Range / Options              |
|--------------------|------------------------------|
| `n_resblocks`      | 1–8 (integer)                |
| `n_feats`          | 32–128 (integer)             |
| `kernel_size`      | 3, 5                         |
| `activation_f`     | None, ReLU, PReLU            |

**Trials**: 20
**Training**: 80 epochs, batch_size=8, gradient accumulation, AMP enabled, normalization on
**Data**: max_samples=4000, all 80 snapshots per simulation (= first 50 sims × 80)

### Results (`experiment_3/hyp_tuning.csv`)

| Trial | Val Loss       | Activation | Features | Kernel | Res Blocks | Status |
|-------|---------------|------------|----------|--------|------------|--------|
| 10    | **0.1616**    | None       | 47       | 3      | 8          | Best   |
| 19    | 0.1986        | None       | 74       | 3      | 8          | 2nd    |
| 13    | 0.2013        | None       | 57       | 3      | 7          | 3rd    |
| 0     | 0.7618        | PReLU      | 54       | 3      | 8          | —      |
| 5     | 0.5272        | ReLU       | 96       | 3      | 8          | —      |
| 1     | 1.38 × 10⁷   | PReLU      | 104      | 5      | 6          | Failed |

### Key Findings

- **Best trial**: 47 features, 8 res-blocks, kernel 3, no activation — val_loss = 0.1616
- **No activation is best**: trials with `activation_f = None` consistently outperformed
  those with ReLU or PReLU
- **Kernel 3 dominates**: all successful trials used kernel_size = 3
- **More res-blocks helps**: 7–8 residual blocks performed best
- **Moderate feature count**: 47–74 features optimal (not too small, not too large)
- Grid search (Exp 1) still achieved a slightly better best result (0.1376 vs 0.1616),
  suggesting the grid happened to find a good configuration

---

## Cross-Experiment Conclusions

1. **EDSR for 3D turbulence is highly sensitive to hyperparameters** — most configurations diverge
2. **Optimal configuration**: small kernel (3), moderate depth (8–16 res-blocks), moderate
   width (47–64 features), no activation function in residual blocks
3. **Best achieved validation loss**: ~0.138 (grid search, Experiment 1)
4. **Data usage**: Both experiments used all 80 snapshots per simulation (no `snapshot_index`
   filtering). Experiment 1 used max_samples=4000 (~50 sims), Experiment 3 also used
   max_samples=4000. No simulation-level train/test split was applied.
