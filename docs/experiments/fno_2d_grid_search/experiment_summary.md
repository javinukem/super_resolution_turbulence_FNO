# FNO 2D Grid Search — Experiment Summary

## Overview

This folder contains a **hyperparameter grid search** for the **FNO_2 (Fourier Neural
Operator, 2nd variant)** model applied to **2D turbulence super-resolution**. Unlike the
other experiments which use 3D volumetric data, this experiment operates on 2D slices
with 4-channel inputs.

**Source papers**:

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhatt, K., Stuart, A., & Anandkumar, A.
  (2021). *"Fourier Neural Operator for Parametric Partial Differential Equations"* (ICLR 2021).
  <https://openreview.net/pdf?id=c8P9NQVtmnO>

---

## Model Architecture (FNO_2)

The FNO_2 model (`src.model.models_fno2_2d`) is a 2D Fourier Neural Operator that:

- Uses **spectral convolution** via FFT to learn in the frequency domain
- Contains **FNO operator blocks** (spectral convolution + pointwise MLP) — analogous
  to the original FNO architecture
- Uses **residual blocks** (Conv2d-based) before upsampling
- Upsamples via learned convolution + pixel shuffle (factor 4×)
- Has a configurable `last_layer_constraint` (ReLU or exp) and `shifting_modes` parameter

### Key Parameters

| Parameter             | Description                                    |
|-----------------------|------------------------------------------------|
| `modes`               | Number of Fourier modes retained in spectral conv |
| `n_channels`          | Width of hidden feature layers                 |
| `n_residual_blocks`   | Residual blocks before upsampling              |
| `n_operator_blocks`   | Fourier operator blocks after upsampling       |
| `shifting_modes`      | Frequency-domain shift parameter               |
| `last_layer_kernel`   | Kernel size for the last convolution           |

---

## Experiment Setup

**Script**: `grid_search_fno2_2d.py`

### Hyperparameter Grid (as configured)

| Parameter             | Values           |
|-----------------------|------------------|
| `modes`               | 12, 22, 32       |
| `chan` (n_channels)   | 24, 32, 64       |
| `res` (n_residual_blocks) | 3            |
| `op` (n_operator_blocks)  | 1, 2, 3      |
| `last_layer_constraint`   | relu, exp     |

**Expected combinations**: 54

**Actual runs saved**: 10 (subset with `modes=8`, likely from an earlier or modified grid)

### Training Configuration

- **Data**: 2D turbulence, 4-channel input, no `max_samples` limit (uses all available data),
  all snapshots included (no `snapshot_index` filtering)
- **Split**: 10% train / 2% test / 88% validation
- **Learning rate**: 0.0002
- **Batch size**: 4
- **Epochs**: 100 (no early stopping)
- **Loss**: MSE
- **AMP**: Disabled
- **Upsample factor**: 4×

---

## Results

### Summary Table (all 10 runs)

| Configuration           | Modes | Channels | Res | Op | Val Loss       | Train Loss      | Test Loss       | Status      |
|------------------------|-------|----------|-----|-----|----------------|-----------------|-----------------|-------------|
| modes8_chan32_res3_op2  | 8     | 32       | 3   | 2   | **0.00799**    | 1.515           | 3.167           | ✅ Best      |
| modes8_chan32_res4_op2  | 8     | 32       | 4   | 2   | 0.02487        | 5.399           | 4.881           | ✅ Good      |
| modes8_chan32_res2_op2  | 8     | 32       | 2   | 2   | 0.07718        | 10.981          | 24.587          | ✅ OK        |
| modes8_chan32_res3_op3  | 8     | 32       | 3   | 3   | 10,606         | 805,997         | 421,198         | ❌ Diverged  |
| modes8_chan64_res2_op2  | 8     | 64       | 2   | 2   | 1.97 × 10¹⁹   | 2.80 × 10²¹    | 1.10 × 10²²    | ❌ Exploded  |
| modes8_chan32_res2_op3  | 8     | 32       | 2   | 3   | NaN            | NaN             | NaN             | ❌ NaN       |
| modes8_chan32_res2_op4  | 8     | 32       | 2   | 4   | NaN            | NaN             | NaN             | ❌ NaN       |
| modes8_chan32_res3_op4  | 8     | 32       | 3   | 4   | NaN            | NaN             | NaN             | ❌ NaN       |
| modes8_chan32_res4_op3  | 8     | 32       | 4   | 3   | NaN            | NaN             | NaN             | ❌ NaN       |
| modes8_chan32_res4_op4  | 8     | 32       | 4   | 4   | NaN            | NaN             | NaN             | ❌ NaN       |

### Statistics (converged models only)

| Metric                | Value     |
|-----------------------|-----------|
| Models converged      | 3 / 10    |
| Best validation loss  | 0.00799   |
| Best configuration    | modes=8, channels=32, res=3, op=2 |

---

## Key Findings

1. **Only 2 operator blocks work**: all configurations with `op ≥ 3` either diverged or
   produced NaN — deeper Fourier operator stacks cause numerical instability in 2D FNO
2. **64 channels is unstable**: the single run with 64 channels exploded to 10¹⁹ loss
3. **3 residual blocks is optimal**: res=3 beat both res=2 and res=4 when op=2
4. **Best model**: `modes8_chan32_res3_op2` with validation loss **0.00799**
5. **2D FNO shows good performance** when properly configured — val_loss nearly two orders
   of magnitude better than the 3D EDSR experiments (though different data dimensions)

---

## Artifacts per Model

Each subdirectory in `models/` contains:
- `config.yaml` — hyperparameter combination as JSON
- `results.json` — final train/test/validation losses
- `model.pt` — trained PyTorch weights (~2.2 MB)
- `loss_curve.png` — training/test loss over epochs
