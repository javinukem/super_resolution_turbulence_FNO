# L1 + Spectral Loss Weighting Experiment

## Overview

Trains **CFNO shift=8** (the best x4 model from `training_best_models_experiment`)
with a combined **L1 + spectral** loss at three different spectral weights, using
dataloader-level normalization. The goal is to find the spectral-loss weight that
best trades off pixel accuracy (L1) against high-frequency structure (spectral),
addressing the blurry outputs of pure MSE training.

A calibration script first measures the L1 and spectral loss magnitudes on a
trilinear baseline prediction, derives the weight that equalizes them, and
generates three weights spanning two orders of magnitude. The training script
then trains one model per weight.

---

## Calibration

`calibrate_weights.py` measures L1 and the velocity log-power spectral loss on
trilinear-upsampled LR vs HR target over 10 normalized training batches. The
ratio `L1 / spectral` gives the weight `w_eq` that makes both terms contribute
equally at the start of training. Three weights are derived:

| Weight name | Value        | Meaning                                    |
|-------------|--------------|--------------------------------------------|
| `w_minor`   | `w_eq * 0.1` | Spectral is a gentle nudge; L1 dominates   |
| `w_equal`   | `w_eq`       | Both terms balanced                        |
| `w_major`   | `w_eq * 10`  | Spectral dominates; aggressive high-freq   |

Results are saved to `calibration.json` and auto-read by the training script.

```bash
python experiments/l1_spectral_weighting/calibrate_weights.py
```

---

## Models Trained

### CFNO (FNO_2) — shift=8, 3 variants

All share the same architecture and normalization; only the spectral loss weight
differs:

| Parameter             | Value |
|-----------------------|-------|
| `in_channel`          | 5     |
| `n_channels`          | 32    |
| `n_residual_blocks`   | 3     |
| `n_operator_blocks`   | 2     |
| `modes`               | 16    |
| `shifting_modes`      | 8     |
| `apply_constraint`    | False |
| `last_layer_kernel`   | 3     |
| `apply_positivity_relu` | False (norm-on) |

| Variant                        | `spectral_weight` |
|--------------------------------|--------------------|
| `cfno_shift8_l1_spectral_w_minor` | `w_eq * 0.1`    |
| `cfno_shift8_l1_spectral_w_equal` | `w_eq`          |
| `cfno_shift8_l1_spectral_w_major` | `w_eq * 10`     |

Parameters per variant: **~33.7M**

---

## Loss

```
total = L1(pred, target) + spectral_weight * SpectralLoss(pred, target)
```

`SpectralLoss` operates on the 3 velocity channels: 3-D `rfftn` (`norm="ortho"`),
power = |F|², mean over channels, log10, MSE, with `avg_pool3d(2)` before the FFT.

---

## Training Setup

| Setting             | Value                        |
|---------------------|------------------------------|
| **Optimizer**       | AdamW (lr=0.001, weight_decay=1e-4) |
| **Scheduler**       | ReduceLROnPlateau (factor=0.5, patience=15) |
| **Loss**            | L1 + spectral (per-run weight) |
| **Epochs**          | 250                          |
| **Early stop**      | patience 40                  |
| **Batch size**      | 8 × grad_accum 8 = effective 64 |
| **AMP**             | Disabled                     |
| **Noise augmentation** | Gaussian, σ=0.01 on LR input |
| **Snapshot**        | Last only (index 79)         |
| **Normalization**   | Dataloader-level (per-channel mean/std) |
| **Positivity ReLU** | Disabled (norm-on)           |

### Data

- **Training**: `/export/scratch/jalegria/full_states_h5/full_states.h5`
  with `snapshot_index=79` → 400 samples
- **Validation**: `/export/scratch/jalegria/full_states_h5/full_states_val.h5`
  with `snapshot_index=79` → 100 samples
- Normalization stats cached at
  `experiments/l1_spectral_weighting/normalization_stats.npz` (shared between
  calibration and training).

---

## Output

Saved to `/export/scratch/jalegria/experiments/l1_spectral_weighting_<timestamp>/`:

```
l1_spectral_weighting_MM-DD_HH-MM/
├── cfno_shift8_l1_spectral_w_minor/
│   ├── weights.pt
│   ├── losses.csv
│   ├── loss_curve.png
│   └── config.json
├── cfno_shift8_l1_spectral_w_equal/
│   └── ...
├── cfno_shift8_l1_spectral_w_major/
│   └── ...
└── summary.json
```

---

## Usage

```bash
# 1. Calibrate the spectral weight
python experiments/l1_spectral_weighting/calibrate_weights.py

# 2. Train the 3 models
python experiments/l1_spectral_weighting/train_l1_spectral.py
```
