# Trying New Losses and Normalization Experiment

## Overview

Retrains the two best-performing models from `training_best_models_experiment`
(**CFNO shift=8** — best x4 MSE on the benchmark; **EDSR**) with different
loss functions and per-channel normalization, to address the blurry (CFNO) and
blocky (EDSR) outputs of the original MSE-only training.

### Motivation

Analysis of the original trained models identified the dominant causes as:

- **`nn.MSELoss()`** — L2 regresses to the posterior mean; for ill-posed SR the
  L2-optimal prediction is the average over sharp solutions, i.e. smooth/blurry.
- **No input/target normalization** — density dominates the loss by ~1-2 orders
  of magnitude, so pressure/velocity (the channels carrying sharp small-scale
  structure) are under-weighted.
- **EDSR `activation_f=False`** — removes the inter-stage nonlinearity between
  the two ×2 PixelShuffle stages, reducing the upsampler's capacity to
  decorrelate the pixelshuffled channels → periodic blockiness.
- **EDSR undertraining** — val loss was still monotonically decreasing at
  epoch 100.

---

## Models Trained

### CFNO (FNO_2) — shift=8

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

Architecture unchanged from `training_best_models_experiment`.

### EDSR — with `activation_f="prelu"`

| Parameter          | Value    |
|--------------------|----------|
| `input_channels`   | 5        |
| `n_resblocks`      | 16       |
| `n_feats`          | 64       |
| `kernel_size`      | 3        |
| `scale`            | 4        |
| `activation_f`     | `"prelu"` (was `False`) |

The `activation_f` change restores a learnable PReLU between the two
PixelShuffle stages in the upsampler (`src/model/models_edsr.py:76-79`),
giving it capacity to decorrelate the shuffled channel planes.

### `apply_positivity_relu` flag

Both `FNO_2` and `EDSR` gained a backwards-compatible
`apply_positivity_relu: bool = True` flag (default `True` = unchanged
behaviour). For **norm-on** runs it is set to `False` so the final ReLU on
density/pressure does not clip ~half of the zero-mean/unit-std normalized
signal. Norm-off runs keep the ReLU as before.

---

## Run Grid (12 runs)

Cross product of 2 models × 3 losses × 2 normalization settings:

| Loss            | Components                              | Weight |
|-----------------|-----------------------------------------|--------|
| `mse_l1`        | MSE + L1                                | L1 weight = 1.0 |
| `mse_spectral`  | MSE + velocity log-power spectral loss  | spectral weight = 0.01 |
| `l1`            | pure L1                                 | — |

Normalization: **off** (raw, as before) / **on** (dataloader-level per-channel
standardization via `dataset_sr(use_normalizing=True)`).

Run name pattern: `{model}_{loss}_{norm|nonorm}`, e.g.
`cfno_shift8_mse_spectral_norm`, `edsr_l1_nonorm`.

The spectral loss operates on the 3 velocity channels only (precedent:
`train_fno2_grid_modes_interp_skip_losses_refine.py:270-289`): 3-D `rfftn`
(`norm="ortho"`), power = |F|², mean over channels, log10, MSE, with
`avg_pool3d(2)` before the FFT to reduce memory.

---

## Training Setup

| Setting             | Value                        |
|---------------------|------------------------------|
| **Optimizer**       | AdamW (lr=0.001, weight_decay=1e-4) |
| **Scheduler**       | ReduceLROnPlateau (factor=0.5, patience=15) |
| **Loss**            | per-run (see grid)           |
| **Epochs**          | 250 (was 100)                |
| **Early stop**      | patience 40 (was 30)         |
| **Batch size**      | 8 × grad_accum 8 = effective 64 |
| **AMP**             | Disabled                     |
| **Noise augmentation** | Gaussian, σ=0.01 on LR input |
| **Snapshot**        | Last only (index 79)         |

### Normalization

Per-channel mean/std computed **once** from the training set via
`compute_mean_std_dataset` (`src/utils/mean_std.py`) and cached to
`normalization_stats.npz` at the output root. The validation dataset reuses
the same (train) statistics. On re-runs the cache is loaded instead of
recomputed. The cache path is recorded in each norm-on run's `config.json`.

This caching is implemented via a new `mean_std_path` kwarg on
`dataset_sr` (`src/dataloader/dataloader_3d.py`): if the file exists it is
loaded, otherwise the stats are computed and saved.

### Data

- **Training**: `/export/scratch/jalegria/full_states_h5/full_states.h5`
  with `snapshot_index=79` → 400 samples (one per training simulation)
- **Validation**: `/export/scratch/jalegria/full_states_h5/full_states_val.h5`
  with `snapshot_index=79` → 100 samples (one per validation simulation)
- Simulation-level split (same as `training_best_models_experiment`).

### Memory Management

Models are trained sequentially — each run is fully trained, its artifacts
saved, and then deleted from GPU memory (`del model` + `gc.collect()` +
`torch.cuda.empty_cache()`) before the next run. Each run is wrapped in
`try/except` so a failure does not prevent the others.

---

## Output

Saved to `/export/scratch/jalegria/experiments/loss_norm_experiment_<timestamp>/`:

```
loss_norm_experiment_MM-DD_HH-MM/
├── cfno_shift8_mse_l1_norm/
│   ├── weights.pt
│   ├── losses.csv
│   ├── loss_curve.png
│   └── config.json
├── cfno_shift8_mse_l1_nonorm/
│   └── ...
├── cfno_shift8_mse_spectral_norm/
│   └── ...
├── ... (12 run folders total)
├── normalization_stats.npz      # cached train mean/std (HR + LR)
└── summary.json                 # overall results for all runs
```

---

## Usage

```bash
python experiments/trying_new_losses_and_norm/train_loss_norm.py
```

After training, evaluate with `evaluation/benchmark.py` (set
`TRAINED_MODELS_DIR` to the run folder). For norm-on models, note that the
model expects normalized inputs — wrap inference with the cached
`normalization_stats.npz` (normalize LR in, denormalize SR out) or extend
`benchmark.py` to handle the `apply_positivity_relu=False` + normalization path.

---

## References

- L1 / Charbonnier loss for sharper SR: Lim et al. (2017), *"Enhanced Deep
  Residual Networks for Single Image Super-Resolution"*.
- Spectral loss (frequency-domain MSE): Gal et al. (2023);
  precedent in-repo at
  `experiments/train_fno2_grid_modes_interp_skip_losses_refine/...py:270-289`.
- CFNO upsampling-before-FNO design: the CFNO paper (arXiv:2305.14452).
