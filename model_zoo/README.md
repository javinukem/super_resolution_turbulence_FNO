# Model Zoo

Pretrained 4× super-resolution models for 3-D compressible turbulence (jf1uids
primitive states: `[density, vx, vy, vz, pressure]`, LR 32³ → HR 128³).

**Weights are not stored in this repository.** They are hosted on HuggingFace at
[`javinukem/turbulence_sr`](https://huggingface.co/javinukem/turbulence_sr) and
fetched on demand:

```bash
python scripts/download_weights.py                 # all models
python scripts/download_weights.py --model edsr_norm_skip_on   # just one
```

Each model folder holds the full training record (`config.yaml`: architecture,
optimizer, loss weights, data split) and its `loss_curve.png`.
`manifest.json` wires every entry into the evaluation stack
(`evaluation/manifest.py::build_model`), and `normalization_stats.npz` holds the
per-channel LR/HR mean/std that all norm-on models share.

## Benchmark (held-out validation set)

| Model | Arch | Loss | Epochs | Best val loss | MSE ×4 | MSE ×2 | Spectral MSE ×4 | PSNR ×4 | SSIM ×4 |
|---|---|---|---|---|---|---|---|---|---|
| `ufno_shift8_mse_spectral_w_minor_500e_clip_p50` | UFNO (op=2, dropout 0.1) | MSE + spectral (w=0.0103) | 500 | 0.0230 | **0.00542** | 0.01067 | **0.0178** | **43.28** | **0.9843** |
| `cfno_shift8_mse_light_l1_500e_clip_p30` | CFNO (shift=8, skip) | MSE + light L1 (w=0.0616) | 500 | 0.0411 | 0.00799 | 0.00856 | 0.0467 | 41.59 | 0.9765 |
| `cfno_shift8_mse_only_500e_clip_p10` | CFNO (shift=8, skip) | MSE | 500 | 0.0341 | 0.00808 | 0.00803 | 0.0499 | 41.55 | 0.9757 |
| `cfno_shift8_mse_light_spectral_500e_clip_p30` | CFNO (shift=8, skip) | MSE + light spectral (w=0.00636) | 500 | 0.0361 | 0.00820 | 0.00836 | 0.0327 | 41.48 | 0.9756 |
| `edsr_norm_skip_on` | EDSR (16 blocks, PReLU, skip) | MSE | 250 | 0.0361 | 0.00847 | — (fixed ×4) | 0.0664 | 41.34 | 0.9751 |
| `cfno_shift8_mse_light_spectral_l1_500e_clip_p30` | CFNO (shift=8, skip) | MSE + light spectral + light L1 | 500 | 0.0457 | 0.00850 | 0.00828 | 0.0404 | 41.33 | 0.9747 |
| Trilinear | — | — | — | — | 0.03716 | 0.03164 | 0.2174 | 34.92 | 0.8753 |

All CFNO models share `FNO_2` (`n_channels=32, n_residual_blocks=3,
n_operator_blocks=2, modes=16, shifting_modes=8`, trilinear skip connection,
~33.7 M params) and differ only in the loss combination — see
`experiments/mse_loss_combinations/` and `experiments/comparing_best_models_mse/`.
Full per-channel metrics: [`benchmark_metrics.csv`](benchmark_metrics.csv).

## Usage

```python
import torch
from evaluation.manifest import load_manifest, load_norm_stats, build_model, get_model_entry

manifest = load_manifest("model_zoo")            # paths resolve relative to model_zoo/
norm = load_norm_stats(manifest, torch.device("cuda"))
entry = get_model_entry(manifest, "ufno_shift8_mse_spectral_w_minor_500e_clip_p50")
run = build_model(entry, torch.device("cuda"), norm)   # handles norm + denorm
sr = run(lr, upsample_factor=4)                        # lr: (B, 5, 32, 32, 32)
```

To re-run the benchmark over the zoo:

```bash
python evaluation/benchmark.py --manifest model_zoo          # GPU required
python evaluation/comparing_models_bar_chart.py --preset model_zoo \
    --output model_zoo/comparing_models_bar_chart_model_zoo.png
```

## Notes

- Run names are historical (e.g. `clip_p10`): the ground-truth training
  hyperparameters are in each folder's `config.yaml`.
- All models were trained on snapshot 79 of each simulation with the
  simulation-level train/val split (`data/split_dataset.py`), Gaussian input
  noise σ=0.01, AdamW + `ReduceLROnPlateau`, effective batch size 64.
- EDSR is fixed-scale (×4 only); CFNO/UFNO take `upsample_factor` as a forward
  argument (evaluated at ×4 and ×2).
- *Maintainer:* upload new/updated weights with `python scripts/upload_weights.py`
  (reads from the scratch training dirs; requires `huggingface-cli login`).
