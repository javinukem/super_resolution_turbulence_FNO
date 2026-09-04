# turbulence_sr

**Super-resolution of compressible turbulence with Fourier neural operators.**
Learns a 4× upsampling operator mapping low-resolution (32³) fluid states to
high-resolution (128³) ones. Training data is synthesized with
[**jf1uids**](https://doi.org/10.5281/zenodo.15052815) (a JAX Euler solver with
Kolmogorov-spectrum forcing); the ML side is entirely PyTorch.

![Snapshot comparison](experiments/usfno/final_snapshot_comparison.png)

![Zoomed comparison](experiments/usfno/zoomed_snapshot_comparison.png)

## Highlights

- **SFNO (`FNO_2`)**: Conv-lift → ResBlocks → upsample → FNO operator stack,
  with `upsample_factor` as a forward argument (variable-scale SR).
- **USFNO**: operator blocks with a parallel 3-D U-Net path — the
  best-performing model (**MSE ×4 = 0.00542**, ~6.9× better than trilinear).
- **EDSR**: the purely-convolutional reference point.
- Physics-aware evaluation: per-channel MSE, velocity-norm/vorticity losses,
  1-D power-spectrum MSE, 3-D perceptual metric, PSNR, SSIM.

## Published models

Six published models (configs and loss curves under
`experiments/<experiment>/<model>/`, weights on HuggingFace). Full catalog in
[`experiments/MODELS.md`](experiments/MODELS.md).

| Model | Loss | MSE ×4 | PSNR ×4 | SSIM ×4 |
|---|---|---|---|---|
| USFNO op=2 drop=0.1 | MSE + spectral | **0.00542** | **43.28** | **0.9843** |
| SFNO MSE + light L1 | MSE + 0.062·L1 | 0.00799 | 41.59 | 0.9765 |
| SFNO MSE only | MSE | 0.00808 | 41.55 | 0.9757 |
| SFNO MSE + light spectral | MSE + 0.0064·spectral | 0.00820 | 41.48 | 0.9756 |
| EDSR (skip on) | MSE | 0.00847 | 41.34 | 0.9751 |
| Trilinear baseline | — | 0.03716 | 34.92 | 0.8753 |

```bash
git clone https://github.com/javinukem/turbulence_sr.git
cd turbulence_sr
pip install -r requirements.txt
python scripts/download_weights.py   # fetch weights from HF into experiments/
```

> `jf1uids` and `autocvd` are not on PyPI — install from source only if you
> need dataset generation / physics metrics / GPU device pinning.

## Usage

```bash
# evaluate a published model (GPU required)
python evaluation/benchmark.py --manifest experiments

# train (Hydra presets: dsfno, cnn, fno_2_2d; default: fno_1)
python train.py preset=dsfno

# run an experiment (unified engine, declarative YAML)
bash experiments/edsr_norm_skip/run.sh

# tests (GPU-free smoke tests)
python -m pytest tests/ -v
```

## Data

HDF5 with keys `hr_states` `(N, 5, 128, 128, 128)` and `lr_states`
`(N, 5, 32, 32, 32)` (channels follow the jf1uids primitive order
`[density, vx, vy, vz, pressure]`). The full dataset is 500 simulations ×
80 snapshots = 40 000 pairs (≈650 GB — not redistributed); regenerate it with
the jf1uids-based scripts under `src/dataset_generation/`.

## Repository layout

```
configs/            Hydra config tree (presets compose model+data+training)
evaluation/         benchmark.py, manifest.py, comparing_models_bar_chart.py
experiments/        one folder per experiment; published models + manifest.json
src/                model/, training/, dataloader/, dataset_generation/, utils/
tests/              unittest smoke tests (pytest-compatible)
train.py            Hydra training entrypoint
```

## Known limitations

- Two train/val split schemes coexist (sample-level vs simulation-level) —
  see `experiments/experiment_summary.md`.
- `n_operator_blocks ≥ 3` diverges (NaN); published configs cap it at 2.
- Some scripts carry hardcoded cluster paths (`/export/...`); pass explicit
  `h5_path=` / `--manifest` arguments elsewhere.

## Citation & license

See [`CITATION.cff`](CITATION.cff). Released under [MIT](LICENSE).
