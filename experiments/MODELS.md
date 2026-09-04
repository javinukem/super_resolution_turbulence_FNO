# Model Catalog — `turbulence_sr`

Reference for future instances: every model variant, where its architecture lives, which configs/experiments use it, its architectural and training particularities, and the three designated **baselines**. `experiments/cfno_2/` and `experiments/edsr/` are deliberately excluded (outdated).

> **Training is consolidated**: "Trained by `experiments/<name>/train_*.py`" below means the experiment's `experiments/<name>/experiment.yaml` run grid, executed by the unified engine `src/training/experiment.py`. The old per-experiment train scripts were removed, and only the 4 published-model-backed experiment folders are retained (`comparing_best_models_mse`, `sfno_loss_combinations_study`, `edsr_norm_skip`, `usfno`); references to other experiment folders are historical (git history / scratch run dirs).

---

## Global conventions

- **"CFNO" = `src/model/models_fno_2.FNO_2` with nonzero `shifting_modes`** (retained Fourier modes shifted away from k=0). It is the same `FNO_2` class — not a separate one.
- **Channel order = jf1uids primitive state.** 3D = 5 channels `[density, vx, vy, vz, pressure]`; 2D = 4 channels. Channel indices should be resolved dynamically via `jf1uids.get_registered_variables` (per AGENTS.md); the positivity ReLU and `SpectralLoss` velocity selection currently hardcode ch0=density / ch(last)=pressure.
- **`n_operator_blocks ≥ 3` is documented-unstable** (NaN/divergence) across all FNO variants (2D and 3D). All live runs cap at 2. The legacy `fno_2_exp4` preset sets op=10 — treat as broken/legacy.
- **`apply_positivity_relu` is coupled to normalization:** norm-on ⟹ positivity ReLU off (else it clips ~half of the zero-mean signal); norm-off ⟹ positivity ReLU on. `FNO_1` (depracated) is always-on; DSFNO is always-off (no flag).
- **Two dataset split schemes exist and are NOT identical:** Hydra pipeline = sample-level `random_split` over snapshots; `data/split_dataset.py` = simulation-level → dedicated HDF5 files. Recent `*_shift8` experiments use the simulation-level split (snapshot 79 only → 400 train / 100 val), via explicit `h5_path=` kwargs at `/export/scratch/jalegria/full_states_h5/full_states{,_val}.h5`.
- **Shared training regime (post-2025-06 experiments):** `AdamW(lr=1e-3, wd=1e-4)`, `ReduceLROnPlateau(factor=0.5, patience=15)`, 250 epochs, early-stop patience 40, `batch_size=8 × grad_accum=8` (eff. 64), AMP off, Gaussian input noise σ=0.01, snapshot index 79 only, eval scales x4 and x2.
- `runs/` held DSFNO grid-search results (25 dirs + `metrics.jsonl`), managed by `experiments/grid_search_dsfno.py` — it is no longer tracked in git (removed in the publish cleanup).
- `src/training/losses.py::loss_cycle` was removed (broken import, zero callers). All live training scripts define their loss inline.

---

## The four baselines

Baselines A, B, and C share the same architecture — `FNO_2` (a.k.a. CFNO) with `shifting_modes=8`, normalized (per-channel mean/std; stats cached at `experiments/l1_spectral_weighting/normalization_stats.npz`), `apply_positivity_relu=False`, `apply_constraint=False`, eval x4 & x2. Baselines A and B use `skip_connection=True` with `interpolation_mode="trilinear"`; Baseline C uses `skip_connection=False`. Baseline A was trained in the `l1_spectral_skip_connection` experiment, Baseline B in `baseline_b_500_epochs` (the stabilized 500-epoch continuation of the original `mse_spectral_weighting` run), and Baseline C in `l1_spectral_weighting`. **Baseline D** is the EDSR reference — `src/model/models_edsr.EDSR` (a different architecture), norm-on with `skip_connection=True`, trained in `edsr_norm_skip`. Baseline D is the only fixed-scale baseline (eval x4 only; no `upsample_factor` forward arg) and the only one trained with `nn.MSELoss()` (no spectral term).

### Baseline A — "fno_2 with skip": `cfno_shift8_skip_trilinear`
- **Trained by:** `experiments/l1_spectral_skip_connection/train_l1_spectral_skip_connection.py`.
- **Bar-chart preset:** `l1_spectral_skip` (in `evaluation/comparing_models_bar_chart.py`).
- **Loss:** `L1SpectralLoss` = `nn.L1Loss() + spectral_weight * SpectralLoss`, with the **`w_minor`** weight from `experiments/l1_spectral_weighting/calibration.json` (`w_minor = ratio * 0.1`). `SpectralLoss`: velocity channels only, 3-D `rfftn(norm="ortho")`, power=|F|², mean over channels, log10, MSE, `avg_pool3d(2)` before FFT.
- **Reported `best_val_loss`:** 0.1397904719 (from `experiments/l1_spectral_skip_connection/summary.json`).
- **CFNO hyperparams:** `n_channels=32, n_residual_blocks=3, n_operator_blocks=2, modes=16, last_layer_kernel=3`.
- **Sibling runs for comparison:** `cfno_shift8_skip_nearest` (skip interp = `nearest`, val 0.1791) and the loaded reference `cfno_shift8_skip_off_ref` (the prior best l1_spectral `w_minor` model with `skip_connection=False`) — all included in the same benchmark/bar-chart/final-snapshot outputs.
- **Prerequisite patch:** the `FNO_2` skip branch in `src/model/models_fno_2.py` was fixed to interpolate `latent` (not the post-FNO `out`) and to omit `align_corners` for `nearest`. The code you read today is the *already-patched* version.

### Baseline B — "MSE+spectral with skip, 500e stabilized": `cfno_shift8_mse_spectral_w_minor_500e_clip_p10`
- **Trained by:** `experiments/baseline_b_500_epochs/train_baseline_b_500.py`; orchestrated by `experiments/baseline_b_500_epochs/run.sh`.
- **Bar-chart preset:** `baseline_b_500_vs_edsr` (in `evaluation/comparing_models_bar_chart.py`).
- **Loss:** `MSESpectralLoss` = `nn.MSELoss() + spectral_weight * SpectralLoss`, with the **`w_minor`** calibration weight (≈0.00636) — same loss as the original `mse_spectral_weighting` run, only the optimizer schedule differs.
- **Training regime (differs from the shared 250-epoch regime):** `AdamW(lr=1e-3, wd=1e-4)`, `ReduceLROnPlateau(factor=0.5, patience=10)`, **500 epochs** (2× the shared budget), early-stop patience 40, **gradient clipping `max_norm=1.0`** (the stabilization knob that distinguishes this run from the original 250-epoch Baseline B), `batch_size=8 × grad_accum=8` (eff. 64), AMP off, Gaussian input noise σ=0.01, snapshot 79, eval x4 & x2.
- **Reported `best_val_loss`:** 0.0358592263 (from the `cfno_shift8_mse_spectral_w_minor_500e_clip_p10` run's `config.json`).
- **Sibling runs for comparison (same `baseline_b_500_epochs` run-group):** `cfno_shift8_mse_spectral_w_minor_500e` (no grad clip, sched_patience=15) and `cfno_shift8_mse_spectral_w_minor_500e_lr3e4_clip_p3` (lr=3e-4, clip=1.0, sched_patience=3 — mirrors the EDSR schedule). The clip_p10 variant is the best-performing of the three on the benchmark MSE (x4: 0.00817 vs 0.01009 vs 0.01104) and is designated the canonical Baseline B.
- **Precursor:** the original 250-epoch `cfno_shift8_mse_spectral_w_minor` from `experiments/mse_spectral_weighting/` was the prior Baseline B. It is now treated as a sibling/precursor, not the baseline. The `mse_spectral_weighting` experiment's `w_equal` / `w_major` runs remain the spectral-weight sweep references (preset `mse_spectral`).
- **vs Baseline A:** identical model, identical skip config, identical normalization, identical `w_minor` spectral weight. The deliberate axes are **MSE vs L1** as the pixel term **and** the stabilized 500-epoch + grad-clip schedule.

### Baseline C — "L1+spectral (no skip, w_minor)": `cfno_shift8_l1_spectral_w_minor`
- **Trained by:** `experiments/l1_spectral_weighting/train_l1_spectral.py`.
- **Bar-chart preset:** `spectral_weighting_l1_vs_mse`.
- **Loss:** `L1SpectralLoss` = `nn.L1Loss() + spectral_weight * SpectralLoss`, with the **`w_minor`** calibration weight (the same `w_minor = ratio * 0.1` used by Baseline A).
- **`skip_connection=False`** — the only baseline without the skip branch. This is the historical precursor to Baselines A and B (the first norm-on CFNO experiment before the skip was added to FNO_2).
- **Sibling runs for comparison:** `cfno_shift8_l1_spectral_w_equal` (w_equal) and `cfno_shift8_l1_spectral_w_major` (w_major) — all three included in the same benchmark/bar-chart outputs.

### Baseline D — "EDSR norm-on with skip": `edsr_norm_skip_on`
- **Trained by:** `experiments/edsr_norm_skip/train_edsr_norm_skip.py`; orchestrated by `experiments/edsr_norm_skip/run.sh`.
- **Bar-chart preset:** `edsr_norm_skip` (in `evaluation/comparing_models_bar_chart.py`); also cross-compared via `baseline_b_500_vs_edsr` and `baselines_vs_edsr_norm_skip`.
- **Architecture:** `src/model/models_edsr.EDSR` — **a different class from Baselines A/B/C** (`FNO_2`/CFNO). `n_resblocks=16, n_feats=64, kernel_size=3, scale=4, activation_f="prelu"`, `skip_connection=True` with `interpolation_mode="trilinear"`, `apply_positivity_relu=False`. Fixed scale (no `upsample_factor` forward arg) → eval **x4 only**.
- **Loss:** plain `nn.MSELoss()` on normalized states — **no spectral term** (the only baseline without one).
- **Training regime (EDSR-specific, mirrors `trying_new_losses_and_norm`):** `AdamW(lr=3e-4, wd=1e-4)`, `ReduceLROnPlateau(factor=0.5, patience=3)`, 250 epochs, early-stop patience 40, **gradient clipping `max_norm=1.0`**, `batch_size=8 × grad_accum=8` (eff. 64), AMP off, Gaussian input noise σ=0.01, snapshot 79. Norm-on reusing `l1_spectral_weighting/normalization_stats.npz`.
- **Reported `best_val_loss`:** 0.0360629842 (MSE on normalized, from `config.json`).
- **Sibling run:** `edsr_norm_skip_off` (`skip_connection=False`, the norm-on no-skip reference, `best_val_loss ≈ 0.0359`) — both included in the same `edsr_norm_skip` bar-chart output. The skip-on variant is designated the canonical EDSR baseline (Baseline D); the two are within noise on the benchmark MSE (x4: 0.00847 on vs 0.00839 off).
- **Role:** the non-CFNO reference. Exposes how a purely-convolutional fixed-scale SR architecture compares to the CFNO baselines under the shared norm-on, snapshot-79, grad-clipped regime. Note the loss-function confound (EDSR uses MSE-only; Baselines A/B/C use L1/MSE + spectral).

### What distinguishes the four baselines
- Baseline A: L1 + spectral with skip (`l1_spectral_skip_connection` experiment).
- Baseline B: MSE + spectral with skip, **500-epoch + grad-clip=1.0 stabilized** (`baseline_b_500_epochs` experiment, `clip_p10` run, w_minor weight). The best performing so far 04.07.
- Baseline C: L1 + spectral **without** skip (`l1_spectral_weighting` experiment, w_minor weight — the no-skip reference).
- Baseline D: EDSR, MSE-only (no spectral), norm-on with skip (`edsr_norm_skip` experiment, `edsr_norm_skip_on`). The only non-CFNO and the only fixed-scale baseline.
- Baselines A, B, C share the same `w_minor` spectral weight and the same `FNO_2` architecture (`n_channels=32, n_residual_blocks=3, n_operator_blocks=2, modes=16`). Baseline B additionally extends the training budget to 500 epochs with gradient clipping. The four-way comparison exposes the effect of the pixel-loss base (L1 vs MSE), the skip branch, the training-budget/optimizer-stability axis, and the architecture class (CFNO vs EDSR).

---

## Models

### 1. FNO_2 (a.k.a. CFNO) — flagship
- **Architecture file:** `src/model/models_fno_2.py:95` (class `FNO_2`); `SoftmaxConstraint` at `:30`; `FNOBlock` at `:53`; `_upsample_3d` at `:15`.
- **Config:** **no `configs/model/fno_2.yaml` exists** — recent 3D experiments instantiate `FNO_2` directly in their `train_*.py`, bypassing Hydra composition. (The live `configs/model/fno_2_exp4.yaml` points at the *legacy* `FNO_2` inside `experiments/cfno_2/` — see Legacy entry below.)
- **Experiments using it:** `training_best_models_experiment/`, `trying_new_losses_and_norm/`, `l1_spectral_weighting/`, `l1_spectral_skip_connection/` ★ (Baseline A), `loss_ablation/`, `mse_spectral_weighting/`, `baseline_b_500_epochs/` ★ (Baseline B), `train_fno2_grid_modes_interp_skip_losses_refine/`.
- **Instantiation in bar-chart engine:** `evaluation/manifest.py` `model_type="cfno"` → `from src.model.models_fno_2 import FNO_2`.

**Architectural particularities**
- `conv1` (3×3×3 Conv, in→n_channels, no ReLU) → `n_residual_blocks` × `ResBlock` (from `models_edsr.ResBlock`: 3D conv-relu-conv + skip) → **LR latent = x1 + x2 (global residual)** → `_upsample_3d(latent, upsample_factor, interpolation_mode)` → `n_operator_blocks` × `FNOBlock` (SpectralConv3d + 1×1 Conv + add + ReLU) → optional `skip_connection` residual → `tail` (3D Conv, n_channels→in_channel, kernel=`last_layer_kernel`, padding="same") → optional `SoftmaxConstraint` → optional positivity ReLU on ch0 (density) and ch4 (pressure).
- **`upsample_factor` is a forward arg** (`forward(x, upsample_factor)`) → variable-scale SR. Upsampling happens **before** the FNO blocks (upsample-then-transform; arXiv:2305.14452 / "CFNO" style).
- **`skip_connection` branch** (ctor default `skip_connection=False`): upsamples the LR `latent` (NOT post-FNO `out`) with `skip_connection_interpolation_mode` and adds it to `out` before the tail. Off by default — recent baselines set it True. Historically buggy (interpolated `out`; caused shape mismatch); fixed in the `l1_spectral_skip_connection` prerequisite patch.
- **`interpolation_mode`** (default `"trilinear"`, `align_corners=False`): main upsample mode. The `_upsample_3d` helper omits `align_corners` for `nearest`/`area`/`nearest-exact`.
- **`shifting_modes`**: shifts the retained `modes` away from k=0 (the "CFNO shift=N" knob).
- **`SoftmaxConstraint`** (arXiv:2208.05424): `avg_pool3d(SR)` / `LR` ratio, Kronecker-tiled, multiplied back into SR → pooled-SR ≈ LR. Gated by `apply_constraint`.
- **Positivity:** optional final ReLU on ch0 & ch4, gated by `apply_positivity_relu` (default True). Disabled under per-channel normalization.
- Common 3D hyperparams across recent experiments: `n_channels=32, n_residual_blocks=3, n_operator_blocks=2, modes=16, last_layer_kernel=3, apply_constraint=False` (~33.7 M params).

**Key FNO_2 experiment — `l1_spectral_weighting` (no-skip L1+spectral sweep):**
  The first norm-on CFNO experiment. Trains `cfno_shift8_l1_spectral_w_minor/w_equal/w_major` with `skip_connection=False`, `L1SpectralLoss` at three spectral weights from calibration. All three runs and a trilinear baseline are benchmarked together. Bar-chart presets: `spectral_weighting_l1_vs_mse` (with MSE+spectral counterparts and loss-ablation singles) and the legacy `l1_spectral_best_vs_best_performing` (best two vs best_performing). The `w_minor` entry serves as **Baseline C** (the no-skip L1+spectral reference).

**Key FNO_2 experiment — `loss_ablation` (single-loss ablation):**
  Trains three CFNO shift=8 runs with `skip_connection=True` differing ONLY in loss: `cfno_shift8_l1_only` (L1), `cfno_shift8_mse_only` (MSE), `cfno_shift8_spectral_only` (velocity-only spectral, no pixel term). Shows the isolated effect of each loss component. Bar-chart preset: `spectral_weighting_l1_vs_mse` (these appear as "L1 only (no spectral)" and "MSE only (no spectral)"; spectral-only is included but omitted from bar presets due to poor standalone performance).

### 2. EDSR
- **Architecture file:** `src/model/models_edsr.py:99` (class `EDSR`); `ResBlock` at `:25`; `ResBlock2d` at `:46` (used by the 2D FNO); `Upsampler` at `:67`; `_upsample_3d` at `:17`.
- **Config (legacy):** `configs/model/cnn.yaml` → `module: src.model.models_edsr`, `class_name: EDSR`, params `input_channels=5, n_resblocks=5, n_feats=64, kernel_size=3, scale=4, activation_f=relu`. Preset `configs/preset/cnn.yaml` composes `model=cnn / data=train_3d_cnn / training=cnn / runtime=cnn`. The `edsr_norm_skip` experiment overrides these: `n_resblocks=16, activation_f="prelu"`.
- **Experiments:** `training_best_models_experiment/`, `trying_new_losses_and_norm/`, `edsr_norm_skip/` ★ (Baseline D).
- **Instantiation:** `evaluation/manifest.py` `model_type="edsr"` → `from src.model.models_edsr import EDSR`.

**Architectural particularities**
- `head` (3D Conv in→n_feats) → `body` = `n_resblocks` × `ResBlock` + closing 3D Conv (n_feats→n_feats) → **global residual `res = body(x) + x`** → `tail` = `Upsampler(scale, n_feats, activation_f)` + 3D Conv (n_feats→in_channel). Optional positivity ReLU on ch0 & ch4.
- **`Upsampler` (fixed-scale):** `PixelShuffle3d` × `log2(scale)` stages (2 stages for scale=4); `Conv3d(n_feats → 8*n_feats)` then `PixelShuffle3d(2)` per stage. Optional `batch_norm` and optional `activation_f ∈ {None, "relu", "prelu"}` between stages. **No `upsample_factor` forward arg** — `scale` is baked in (fixed-scale; the manifest forward passes no upscale arg, `supports_variable_scale` distinguishes this).
- `activation_f=False` (original setup) removes inter-stage nonlinearity → blocky outputs; `trying_new_losses_and_norm` introduced `activation_f="prelu"` to fix the blockiness. The `edsr_norm_skip` experiment uses `prelu` throughout.
- **Skip connection:** `skip_connection: bool` ctor flag (default False). When True, upsamples the post-body `res` (at n_feats channels) with `skip_connection_interpolation_mode` and adds it to the HR latent **before** the final channel projection. Implemented by indexing `self.tail` so non-skip `state_dict` keys remain loadable. `_skip_scale = scale`.

**Key EDSR experiment — `edsr_norm_skip` (norm-on, 250-ep, skip off/on):**
  Two runs with normalization ON, reusing `l1_spectral_weighting/normalization_stats.npz`:
  1. `edsr_norm_skip_off` — `skip_connection=False` (norm-on baseline, `best_val_loss ≈ 0.0359` MSE-on-normalized).
  2. `edsr_norm_skip_on` — `skip_connection=True, interpolation_mode="trilinear"` (skip variant, `best_val_loss ≈ 0.0361` MSE-on-normalized).
  Both use `n_resblocks=16, n_feats=64, activation_f="prelu"`, the shared 250-epoch training regime (`ReduceLROnPlateau(factor=0.5, patience=15)`, `lr=3e-4` vs the CFNO `lr=1e-3`, gradient clipping at 1.0), `apply_positivity_relu=False`, snapshot 79, eval x4 only (fixed-scale). Bar-chart preset: `edsr_norm_skip` (in `evaluation/comparing_models_bar_chart.py`).

  On MSE-derived benchmark metrics at x4 these runs are competitive with (or beat) the CFNO baselines, but the comparison is loss-function-dependent — EDSR trains with `nn.MSELoss()`, the baselines with L1+spectral or MSE+spectral. The spectral and vorticity metrics show mixed results (EDSR-on beats Baseline A on `Spectral_MSE` at x4; EDSR-off loses). An earlier scratch attempt diverged (val_loss 25.7), confirming EDSR instability is still operative — the published results required a second training attempt.

**Caveats:** Extremely unstable for 3D turbulence — only 1/12 grid configs converged in the original norm-OFF `training_best_models` grid (best val_loss 0.138 vs CFNO 0.00125, ~110× worse). The norm-ON, 250-epoch `edsr_norm_skip` regime fixes the undertraining and yields functional models, but instability persists (first training attempt diverged). Fixed scale: cannot eval at x2 by changing a forward arg. The EDSR param count (~single-digit millions for 16×64 3D Conv) is not persisted on disk — only printed at runtime by `train_edsr_norm_skip.py:290`.

### 3. UFNO_2 (U-FNO + U-Net path)
- **Architecture file:** `src/model/models_ufno_2.py:174` (class `UFNO_2`); `UFNOBlock` at `:126`; `UNet3d` at `:87`; `Conv3dBlock` at `:56`, `Deconv3dBlock` at `:75`; own `SoftmaxConstraint` at `:28`; `_upsample_3d` at `:13`.
- **Config:** no Hydra `configs/model/` entry. Instantiated directly in `experiments/ufno_l1_spectral_unet/train_ufno_l1_spectral_unet.py`. A `model_type="ufno"` branch was added to `evaluation/manifest.py` (`from src.model.models_ufno_2 import UFNO_2`) as a prerequisite patch.
- **Experiment:** `ufno_l1_spectral_unet/` only. Bar-chart preset `ufno`.

**Architectural particularities**
- Same skeleton as FNO_2 (conv1 → ResBlocks → LR latent = x1+x2 → upsample → operator blocks → optional skip → tail → optional constraint → optional positivity ReLU). Same ctor signature plus `dropout_rate` (default 0.0).
- **Difference:** each operator block is `UFNOBlock` = `SpectralConv3d` (x1) + 1×1 Conv (x2) + **parallel `UNet3d` path (x3)**, summed then ReLU. The U-Net path follows Wen et al. / RT-JAX (`RuneRost/RT-JAX .../ufno_3d.py`).
- `UNet3d`: 3 stride-2 `Conv3dBlock`s (conv → `GroupNorm(num_groups=1)`=instance-norm → `leaky_relu(0.1)` → `Dropout(p=dropout_rate)`), mirrored `ConvTranspose3d` upsampling with **concatenation skip**; final 3×3 conv projects back to `channels`. **Encoder downsamples by 8** → input spatial size must be divisible by 8.
- Skip connection enabled (same `latent`-interpolate logic as FNO_2).

**Training:** mirrors Baseline A reference (L1 + spectral `w_minor`, normalized, 250 ep / patience 40, snapshot 79, eval x4 & x2). 2×2 sweep: `n_operator_blocks ∈ {1,2}` × `dropout_rate ∈ {0.0,0.1}` → `ufno_shift8_op1_drop0`, `ufno_shift8_op1_drop01`, `ufno_shift8_op2_drop0`, `ufno_shift8_op2_drop01`.

**Caveats:** input spatial size must be divisible by 8; memory heavier than FNO_2 due to the per-block U-Net.

### 4. DSFNO (Downscaling FNO, Yang et al.) — 3D variant
- **Architecture file:** `src/model/models_dsfno_3d.py:354` (class `DSFNO`); own `SoftmaxConstraint` at `:15` (note **stray duplicate `return out`** at `:36` — dead code, harmless); own `SpectralConv3d` at `:42`; `OperatorBlock` at `:273` (uses **GELU**, no `shifting_modes`); `ResidualBlock` at `:314` (bias=False).
- **Dead siblings:** `src/model/models_dsfno_3d_noncomplex.py:213` (`DSFNO`, real-only port — unreferenced) and `src/model/models_dsfno.py:213` (`DSFNO`, alternate — unreferenced). Don't wire new code through them.
- **Config:** `configs/model/dsfno.yaml` → `module: src.model.models_dsfno_3d`, `class_name: DSFNO`, params `in_channel=5, modes=10, n_channels=32, n_residual_blocks=3, n_operator_blocks=2, apply_constraint=true`. Preset `configs/preset/dsfno.yaml` composes `model=dsfno / data=train_3d_dsfno / training=dsfno / runtime=dsfno`.
- **Experiment:** none under `experiments/`. The standalone `experiments/grid_search_dsfno.py` (Hydra) runs an 81-combo grid (modes × n_channels × n_residual_blocks × n_operator_blocks) → output `runs/dsfno_grid/` (25 dirs + `runs/dsfno_grid/metrics.jsonl`). Config: `configs/experiments/grid_search_dsfno.yaml`.

**Architectural particularities**
- `conv1` (Conv3d + ReLU) → `n_residual_blocks` × `ResidualBlock` (bias=False; ReLU(inplace)) → `conv2` (Conv3d + ReLU) → **trilinear** interpolate (no interp knob) → (`n_operator_blocks-1`) OperatorBlocks(GELU) + final OperatorBlock(no activation) → permute → **`fc1` (Linear n_channels→128) + GELU + `fc2` (128→in_channel)`** with `torch.utils.checkpoint` (gradient checkpointing, `use_reentrant=False`) → optional `SoftmaxConstraint`.
- No `last_layer_kernel`, no positivity ReLU on density/pressure, no skip-connection branch, no `shifting_modes` on the `DSFNO` ctor (`OperatorBlock`'s internal `SpectralConv3d` defaults `shifting_modes=0`). Upsample is a forward arg.

**Caveats:** no positivity constraint on output — physically inconsistent density/pressure unless `apply_constraint=true` does the pooling-based fix.

### 5. FNO_2 (2D variant)
- **Architecture file:** `src/model/models_fno2_2d.py:73` (class `FNO_2`); `FNOBlock` at `:32`; `SoftmaxConstraint` at `:10` (uses `exp_factor` and `torch.exp` — a different formulation than the 3D one); uses `ResBlock2d` from `models_edsr.py` and `SpectralConv2d` from `src/model/fno_layer.py`.
- **Config:** `configs/model/fno_2_2d.yaml` → `module: src.model.models_fno2_2d`, `class_name: FNO_2`, params `in_channel=4, modes=16, n_channels=32, n_residual_blocks=2, n_operator_blocks=2, apply_constraint=false, shifting_modes=3, last_layer_kernel=3`. Preset `configs/preset/fno_2_2d.yaml` composes `data=train_2d_fno2 / training=fno_2_2d / runtime=fno_2_2d`.
- **Experiment:** `experiments/fno_2d_grid_search/` (`grid_search_fno2_2d.py`, `experiment_summary.md`, 10 trained configs).

**Architectural particularities**
- 2D mirror: Conv2d lift → `n_residual_blocks` × `ResBlock2d` → latent = x1+x2 → **bilinear** interpolate (no interp knob) → FNO blocks → tail Conv2d → optional `SoftmaxConstraint` (exp-based 2D version) → **hardcoded** `out[:,0]=exp(...); out[:,3]=exp(...)` (density ch0 & pressure ch3 → `exp`, unconditional, no flag).
- 4 channels (2D jf1uids primitive state). No `skip_connection`, no `interpolation_mode`, no `apply_positivity_relu`.

**Caveats:** the grid's best run used `modes=8`, NOT the `modes=16` from the Hydra yaml. ≥3 operator blocks → NaN/divergence in 2D too. 64 channels exploded (loss ~10¹⁹).

### 6. FNO_1 (DEFAULT preset; lives in `depracated/`)
- **Architecture file:** `depracated/models_fno_1.py:312` (class `FNO_1`); own `SoftmaxConstraint` at `:12`; own `SpectralConv3d` at `:38`; `OperatorBlock` at `:225` (uses two 1×1 convs `w1`/`w2` + double ReLU, then `x1+x` residual); `ResidualBlock` at `:272` (bias=True; ReLU(inplace)).
- **Config:** `configs/model/fno_1.yaml` → `module: depracated.models_fno_1`, `class_name: FNO_1`, params `in_channel=5, modes=20, n_channels=64, n_residual_blocks=3, n_operator_blocks=2, apply_constraint=true, shifting_modes=4`. **This is the DEFAULT preset** (`configs/preset/fno_1.yaml` → `data=train_3d_fno1 / training=fno_1 / runtime=fno_1`). Running `python train.py` with no `preset=` uses `fno_1`.
- **Experiments:** none of the modern (non-ignored) experiment folders train FNO_1.

**Architectural particularities**
- conv1 (Conv3d, no ReLU) → `n_residual_blocks` × `ResidualBlock` (bias=True) → latent = x1+x2 → **trilinear** interpolate (no interp knob) → `n_operator_blocks` × `OperatorBlock` (SpectralConv3d + w1 + ReLU + w2 + ReLU + residual add; **no further skip**) → tail (Conv3d 3×3, same) → optional `SoftmaxConstraint` → **unconditional** ReLU on ch0 & ch4 (hardcoded — no `apply_positivity_relu` flag; equivalent to always-on).
- No `last_layer_kernel`, no `skip_connection`, no `interpolation_mode`. `OperatorBlock` accepts `shifting_modes` (unlike DSFNO's). Upsample is a forward arg.

**Caveats:** per AGENTS.md, `depracated/` must NOT be deleted — it is a live dependency of the default preset. Positivity ReLU is hardcoded — cannot do the norm-on regime with this class.

### 7. Legacy FNO_2 (`fno_2_exp4`)
- **Architecture file:** `experiments/cfno_2/experiment_4/fno_2_exp_2.py:82` (class `FNO_2`); own `SoftmaxConstraint` at `:12`, `FNOBlock` at `:35`, `Exp` module at `:77`. (Inside the ignored `experiments/cfno_2/` folder, but referenced by a live Hydra config so it's a live dependency.)
- **Config:** `configs/model/fno_2_exp4.yaml` → `module: experiments.cfno_2.experiment_4.fno_2_exp_2`, `class_name: FNO_2`, params `in_channel=5, modes=26, n_channels=52, n_residual_blocks=1, n_operator_blocks=10, shifting_modes=6, last_layer_kernel=3, last_layer_constraint=relu`. Preset `configs/preset/fno_2_exp4.yaml`.
- **Load path:** `src/utils/pipeline.py::load_model_from_folder` has a legacy branch (top-level `fno_2:` key → `experiments.cfno_2.experiment_4.fno_2_exp_2`) that reconstructs models saved from this old config. Preserve it.

**Caveats:** `n_operator_blocks=10` contradicts the documented "≥3 NaN/divergent" finding — treat as broken/legacy. This older `FNO_2` is a *different implementation* from `src/model/models_fno_2.py` (no `skip_connection`, no `interpolation_mode`, no `apply_positivity_relu`, no `shifting_modes` on the modern signature). Marked outdated per the user's exclusion of `experiments/cfno_2/`.

---

## Cross-reference matrix (model → experiment folders)

| Model (class@file) | training_best_models | trying_new_losses_and_norm | l1_spectral_weighting | l1_spectral_skip_connection | loss_ablation | mse_spectral_weighting | baseline_b_500_epochs | edsr_norm_skip | ufno_l1_spectral_unet | train_fno2_grid_modes_interp_skip_losses_refine | fno_2d_grid_search | grid_search_dsfno (runs/) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FNO_2 (`src/model/models_fno_2.py`) CFNO | ✓ (shifts 0–16, no skip, MSE, norm-off) | ✓ (shift8, 3-loss×2-norm, no skip) | ✓ **Baseline C** (shift8, no skip, L1+spec `w_minor`, norm-on) | ✓ **Baseline A** (shift8, skip, L1+spec `w_minor`, norm-on) | ✓ (shift8, skip, single-loss ablation l1/mse/spectral) | ✓ (shift8, skip, MSE+spec `w_minor`, 250e — precursor to Baseline B) | ✓ **Baseline B** (shift8, skip, MSE+spec `w_minor`, 500e clip_p10, norm-on) | — | — | ✓ (FNO_2 wrapped, modes×interp×skip grid, shift=8) | — | — |
| EDSR (`src/model/models_edsr.py`) | ✓ (rb16/f64, act_f=False, MSE, norm-off) | ✓ (same + `prelu`, norm grid) | — | — | — | — | — | ✓ **Baseline D** (`edsr_norm_skip_on`, norm-on, `prelu`, skip on) + off sibling | — | — | — | — |
| UFNO_2 (`src/model/models_ufno_2.py`) | — | — | — | — | — | — | — | ✓ (shift8, skip, L1+spec, op×dropout sweep) | — | — | — |
| DSFNO 3D (`src/model/models_dsfno_3d.py`) | — | — | — | — | — | — | — | — | — | — | ✓ (81-combo → `runs/dsfno_grid/`) |
| FNO_2 2D (`src/model/models_fno2_2d.py`) | — | — | — | — | — | — | — | — | — | ✓ (10 runs) | — |
| FNO_1 (`depracated/models_fno_1.py`) | — (history only) | — | — | — | — | — | — | — | — | — | — |
| Legacy FNO_2 (`experiments/cfno_2/experiment_4/fno_2_exp_2.py`) | — (legacy/outdated) | — | — | — | — | — | — | — | — | — | — |

## Live Hydra presets → model classes

| `preset=` | model yaml | module.class | notes |
|---|---|---|---|
| `fno_1` (DEFAULT) | `configs/model/fno_1.yaml` | `depracated.models_fno_1.FNO_1` | 5-ch 3D, hardcoded positivity ReLU, modes=20 nc=64 res=3 op=2 shift=4, apply_constraint=true |
| `cnn` | `configs/model/cnn.yaml` | `src.model.models_edsr.EDSR` | 5-ch 3D, n_resblocks=5 n_feats=64 scale=4 activation_f=relu |
| `dsfno` | `configs/model/dsfno.yaml` | `src.model.models_dsfno_3d.DSFNO` | 5-ch 3D, modes=10 nc=32 res=3 op=2 apply_constraint=true |
| `fno_2_2d` | `configs/model/fno_2_2d.yaml` | `src.model.models_fno2_2d.FNO_2` | 4-ch 2D, modes=16 nc=32 res=2 op=2 shift=3 apply_constraint=false |
| `fno_2_exp4` | `configs/model/fno_2_exp4.yaml` | `experiments.cfno_2.experiment_4.fno_2_exp_2.FNO_2` | LEGACY, op=10 (unstable per docs), modes=26 nc=52 res=1 shift=6, last_layer_constraint=relu |

The flagship modern `FNO_2` (`src/model/models_fno_2.py`) has **no Hydra `configs/model/` entry** — recent experiments instantiate it directly and bypass the model Hydra group.

---

## Shared code notes (gotchas)
- `SoftmaxConstraint` (arXiv:2208.05424) exists in **4 near-duplicate copies**: `models_fno_2.py:30`, `models_ufno_2.py:28`, `models_dsfno_3d.py:15` (with the stray duplicate `return out` at `:36`), `depracated/models_fno_1.py:12`, plus a different `exp`-based 2D version at `models_fno2_2d.py:10`.
- `src/training/losses.py::loss_cycle` is broken — don't call it.
- `load_model_from_folder` (`src/utils/pipeline.py`) keeps a legacy branch for `fno_2_exp4`-style saved configs — preserve it when editing.

---

## Published models → `experiments/`

The 6 published models (Baseline-family CFNO MSE-variants, EDSR skip-on, UFNO
clip_p50) are centralized under `experiments/`: per-model `config.yaml`
training records + `loss_curve.png` live in their source experiment folders
(e.g. `experiments/comparing_best_models_mse/sfno/`),
the portable `experiments/manifest.json` wires them into the evaluation stack,
and the full benchmark is `experiments/benchmark_metrics.csv`. Weights are on
HuggingFace (fetched via `scripts/download_weights.py` into
`experiments/<experiment>/<name>/`). Benchmark + bar chart:
`python evaluation/benchmark.py --manifest experiments` and
`python evaluation/comparing_models_bar_chart.py --manifest experiments`.
