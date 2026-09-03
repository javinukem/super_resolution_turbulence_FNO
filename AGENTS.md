# AGENTS.md

`turbulence_sr` — PyTorch super-resolution for compressible turbulence. Learns a 4× upsampling operator from low-resolution (LR) to high-resolution (HR) fluid states. The training data comes from **jf1uids** (the externally-named "astronomix" JAX Euler solver) — so JAX/jf1uids is used **only** for dataset generation and physics-aware evaluation metrics (energy spectra, total quantities). The ML side is **entirely PyTorch**. If a task mentions JAX, you are almost certainly in the dataset-generation or evaluation path, not the training path.

## Tests are unittest, ran with pytest

- `tests/**` uses `unittest.TestCase` classes — runnable with both `python -m pytest tests/ -v` and `python -m unittest discover tests`. There is no pytest config file (no `pytest.ini`, `setup.cfg`, `conftest.py`); default discovery applies.
- `test_pipeline_helpers.py` exercises `src/utils/pipeline.py` with temp dirs and a dynamically injected dummy module — no GPU, no real data.
- `test_config_and_cli_smoke.py` smoke-tests Hydra composition by running `python <script> --cfg job` (composes config and exits, no training) for `train.py` and every experiment entrypoint. These import `omegaconf`, `yaml`, `autocvd` — make sure the config compose path stays import-clean.
- Don't add a pytest-only layer without asking; don't break the `unittest`-runnable contract of these files.

## Runtime gotchas

- **`autocvd` first, always (when touching a GPU).** Any script that will hit CUDA must, before any `torch`/`jax`/`cuda` import, do:
  ```python
  from autocvd import autocvd
  autocvd(num_gpus=1)
  ```
  Skipping yields wrong-device errors. The late import triggers `E402`, which is intentionally ignored by the preset `ruff` config (`pyproject.toml`). In `train.py` this is gated by `cfg.runtime.autocvd.enabled` — do not remove the gate, it lets the CLI smoke tests run GPU-free.
- **Notebooks are discouraged for GPU runs.** The `dataset_generation/*` scripts carry an explicit warning: notebooks block GPU memory if not reset properly. Prefer scripts.
- **`jf1uids` and `autocvd` are not on PyPI.** `pyproject.toml` only declares `hydra-core>=1.3`; `requirements.txt` documents the full environment (torch, jax, h5py, astropy, `Pk_library`, optuna, pandas, ...) for humans, but is intentionally unpinned — the cluster env provides it. Don't add jf1uids/autocvd as declared deps without asking.
- **Deeper FNO operator stacks (≥3 blocks) go NaN/divergent.** This is a documented experimental finding; configs cap `n_operator_blocks` at 2. If you raise it, expect instability.

## Two config systems — do not conflate them

- **Modern (live):** the Hydra tree under `configs/`. Entry: `@hydra.main(version_base="1.3", config_name="config")`. Groups: `model/`, `data/`, `training/`, `runtime/`, `preset/`, plus standalone `experiments/`, `evaluation/`, `figures/` for non-`train.py` scripts. Hydra is set to **not** chdir (`job.chdir: false`) and **not** write the `.hydra/` subdir (`output_subdir: null`) — output is managed manually via `runtime.output`.
- **Legacy:** the top-level `config.yaml` (flat, `gpu`/`training`/`dsfno`/`fno_1`/`fno_2`/`data`/`turbulent_sim:` blocks). It is read **only** by `src/dataset_generation/*` scripts and old notebooks. Edit it only when changing simulation parameters for data generation. Do not wire new training code through it.
- Use **presets** for end-to-end training: `python train.py preset=dsfno` (or `cnn`, `fno_2_2d`, default `fno_1`). Presets are `_global_`-package files that override model+data+training+runtime together.
- **The default preset (`fno_1`) imports its model from `depracated/`** (`configs/model/fno_1.yaml` → `module: depracated.models_fno_1`). Do not delete `depracated/` — it is a live dependency of the default config.

## Models — relevance & gotchas

- Architectures live in `src/model/` (`models_fno_2.py` is the flagship CFNO/FNO_2; also `models_ufno_2.py` UFNO, `models_edsr.py` EDSR, `models_dsfno*.py` DSFNO variants, `models_fno2_2d.py` 2D variant, `depracated/models_fno_1.py`).
- **Channel order = jf1uids primitive state.** 3D = 5 channels `[density, vx, vy, vz, pressure]`; 2D = 4 channels. Density (ch0) and pressure (last) are positivity-constrained via a final ReLU in the model. Indices are resolved dynamically via `jf1uids.get_registered_variables` in `evaluation/benchmark.py` — do not hardcode them in new code.
- **`upsample_factor` is a forward arg** on FNO_2 (variable-scale SR). Some configs train/eval at multiple shifts `{0,4,8,12,16}` at x4 and x2.
- **`SoftmaxConstraint`** (arXiv:2208.05424) optionally enforces that pooled-SR matches LR. Keep `last_layer_constraint` / `apply_constraint` params consistent between training and eval configs.
- Stray duplicate `return out` at `src/model/models_dsfno_3d.py:36` is dead code after an early return — harmless, indicative of copy-paste.
- **`src/training/losses.py` was removed** (its only function, `loss_cycle`, had an unfixable import and zero callers). Training loops default to `nn.MSELoss()`; experiments define their losses inline.

## Data flow — lazy HDF5, sim-level split

- **Canonical training format: HDF5.** Keys: `hr_states`, `lr_states`, `first_snapshot_energy`, `first_snapshot_mass`. 3D: `(sims*80, 5, 128, 128, 128)` HR vs `(..., 5, 32, 32, 32)` LR. 2D: 4 channels, 100 snapshots/sim.
- `src/dataloader/dataset_sr` keeps the `h5py.File(..., "r")` open and indexes lazily — returns `(hr, lr, energy, mass)` tuples. `__del__` closes the file. Don't load whole files into memory.
- **Two split schemes exist and they are *not* the same.** The Hydra pipeline splits *sample-level* (`random_split` over snapshots in `build_train_test_loader`). `data/split_dataset.py` splits *simulation-level* producing dedicated files. Treat this as a known limitation (documented in `experiments/experiment_summary.md`); don't silently change one to match the other.
- Default HDF5 paths (resolve relative to the dataloader file via `Path(__file__).resolve().parents[6]`):
  - Source full dataset: `/export/data/jalegria/full_states_h5/full_states.h5`
  - Train split (40k→32k): `/export/scratch/jalegria/full_states_h5/full_states.h5`
  - Val split: `/export/scratch/jalegria/full_states_h5/full_states_val.h5`
- `evaluation/benchmark.py` and `experiments/training_best_models_experiment/train_best_models.py` pass explicit `h5_path=` kwargs pointing at `/export/scratch/...`. The bare dataloader defaults point at `/export/data/...`. When in doubt, check the caller.
- RNG seeds for the sample-level split live in `data/rng_seeds_160_40.pkl` — reuse them for reproducibility.

## dataset_generation scripts (the JAX side)

- `src/dataset_generation/*` runs **jf1uids** (JAX) to synthesize 128³ HR turbulence states with Kolmogorov-spectrum forcing, then convolve down by `upsample_factor=4` to produce LR↔HR pairs and write HDF5. These scripts read the **legacy top-level** `config.yaml`, resolved relative to the script's own path (not the Hydra tree).
- Physical units are set via `astropy.units` + `jf1uids.CodeUnits` (code_length=3 pc, code_mass=1 M_sun, code_velocity=100 km/s). Simulation params (`num_cells: 128`, `turbulence_slope: -2`, `wanted_rms: 50`, `max_sims: 500`, `num_snapshots: 100`) live under `turbulent_sim:` in the legacy `config.yaml`.
- Do not import training/eval PyTorch code from here; this is the only layer that touches JAX for data synthesis.

## Checkpoints, runs, evaluation

- **Modern pipeline** (`src/utils/pipeline.py::save_training_artifacts`) writes per-run: `config.yaml` (full resolved Hydra config), `weights_<date>.pt` (`model.state_dict()`), `loss_curve.png` (train vs test). Run dir is `runtime.output.root_dir` + `run_dir_template` (e.g. `experiments/cfno_2/{timestamp}_{model_tag}`, `model/cnn/{model_tag}`).
- **`load_model_from_folder`** reconstructs a model from a saved `config.yaml` + `weights.pt`. It supports a legacy format where the config had a top-level `fno_2:` key mapped to `experiments.cfno_2.experiment_4.fno_2_exp_2`. Preserve this legacy path when editing.
- `runs/` held DSFNO grid-search results and is **no longer tracked** (removed in the publish cleanup; gitignored). `experiments/grid_search_dsfno.py` still writes there if re-run.
- `evaluation/benchmark.py` (flagship, 615 lines) computes MSE (per-channel, velocity-norm, vorticity), **spectral MSE** (1-D power spectrum via `Pk_library` + jf1uids unit conversion), a lightweight 3-D perceptual loss (seeded-42 random ConvNet encoder), PSNR, SSIM (3-D Gaussian-windowed). Evaluates CFNO at multiple shifts, EDSR @ x4, and a trilinear baseline. OOM is **caught** — the model is skipped, not crashed.
- `evaluation/experiment_2/cfno_2_eval.py` does OOM-adaptive batch sizing (start 16, halve down to 1). Reuse this pattern for new eval scripts.
- `experiments/experiment_summary.md` is the cross-model ledger — keep it updated when you add a model variant.

## Model zoo (published models)

- `model_zoo/` holds the 6 published models: per-model `config.yaml` (full training record), `manifest.json` (repo-relative paths; consumed by `evaluation/manifest.py`), `normalization_stats.npz`, aggregated `benchmark_metrics.csv`.
- **Weights are NOT in git** — they live on HuggingFace (`javinukem/turbulence_sr`) and are fetched with `scripts/download_weights.py`. `model_zoo/*/weights.pt` is gitignored.
- `load_manifest` resolves non-absolute `weights`/`normalization_stats` relative to the manifest's folder — keep zoo manifest paths relative so the repo stays portable.
- Add a new published model by adding a folder + manifest entry + zoo README row; the `model_zoo` bar-chart preset picks it up automatically.
- Notebooks under `notebooks/` are kept with **outputs stripped** — don't commit executed notebooks.

## File path restrictions (shared cluster)

Never save files outside of:
- `/export/home/jalegria/` (home dir)
- `/export/data/jalegria/` (personal data dir)
- `/export/scratch/jalegria/` (personal scratch dir, if it exists)

Do NOT write to shared `/tmp`, other users' dirs, or any path not under `jalegria`'s personal namespaces. Before writing any file (logs, dumps, temp files, checkpoints), verify the path starts with one of the above prefixes. Note that multiple existing files already have hardcoded absolute paths under `/export/{data,scratch}/jalegria/...` — mirror that convention rather than introducing new root paths.

## Experiment conventions — the unified engine

All manifest-era experiments train through **one engine**: `src/training/experiment.py`, driven by a declarative `experiments/<name>/experiment.yaml` (run grid, losses, hyperparameters, normalization, references, trilinear flag). Invoke per experiment:

```bash
python -m src.training.experiment experiments/<name>/experiment.yaml
python -m src.training.experiment experiments/<name>/experiment.yaml --run <run_name>   # single run
python -m src.training.experiment --manifest <run-group dir>                            # eval-only
```

The engine owns the training loop (AdamW + ReduceLROnPlateau, grad accumulation, noise, early stopping, optional clipping/AMP), `weights.pt`/`losses.csv`/`loss_curve.png`/`config.json`, the run-group `manifest.json`, and the benchmark stage (`benchmark_metrics.csv` via `evaluation.benchmark`, copied to the home experiment folder). Do NOT write a new per-experiment training script — add a `runs:` entry to the experiment's YAML. Only the 4 model-zoo-backed experiment folders are retained (`comparing_best_models_mse`, `mse_loss_combinations`, `edsr_norm_skip`, `ufno_mse_spectral`); the other experiment folders (incl. the legacy frozen scripts) were removed — their record lives in git history and in `experiments/experiment_summary.md` / `MODELS.md`.

Beyond training, the conventions below are **mandatory for new experiments**:

- **Bar charts via the shared `evaluation/comparing_models_bar_chart.py`.** Do NOT write a bespoke bar-chart plotting routine inside the experiment. Add a preset to `_PRESET_BUILDERS` (and `PRESET_NAMES` + a `_*_manifest()` discovery helper) and invoke it from `run.sh` with `--output <path>`. Cross-experiment comparison is one of the preset's jobs; reuse it.
- **Plots and `benchmark_metrics.csv` go in the home experiment folder** (`experiments/<name>/`), NOT in the scratch run dir. The scratch dir holds `weights.pt`, per-run `loss_curve.png`, `losses.csv`, `config.json`, `manifest.json`, `summary.json` (all tied to a particular training run-group). The aggregate artifacts that a human inspects across runs — `comparing_models_bar_chart_<preset>.png`, `final_snapshot_comparison.png`, `comparison_states.npy`, `benchmark_metrics.csv` — are copied/written to the home experiment folder so they survive scratch cleanup and are version-controlled.
- **`run.sh` orchestrates the stages**, each as its own `python …` invocation (autocvd runs inside each child, not in the wrapper). Typical stages: (1) optional `calibrate_<...>.py`, (2) `python -m src.training.experiment experiments/<name>/experiment.yaml`, (3) `evaluation/comparing_models_bar_chart.py --preset <name> --output experiments/<name>/comparing_models_bar_chart_<name>.png`, (4) `plot_final_snapshot.py`. The wrapper only sets `CUDA_VISIBLE_DEVICES` and forwards `"$@"` to the training stage.

## Behavioral guidelines

## Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes. These bias toward caution over speed — for trivial tasks, use judgment.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused; don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```