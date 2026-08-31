#!/usr/bin/env bash
# Wrapper for the mse_loss_combinations experiment.
#
# Stages (each stage is a separate python process — nothing is inlined into
# the training script):
#   1. calibrate_weights.py            — measure MSE/L1/spectral on a trilinear
#                                        upsampler, derive "light" (w_minor)
#                                        weights, write calibration.json
#   2. train_mse_loss_combinations.py  — train 3 CFNO runs (shift=8, skip=
#                                        trilinear, lr=1e-3, grad clip 0.8,
#                                        sched_patience=30, 500 epochs) with
#                                        MSE-only loss (reuse from the
#                                        comparing_best_models_mse experiment)
#                                        is NOT retrained here — only the 3
#                                        new combos are trained in this script:
#                                          - MSE + light spectral
#                                          - MSE + light L1
#                                          - MSE + light spectral + light L1
#                                        Then evaluate via evaluation.benchmark,
#                                        and write manifest.json +
#                                        benchmark_metrics.csv
#   3. comparing_models_bar_chart.py --preset mse_loss_combinations
#                                      — shared bar-chart tool (no duplicated
#                                        plotting). Compares the 3 new combos
#                                        + the MSE-only run (cross-referenced
#                                        from the comparing_best_models_mse
#                                        manifest) + the trilinear baseline.
#   4. plot_final_snapshot.py         — jf1uids sim + state-grid comparison;
#                                        output PNG written to the home
#                                        experiment folder
#
# GPU selection is handled INSIDE each python script via autocvd
# (per AGENTS.md) — this wrapper only optionally restricts the visible set
# before autocvd runs. Set CUDA_VISIBLE_DEVICES in the environment to
# override (default 0). Pass extra args through to the training stage.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "== Stage 1/4: calibrate MSE/L1/spectral weights =="
python "${SCRIPT_DIR}/calibrate_weights.py"

echo "== Stage 2/4: train 3 CFNO combos (shift=8, MSE-based, 500e clip_p30) =="
cd "${REPO_ROOT}" && python -m src.training.experiment "${SCRIPT_DIR}/experiment.yaml" "$@"

echo "== Stage 3/4: bar-chart comparison (preset mse_loss_combinations) =="
python "${REPO_ROOT}/evaluation/comparing_models_bar_chart.py" \
    --preset mse_loss_combinations \
    --output "${SCRIPT_DIR}/comparing_models_bar_chart_mse_loss_combinations.png"

echo "== Stage 4/4: final-snapshot comparison =="
python "${SCRIPT_DIR}/plot_final_snapshot.py"