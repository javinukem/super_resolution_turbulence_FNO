#!/usr/bin/env bash
# Wrapper for the mse_spectral_weighting experiment.
#
# Stages (each stage is a separate python process — nothing is inlined into
# the training script):
#   1. calibrate_weights.py  — measure MSE/spectral ratio, write calibration.json
#   2. train_mse_spectral.py — train 3 CFNO shift=8 models (one per weight),
#                              evaluate via evaluation.benchmark, write
#                              manifest.json + benchmark_metrics.csv
#   3. comparing_models_bar_chart.py --preset mse_spectral
#                              — shared bar-chart tool (no duplicated plotting);
#                              output PNG written to the home experiment folder
#   4. plot_final_snapshot.py — jf1uids sim + state-grid comparison,
#                                output PNG written to the home experiment folder
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

echo "== Stage 1/4: calibrate spectral weights =="
python "${SCRIPT_DIR}/calibrate_weights.py"

echo "== Stage 2/4: train (MSE + spectral, 3 weights) =="
python "${SCRIPT_DIR}/train_mse_spectral.py" "$@"

echo "== Stage 3/4: bar-chart comparison (preset mse_spectral) =="
python "${REPO_ROOT}/evaluation/comparing_models_bar_chart.py" \
    --preset mse_spectral \
    --output "${SCRIPT_DIR}/comparing_models_bar_chart_mse_spectral.png"

echo "== Stage 4/4: final-snapshot comparison =="
python "${SCRIPT_DIR}/plot_final_snapshot.py"