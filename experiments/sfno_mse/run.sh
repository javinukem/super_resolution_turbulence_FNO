#!/usr/bin/env bash
# Wrapper for the comparing_best_models_mse experiment.
#
# Stages (each stage is a separate python process — nothing is inlined into
# the training script):
#   1. train_comparing_best_models_mse.py — train the canonical stabilized
#      SFNO Baseline B config (shift=8, skip=trilinear, lr=1e-3, grad clip 0.8,
#      sched_patience=30, 500 epochs) with **MSE-only** loss; evaluate via
#      evaluation.benchmark; write manifest.json + benchmark_metrics.csv
#   2. comparing_models_bar_chart.py --preset comparing_best_models_mse
#      — shared bar-chart tool (no duplicated plotting); compares the MSE-only
#      SFNO trained here against the best MSE-only EDSR (edsr) and
#      the trilinear interpolation baseline. Output PNG written to the home
#      experiment folder.
#
# GPU selection is handled INSIDE the training python script via autocvd
# (per AGENTS.md) — this wrapper only optionally restricts the visible set
# before autocvd runs. Set CUDA_VISIBLE_DEVICES in the environment to
# override (default 0). Pass extra args through to the training stage.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "== Stage 1/2: train SFNO (shift=8, MSE only, 500e clip_p30) =="
cd "${REPO_ROOT}" && python -m src.training.experiment "${SCRIPT_DIR}/experiment.yaml" "$@"

echo "== Stage 2/2: bar-chart comparison (preset comparing_best_models_mse) =="
python "${REPO_ROOT}/evaluation/comparing_models_bar_chart.py" \
    --preset comparing_best_models_mse \
    --output "${SCRIPT_DIR}/comparing_models_bar_chart_comparing_best_models_mse.png"