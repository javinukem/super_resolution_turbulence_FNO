#!/usr/bin/env bash
# Orchestration for the baseline_b_500_epochs experiment.
#
# Stages:
#   1. train_baseline_b_500.py — train 1 CFNO shift=8 model (500 epochs,
#      MSE+spectral w_minor, skip=trilinear), evaluate via
#      evaluation.benchmark, write manifest.json + benchmark_metrics.csv
#   2. comparing_models_bar_chart.py --preset baseline_b_500_vs_edsr
#      — shared bar-chart tool comparing the 500-epoch Baseline B against
#      the EDSR norm-skip models (off/on) and trilinear baseline;
#      output PNG written to the home experiment folder
#   3. plot_final_snapshot.py — jf1uids sim + state-grid comparison,
#      output PNG written to the home experiment folder
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

echo "== Stage 1/3: train Baseline B (500 epochs) =="
python "${SCRIPT_DIR}/train_baseline_b_500.py" "$@"

echo "== Stage 2/3: bar-chart comparison (preset baseline_b_500_vs_edsr) =="
python "${REPO_ROOT}/evaluation/comparing_models_bar_chart.py" \
    --preset baseline_b_500_vs_edsr \
    --output "${SCRIPT_DIR}/comparing_models_bar_chart_baseline_b_500_vs_edsr.png"

echo "== Stage 3/3: final-snapshot comparison =="
python "${SCRIPT_DIR}/plot_final_snapshot.py"
