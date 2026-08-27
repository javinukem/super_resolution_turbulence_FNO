#!/usr/bin/env bash
# Wrapper for the edsr_norm_skip experiment.
#
# Stages (each stage is a separate python process — nothing is inlined into
# the training script):
#   1. train_edsr_norm_skip.py   — train 2 EDSR models (norm-on, skip off/on),
#                                  evaluate via evaluation.benchmark, write
#                                  manifest.json + benchmark_metrics.csv
#   2. comparing_models_bar_chart.py --preset edsr_norm_skip
#                                  — shared bar-chart tool (no duplicated
#                                  plotting); output PNG written to the home
#                                  experiment folder
#   3. plot_final_snapshot.py    — jf1uids sim + state-grid comparison;
#                                  output PNG written to the home experiment
#                                  folder
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

echo "== Stage 1/3: train EDSR (norm-on, skip off/on) =="
python "${SCRIPT_DIR}/train_edsr_norm_skip.py" "$@"

echo "== Stage 2/3: bar-chart comparison (preset edsr_norm_skip) =="
python "${REPO_ROOT}/evaluation/comparing_models_bar_chart.py" \
    --preset edsr_norm_skip \
    --output "${SCRIPT_DIR}/comparing_models_bar_chart_edsr_norm_skip.png"

echo "== Stage 3/3: final-snapshot comparison =="
python "${SCRIPT_DIR}/plot_final_snapshot.py"