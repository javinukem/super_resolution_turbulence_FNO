#!/usr/bin/env bash
# Orchestration for the ufno_mse_spectral_500e experiment.
#
# Trains the best UFNO_2 model from ufno_l1_spectral_unet (op=2, drop=0.1)
# with the Baseline-B loss (MSESpectralLoss, w_minor) and training regime
# (500 epochs, grad clip 1.0, sched patience 10), then compares it against
# Baseline B and Baseline D in a shared bar chart.
#
# Stages:
#   1. train_ufno_mse_spectral_500.py — train 1 UFNO_2 model (500 epochs,
#      MSE+spectral w_minor, skip=trilinear), evaluate via
#      evaluation.benchmark, write manifest.json + benchmark_metrics.csv
#   2. comparing_models_bar_chart.py --preset ufno_mse_spectral_500
#      — shared bar-chart tool comparing the UFNO run against Baseline B
#      (clip_p10) and Baseline D (edsr_norm_skip_on) + trilinear;
#      output PNG written to the home experiment folder
#   2b. comparing_models_bar_chart.py --preset ufno_mse_spectral_500
#       --upsample-factor 2 — same comparison at the x2 evaluation scale
#       (Baseline D lacks an x2 row and is skipped; the UFNO-vs-Baseline B
#       comparison is the focus).
#   3. plot_final_snapshot.py — jf1uids sim + state-grid comparison (full
#      HR/UFNO/Trilinear/LR grid + central-quarter zoom HR/LR/UFNO);
#      output PNGs written to the home experiment folder
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

echo "== Stage 1/3: train UFNO (best config, MSE+spectral, 500e clip_p10) =="
python "${SCRIPT_DIR}/train_ufno_mse_spectral_500.py" "$@"

echo "== Stage 2/4: bar-chart comparison @ x4 (preset ufno_mse_spectral_500) =="
python "${REPO_ROOT}/evaluation/comparing_models_bar_chart.py" \
    --preset ufno_mse_spectral_500 \
    --output "${SCRIPT_DIR}/comparing_models_bar_chart_ufno_mse_spectral_500.png"

echo "== Stage 3/4: bar-chart comparison @ x2 (UFNO vs Baseline B) =="
python "${REPO_ROOT}/evaluation/comparing_models_bar_chart.py" \
    --preset ufno_mse_spectral_500 \
    --upsample-factor 2 \
    --output "${SCRIPT_DIR}/comparing_models_bar_chart_ufno_mse_spectral_500_x2.png"

echo "== Stage 4/4: final-snapshot comparison (full + zoom) =="
python "${SCRIPT_DIR}/plot_final_snapshot.py"
