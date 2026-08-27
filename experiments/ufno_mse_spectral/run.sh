#!/usr/bin/env bash
# Orchestration for the ufno_mse_spectral experiment.
#
# Trains the best UFNO_2 model from ufno_l1_spectral_unet (op=2, drop=0.1)
# with MSE + spectral (w_minor) loss under the clip_p30 optimization regime
# (500 epochs, grad clip 0.8, sched patience 30) borrowed from
# experiments/comparing_best_models_mse, then produces the final-snapshot
# state-grid (full + central-quarter zoom) and the energy-spectrum
# comparison plots.
#
# Stages (each stage is a separate python process — nothing is inlined into
# the training script):
#   1. train_ufno_mse_spectral.py — train 1 UFNO_2 model (500 epochs,
#      MSE+spectral w_minor, clip_p30 regime, skip=trilinear), evaluate via
#      evaluation.benchmark, write manifest.json + benchmark_metrics.csv
#   2. plot_final_snapshot.py — jf1uids sim + state-grid comparison (full
#      HR/UFNO/Trilinear/LR grid + central-quarter zoom HR/LR/UFNO);
#      output PNGs written to the home experiment folder
#   3. plot_spectra.py — 1-D total-energy P(k) spectrum of the UFNO SR×4
#      output vs HR/LR/Trilinear + a k^{-2} reference; output PNG written
#      to the home experiment folder
#
# GPU selection is handled INSIDE each python script via autocvd
# (per AGENTS.md) — this wrapper only optionally restricts the visible set
# before autocvd runs. Set CUDA_VISIBLE_DEVICES in the environment to
# override (default 0). Pass extra args through to the training stage.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "== Stage 1/3: train UFNO (best config, MSE+spectral, 500e clip_p30) =="
python "${SCRIPT_DIR}/train_ufno_mse_spectral.py" "$@"

echo "== Stage 2/3: final-snapshot comparison (full + zoom) =="
python "${SCRIPT_DIR}/plot_final_snapshot.py"

echo "== Stage 3/3: energy-spectrum comparison =="
python "${SCRIPT_DIR}/plot_spectra.py"