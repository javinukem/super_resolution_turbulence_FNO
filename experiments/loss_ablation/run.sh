#!/usr/bin/env bash
# Wrapper for the loss_ablation experiment.
# Training runs via the unified engine (src/training/experiment.py) with the
# experiment config in this folder; bar charts / final-snapshot plots are
# separate scripts (per AGENTS.md conventions).
# GPU selection is handled INSIDE the engine via autocvd (per AGENTS.md).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
exec bash -c 'cd "${REPO_ROOT}" && python -m src.training.experiment "${SCRIPT_DIR}/experiment.yaml" "$@"' _ "$@"
