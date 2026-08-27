#!/usr/bin/env bash
# Wrapper for the l1_spectral_skip_connection experiment.
# GPU selection is handled INSIDE the python script via autocvd
# (per AGENTS.md) — this wrapper only optionally restricts the visible set
# before autocvd runs.  Set CUDA_VISIBLE_DEVICES in the environment to
# override (default 0).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
exec python "${SCRIPT_DIR}/train_l1_spectral_skip_connection.py" "$@"