#!/usr/bin/env python3
"""Upload published-model weights to HuggingFace (one subfolder per model).

Counterpart to ``scripts/download_weights.py``: pushes each model's
``weights.pt`` from the local training-output (scratch) dirs to the Hub repo
(``<name>/weights.pt``, mirrored by each manifest entry's ``hf_filename``), so
that ``download_weights.py`` can fetch them back into
``experiments/<experiment>/<name>/``.

Prerequisites:

    pip install huggingface_hub
    huggingface-cli login                      # token with write access
    huggingface-cli repo create turbulence_sr  # one-time repo creation

Usage:

    python scripts/upload_weights.py                 # all models
    python scripts/upload_weights.py --model edsr_norm_skip_on
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "javinukem/turbulence_sr"

# Published-model entry -> authoritative local weights file (training run
# output; scratch run groups still carry the historical run-folder names).
SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
SOURCES = {
    "sfno":
        "comparing_best_models_mse_07-23_19-47/cfno_shift8_mse_only_500e_clip_p10",
    "sfno_spectral":
        "mse_loss_combinations_07-24_02-09/cfno_shift8_mse_light_spectral_500e_clip_p30",
    "sfno_l1":
        "mse_loss_combinations_07-24_02-09/cfno_shift8_mse_light_l1_500e_clip_p30",
    "sfno_spectral_l1":
        "mse_loss_combinations_07-24_02-09/cfno_shift8_mse_light_spectral_l1_500e_clip_p30",
    "edsr":
        "edsr_norm_skip_07-03_19-48/edsr_norm_skip_on",
    "usfno":
        "ufno_mse_spectral_07-30_14-15/ufno_shift8_mse_spectral_w_minor_500e_clip_p50",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="upload only this published-model entry")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo id (default: {DEFAULT_REPO_ID})",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import upload_file
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

    manifest = json.loads((ROOT / "experiments" / "manifest.json").read_text())
    entries = {m["name"]: m for m in manifest["models"] if m.get("weights")}
    names = list(entries)
    if args.model:
        if args.model not in names:
            sys.exit(f"model '{args.model}' not in published models (have: {names})")
        names = [args.model]

    for name in names:
        src = SCRATCH_BASE / SOURCES[name] / "weights.pt"
        if not src.exists():
            print(f"[miss] {name}: {src} not found — skipping")
            continue
        # Hub layout is independent of local names: upload to the entry's
        # hf_filename (the existing Hub path), not the local model name.
        hf_rel = entries[name].get("hf_filename", f"{name}/weights.pt")
        print(f"[up]   {name}: {src} -> {args.repo_id}:{hf_rel}")
        upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=str(src),
            path_in_repo=hf_rel,
        )
    print("done.")


if __name__ == "__main__":
    main()
