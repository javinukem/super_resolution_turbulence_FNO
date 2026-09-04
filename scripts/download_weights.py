#!/usr/bin/env python3
"""Download published-model weights from HuggingFace into ``experiments/``.

Weights are not tracked in git (see ``experiments/MODELS.md``); they are hosted
on the HuggingFace Hub (one repo, one subfolder per model) and fetched with
``huggingface_hub.hf_hub_download``. The Hub layout (``<name>/weights.pt``,
mirrored by each manifest entry's ``hf_filename``) is independent of the local
layout under ``experiments/<experiment>/<name>/``. Run once after cloning:

    python scripts/download_weights.py                 # all models
    python scripts/download_weights.py --model edsr
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "javinukem/turbulence_sr"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="download only this published-model entry")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "experiments" / "manifest.json"),
        help="path to the published-models manifest.json",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    zoo_dir = manifest_path.parent

    entries = [m for m in manifest["models"] if m.get("weights")]
    if args.model:
        entries = [m for m in entries if m["name"] == args.model]
        if not entries:
            sys.exit(
                f"model '{args.model}' not found (have: "
                f"{[m['name'] for m in manifest['models'] if m.get('weights')]})"
            )

    failures = 0
    for entry in entries:
        rel = entry["weights"]  # local layout, e.g. "<experiment>/<name>/weights.pt"
        hf_rel = entry.get("hf_filename", rel)  # Hub layout, e.g. "<name>/weights.pt"
        dest = zoo_dir / rel
        if dest.exists():
            print(f"[skip]  {entry['name']} (already at {dest})")
            continue
        print(f"[fetch] {entry['name']}  <-  {args.repo_id}:{hf_rel}")
        try:
            cached = hf_hub_download(repo_id=args.repo_id, filename=hf_rel)
        except Exception as e:  # network, auth, or not-yet-uploaded
            print(f"  FAILED: {e}")
            print("  (weights may not be uploaded yet — see experiments/MODELS.md)")
            failures += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, dest)

    if failures:
        sys.exit(f"{failures} download(s) failed")
    print("done.")


if __name__ == "__main__":
    main()
