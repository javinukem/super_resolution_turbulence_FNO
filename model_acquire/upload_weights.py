#!/usr/bin/env python3
"""Upload published-model weights to HuggingFace (one subfolder per model).

Counterpart to ``scripts/download_weights.py``: pushes each published model's
``weights.pt`` from the latest training run-group (scratch) dir to the Hub repo
at the path recorded in ``experiments/<experiment>/<name>/config.yaml`` under
``weights.filename``, so that ``download_weights.py`` can fetch them back.

Discovery (no central manifest — it was removed in the repo cleanup):

- ``experiments/<experiment>/experiment.yaml`` gives the scratch ``folder_prefix``
  and the run names; the newest run-group folder under the scratch root (by
  ``weights.pt`` mtime) supplies the local weights paths via its ``manifest.json``.
- Only runs with a per-model ``experiments/<experiment>/<name>/config.yaml``
  (the published-model record) are uploaded.

Prerequisites:

    pip install huggingface_hub
    huggingface-cli login                      # token with write access
    huggingface-cli repo create turbulence_sr  # one-time repo creation

Usage:

    python scripts/upload_weights.py                 # all models
    python scripts/upload_weights.py --model usfno
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "javinukem/turbulence_sr"
EXPERIMENTS_DIR = ROOT / "experiments"


def _latest_run_group(scratch_root: Path, prefix: str) -> Path:
    """Newest run-group folder (by weights.pt mtime) holding a manifest.json."""
    candidates: list[tuple[float, Path]] = []
    for sub in Path(scratch_root).glob(prefix + "*"):
        manifest = sub / "manifest.json"
        if not (sub.is_dir() and manifest.exists()):
            continue
        w_times = [p.stat().st_mtime for p in sub.rglob("weights.pt")]
        candidates.append(
            (max(w_times) if w_times else manifest.stat().st_mtime, sub)
        )
    if not candidates:
        raise FileNotFoundError(
            f"No manifest.json found under {scratch_root}/{prefix}*. "
            "Run the corresponding training script first (it writes manifest.json)."
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _collect_models() -> dict[str, tuple[Path, str]]:
    """name -> (local weights path, Hub filename) for every published model."""
    models: dict[str, tuple[Path, str]] = {}
    for exp_yaml in sorted(EXPERIMENTS_DIR.glob("*/experiment.yaml")):
        exp = yaml.safe_load(exp_yaml.read_text())
        out = exp.get("output") or {}
        prefix = out.get("folder_prefix")
        scratch_root = out.get("scratch_root")
        if not prefix or not scratch_root:
            continue
        run_group = _latest_run_group(Path(scratch_root), prefix)
        manifest = json.loads((run_group / "manifest.json").read_text())
        entries = {m["name"]: m for m in manifest.get("models", [])}
        for run in exp.get("runs") or []:
            name = run["name"]
            cfg_path = exp_yaml.parent / name / "config.yaml"
            if not cfg_path.exists():
                continue
            cfg = yaml.safe_load(cfg_path.read_text())
            hf_rel = (cfg.get("weights") or {}).get("filename")
            if not hf_rel:
                continue
            # New-style run-groups name entries after the run; historical ones
            # kept the run-folder name. Match on run name, then label, then
            # the sole entry of a single-run group.
            entry = entries.get(name)
            if entry is None:
                entry = next(
                    (e for e in entries.values() if e.get("label") == run.get("label")),
                    None,
                )
            if entry is None and len(entries) == 1:
                entry = next(iter(entries.values()))
            if entry is None:
                continue
            weights = Path(entry["weights"])
            if not weights.is_absolute():
                weights = run_group / weights
            models[name] = (weights, hf_rel)
    return models


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

    models = _collect_models()
    if not models:
        sys.exit(f"no published models found under {EXPERIMENTS_DIR}")
    if args.model:
        if args.model not in models:
            sys.exit(
                f"model '{args.model}' not in published models (have: {list(models)})"
            )
        models = {args.model: models[args.model]}

    for name, (src, hf_rel) in models.items():
        if not src.exists():
            print(f"[miss] {name}: {src} not found — skipping")
            continue
        print(f"[up]   {name}: {src} -> {args.repo_id}:{hf_rel}")
        upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=str(src),
            path_in_repo=hf_rel,
        )
    print("done.")


if __name__ == "__main__":
    main()
