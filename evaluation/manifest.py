"""
Shared manifest loader & model builder for experiment benchmarks/plots.

A *manifest* is a JSON file (one per training run-group, written into the
scratch output dir next to the per-run ``weights.pt``) that aggregates
everything the downstream evaluation/plotting scripts need:

    {
      "experiment": "training_best_models",
      "output_dir": "/export/scratch/.../best_models_2026-03-10_19-27-15",
      "benchmark_csv": "experiments/.../benchmark_results.csv",   # repo-relative
      "benchmark_row_key": "model",                                # CSV key column
      "normalization_stats": null | "/abs/path/normalization_stats.npz",
      "upsample_factor": 4,
      "models": [
        {
          "name": "cfno_shift8", "label": "CFNO (shift=8)",
          "model_type": "cfno" | "edsr" | "trilinear",
          "model_params": {...},                  # kwargs for the model ctor
          "weights": "/abs/path/weights.pt" | null,
          "apply_positivity_relu": true,          # ctor kwarg for cfno/edsr
          "use_norm": false,                      # norm wrapping on/off
          "loss": null | "mse_l1" | ...,
          "supports_variable_scale": true,        # cfno/trilinear vs edsr
          "eval_scales": [4, 2]
        },
        ...
      ]
    }

Downstream scripts (``evaluation/benchmark.py``, the two
``plot_final_snapshot_comparison.py`` files, ``comparing_models_bar_chart.py``,
and ``experiments/trying_new_losses_and_norm/benchmark.py``) all consume a
manifest via ``load_manifest`` and build models via ``build_model``, so the
trained-models knowledge lives in exactly one place per experiment.

Auto-discovery: each script defaults to the newest ``manifest.json`` under a
scratch glob (e.g. ``best_models_*/manifest.json``); pass an explicit
``--manifest PATH`` to override.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


# =====================================================================
# Discovery & loading
# =====================================================================


def discover_latest(scratch_base: Path, glob: str) -> Path:
    """Return the **folder** containing the newest manifest under ``scratch_base``.

    ``glob`` is matched against subdirectories of ``scratch_base`` (e.g.
    ``"best_models_*"``); each match is expected to contain a ``manifest.json``.

    "Newest" is measured by the most recent ``weights.pt`` modification time
    inside the folder — this reflects actual training completion and is robust
    to later ``manifest.json`` writes (e.g. a bootstrap manifest written into an
    old experiment dir won't make it look recent). Folders without any
    ``weights.pt`` fall back to the ``manifest.json`` mtime.

    Raises ``FileNotFoundError`` if no folder with a manifest is found.
    """
    candidates: list[tuple[float, Path]] = []
    for sub in scratch_base.glob(glob):
        if not sub.is_dir():
            continue
        manifest = sub / "manifest.json"
        if not manifest.exists():
            continue
        weight_times = [p.stat().st_mtime for p in sub.rglob("weights.pt")]
        sort_key = max(weight_times) if weight_times else manifest.stat().st_mtime
        candidates.append((sort_key, sub))
    if not candidates:
        raise FileNotFoundError(
            f"No manifest.json found under {scratch_base}/{glob}. "
            "Run the corresponding training script first (it writes manifest.json)."
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    chosen = candidates[0][1]
    print(f"  Auto-discovered experiment dir: {chosen}")
    return chosen


def _resolve_path(p, anchor: Path) -> Path:
    """Resolve ``p``: absolute as-is, else relative to ``anchor``."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (anchor / p)


def load_manifest(path) -> dict:
    """Load a manifest JSON and resolve its internal paths.

    ``path`` may be either the experiment folder (containing
    ``manifest.json``) or the ``manifest.json`` file itself — so callers can
    pass a ``--manifest`` folder directly.

    ``benchmark_csv`` is resolved relative to the repo ``ROOT``; ``weights``
    and ``normalization_stats`` are taken as-is when absolute, and resolved
    relative to the manifest's own folder otherwise (portable manifests).
    """
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found at {path}")
    with open(path) as f:
        manifest = json.load(f)
    manifest["_manifest_path"] = path
    manifest["benchmark_csv"] = _resolve_path(manifest.get("benchmark_csv"), ROOT)
    # Non-absolute paths resolve relative to the manifest's own folder so
    # manifests stay portable across machines (absolute paths pass through).
    manifest["normalization_stats"] = _resolve_path(
        manifest.get("normalization_stats"), path.parent
    )
    for entry in manifest.get("models", []):
        entry["weights"] = _resolve_path(entry.get("weights"), path.parent)
    return manifest


def get_model_entry(manifest: dict, name: str) -> dict:
    for entry in manifest["models"]:
        if entry["name"] == name:
            return entry
    raise KeyError(
        f"Model '{name}' not in manifest (have: "
        f"{[m['name'] for m in manifest['models']]})"
    )


# =====================================================================
# Normalization statistics
# =====================================================================


def load_norm_stats(manifest: dict, device: torch.device) -> dict | None:
    """Load cached per-channel mean/std for norm-on experiments.

    Returns ``None`` when ``manifest["normalization_stats"]`` is null.
    """
    stats_path = manifest.get("normalization_stats")
    if stats_path is None:
        return None
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    data = np.load(stats_path)
    return {
        "mean_lr": torch.tensor(data["mean_lr"], device=device, dtype=torch.float32),
        "std_lr": torch.tensor(data["std_lr"], device=device, dtype=torch.float32),
        "mean_hr": torch.tensor(data["mean_hr"], device=device, dtype=torch.float32),
        "std_hr": torch.tensor(data["std_hr"], device=device, dtype=torch.float32),
    }


# =====================================================================
# Model construction
# =====================================================================


class _TrilinearBaseline(nn.Module):
    """Trilinear (3-D bicubic analog) upsampling baseline."""

    def forward(self, x: torch.Tensor, upsample_factor: int = 4) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=upsample_factor,
            mode="trilinear",
            align_corners=False,
        )


def _instantiate(entry: dict) -> nn.Module:
    mtype = entry["model_type"]
    params = dict(entry.get("model_params") or {})
    apply_relu = entry.get("apply_positivity_relu", True)
    if mtype == "cfno":
        from src.model.models_fno_2 import FNO_2

        return FNO_2(**params, apply_positivity_relu=apply_relu)
    if mtype == "edsr":
        from src.model.models_edsr import EDSR

        return EDSR(**params, apply_positivity_relu=apply_relu)
    if mtype == "ufno":
        from src.model.models_ufno_2 import UFNO_2

        return UFNO_2(**params, apply_positivity_relu=apply_relu)
    if mtype == "trilinear":
        return _TrilinearBaseline()
    raise ValueError(f"Unknown model_type: {mtype}")


def build_model(
    entry: dict,
    device: torch.device,
    norm_stats: dict | None = None,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Build a model from a manifest entry and return a uniform ``run`` callable.

    The returned callable has signature ``run(lr, upsample_factor) -> sr`` and
    handles: optional LR normalization, the cfno-vs-edsr forward branch
    (cfno/trilinear take ``upsample_factor``, edsr does not), and optional SR
    denormalization. The underlying ``nn.Module`` is accessible via
    ``run.model`` (useful for ``del`` + ``empty_cache`` in callers).
    """
    model = _instantiate(entry).to(device)
    weights = entry.get("weights")
    if weights is not None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True)
        )
    model.eval()

    use_norm = entry.get("use_norm", False)
    supports_variable_scale = entry.get("supports_variable_scale", False)

    def run(lr: torch.Tensor, upsample_factor: int = 4) -> torch.Tensor:
        with torch.no_grad():
            if use_norm and norm_stats is not None:
                ml = norm_stats["mean_lr"][None, :, None, None, None]
                sl = norm_stats["std_lr"][None, :, None, None, None]
                mh = norm_stats["mean_hr"][None, :, None, None, None]
                sh = norm_stats["std_hr"][None, :, None, None, None]
                lr_n = (lr - ml) / sl
                if supports_variable_scale:
                    sr_n = model(lr_n, upsample_factor=upsample_factor)
                else:
                    sr_n = model(lr_n)
                return sr_n * sh + mh
            if supports_variable_scale:
                return model(lr, upsample_factor=upsample_factor)
            return model(lr)

    run.model = model
    return run
