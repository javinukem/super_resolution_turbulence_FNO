"""
Bar-chart comparison of model metrics across experiments.

Reads benchmark CSV files referenced by experiment manifests and produces a
grouped bar chart with one subplot per metric (MSE, per-channel losses,
spectral, perceptual, PSNR, SSIM).

The script is configuration-driven: choose a ``--preset`` (which references one
or more manifests and picks entries by name) or pass ``--manifest`` (optionally
with ``--models name1,name2``) for a custom comparison.  Each series loads one
row from a manifest's ``benchmark_csv`` by filtering on the manifest's
``benchmark_row_key`` column, so the same code works for:

  - ``training_best_models`` manifest (row key ``model``, value = entry label)
  - ``trying_new_losses_and_norm`` manifest (row key ``run``, value = entry name)

Usage
-----
    # default preset: best CFNO shift=8 vs EDSR vs trilinear (baseline experiment)
    python evaluation/comparing_models_bar_chart.py

    # all 12 loss-norm runs + trilinear baseline
    python evaluation/comparing_models_bar_chart.py --preset loss_norm

    # CFNO shift=8 MSE-only vs best EDSR vs trilinear
    python evaluation/comparing_models_bar_chart.py --preset loss_ablation_mse_vs_edsr

    # custom: pick models from a manifest
    python evaluation/comparing_models_bar_chart.py --manifest /path/to/manifest.json \
        --models cfno_shift8,edsr

    # custom ad-hoc CSV (no manifest): legacy single-series mode
    python evaluation/comparing_models_bar_chart.py --csv path/to/csv \
        --filter model=CFNO\\(shift=8\\),upsample_factor=4 --label "CFNO s8"
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.manifest import discover_latest, get_model_entry, load_manifest

SCRATCH_BASE = Path("/export/scratch/jalegria/experiments")
OUTPUT_DIR = ROOT / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Metrics to plot (name, direction: "↓" = lower better, "↑" = higher better) ──

METRICS = [
    ("MSE", "↓"),
    ("Loss_density", "↓"),
    ("Loss_pressure", "↓"),
    ("Loss_vx", "↓"),
    ("Loss_vy", "↓"),
    ("Loss_vz", "↓"),
    ("Loss_v_norm", "↓"),
    ("Loss_vorticity", "↓"),
    ("Spectral_MSE", "↓"),
    ("Perceptual", "↓"),
    ("PSNR", "↑"),
    ("SSIM", "↑"),
]

PRESET_NAMES = [
    "baseline",
    "loss_norm",
    "loss_norm_cfno",
    "best_performing",
    "l1_spectral_best_vs_best_performing",
    "l1_spectral_skip",
    "ufno",
    "loss_ablation_mse_vs_edsr",
    "mse_spectral",
    "edsr_norm_skip",
    "edsr_vs_cfno_baseline_a",
    "spectral_weighting_l1_vs_mse",
    "baselines_vs_edsr_norm_skip",
    "baseline_b_500_vs_edsr",
    "ufno_mse_spectral_500",
    "comparing_best_models_mse",
    "mse_loss_combinations",
    "model_zoo",
]


# =====================================================================
# Manifest discovery
# =====================================================================


def _best_models_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "best_models_*"))


def _loss_norm_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "loss_norm_experiment_*"))


def _l1_spectral_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "l1_spectral_weighting_*"))


def _l1_spectral_skip_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "l1_spectral_skip_connection_*"))


def _ufno_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "ufno_l1_spectral_unet_*"))


def _loss_ablation_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "loss_ablation_*"))


def _mse_spectral_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "mse_spectral_weighting_*"))


def _edsr_norm_skip_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "edsr_norm_skip_*"))


def _baseline_b_500_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "baseline_b_500_epochs_*"))


def _ufno_mse_spectral_500_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "ufno_mse_spectral_500_*"))


def _comparing_best_models_mse_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "comparing_best_models_mse_*"))


def _mse_loss_combinations_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "mse_loss_combinations_*"))


def _model_zoo_manifest() -> dict:
    """The repo-local published-model zoo (no scratch discovery needed)."""
    return load_manifest(ROOT / "model_zoo")


# =====================================================================
# Preset builders — each returns a list of series specs
# =====================================================================


def _build_baseline(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    return [
        {"manifest": bm, "model": "cfno_shift8", "label": "CFNO shift=8"},
        {"manifest": bm, "model": "edsr", "label": "EDSR"},
        {"manifest": bm, "model": "trilinear", "label": "Trilinear"},
    ]


def _build_loss_norm(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    series = []
    for e in ln["models"]:
        series.append({"manifest": ln, "model": e["name"], "label": e["label"]})
    return series


def _build_loss_norm_cfno(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    series = []
    for e in ln["models"]:
        if e["model_type"] == "cfno" or e["model_type"] == "trilinear":
            series.append({"manifest": ln, "model": e["name"], "label": e["label"]})
    return series


def _build_best_performing(bm: dict, ln: dict, l1s: dict = None) -> list[dict]:
    series = []
    for e in ln["models"]:
        if e["model_type"] == "cfno" and e.get("use_norm"):
            series.append({"manifest": ln, "model": e["name"], "label": e["label"]})
    series.append(
        {"manifest": bm, "model": "cfno_shift8", "label": "cfno_shift8_mse_nonorm"}
    )
    return series


def _build_l1_spectral_best_vs_best_performing(
    bm: dict, ln: dict, l1s: dict
) -> list[dict]:
    """Top-2 l1_spectral runs (by MSE) + all best_performing models."""
    series = []
    csv_path = l1s["benchmark_csv"]
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        row_key = l1s["benchmark_row_key"]
        value_field = l1s["benchmark_row_value_field"]
        scored = []
        for e in l1s["models"]:
            if e["model_type"] == "trilinear":
                continue
            matches = df[df[row_key] == e[value_field]]
            if not matches.empty:
                scored.append((float(matches.iloc[0]["MSE"]), e))
        scored.sort(key=lambda t: t[0])
        for _, e in scored[:2]:
            series.append({"manifest": l1s, "model": e["name"], "label": e["label"]})
    else:
        print(f"  Warning: {csv_path} not found — skipping l1_spectral entries")
    series.extend(_build_best_performing(bm, ln))
    return series


def _build_l1_spectral_skip(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """All cfno entries from the l1_spectral_skip_connection manifest (the two
    skip=True runs and the skip=False reference) plus the trilinear baseline."""
    series = []
    try:
        skip = _l1_spectral_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: l1_spectral_skip manifest not found ({e}) — skipping")
        return series
    for e in skip["models"]:
        series.append({"manifest": skip, "model": e["name"], "label": e["label"]})
    return series


def _build_ufno(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """All entries from the ufno_l1_spectral_unet manifest (the four UFNO
    runs, the cfno skip=False reference, and the trilinear baseline)."""
    series = []
    try:
        ufno = _ufno_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: ufno manifest not found ({e}) — skipping")
        return series
    for e in ufno["models"]:
        series.append({"manifest": ufno, "model": e["name"], "label": e["label"]})
    return series


def _build_loss_ablation_mse_vs_edsr(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """CFNO shift=8 MSE-only (from loss_ablation) vs best EDSR vs trilinear."""
    series = []
    try:
        la = _loss_ablation_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: loss_ablation manifest not found ({e}) — skipping CFNO")
    else:
        series.append(
            {
                "manifest": la,
                "model": "cfno_shift8_mse_only",
                "label": "FNO",
            },
        )
    series.append({"manifest": bm, "model": "edsr", "label": "EDSR"})
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_mse_spectral(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """All 3 MSE+spectral CFNO shift=8 runs + trilinear baseline (for context)."""
    series = []
    try:
        mse_spec = _mse_spectral_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: mse_spectral manifest not found ({e}) — skipping")
        return series
    for e in mse_spec["models"]:
        series.append({"manifest": mse_spec, "model": e["name"], "label": e["label"]})
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_edsr_norm_skip(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """Both EDSR norm-on runs (skip off/on) + trilinear baseline (for context)."""
    series = []
    try:
        ens = _edsr_norm_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: edsr_norm_skip manifest not found ({e}) — skipping")
        return series
    for e in ens["models"]:
        series.append({"manifest": ens, "model": e["name"], "label": e["label"]})
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_edsr_vs_cfno_baseline_a(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """EDSR norm-on (skip off/on) vs CFNO Baseline A (cfno_shift8_skip_trilinear
    from l1_spectral_skip_connection) + trilinear baseline for context."""
    series = []
    try:
        ens = _edsr_norm_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: edsr_norm_skip manifest not found ({e}) — skipping EDSR")
    else:
        for e in ens["models"]:
            series.append({"manifest": ens, "model": e["name"], "label": e["label"]})
    try:
        skip = _l1_spectral_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: l1_spectral_skip manifest not found ({e}) — skipping CFNO")
    else:
        for e in skip["models"]:
            if e["name"] == "cfno_shift8_skip_trilinear":
                series.append(
                    {
                        "manifest": skip,
                        "model": e["name"],
                        "label": "CFNO Baseline A (skip=trilinear)",
                    }
                )
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_spectral_weighting_l1_vs_mse(
    bm: dict, ln: dict, l1s: dict
) -> list[dict]:
    """L1+spectral vs MSE+spectral CFNO shift=8 weighting sweep + the base
    L1-only and MSE-only references from loss_ablation. All three
    ``l1_spectral_weighting`` runs (w_minor/equal/major), all three
    ``mse_spectral_weighting`` runs (w_minor/equal/major), plus
    ``cfno_shift8_l1_only`` and ``cfno_shift8_mse_only`` from loss_ablation as
    the no-spectral baselines."""
    series = []
    for e in l1s["models"]:
        if e.get("model_type") == "cfno":
            series.append({"manifest": l1s, "model": e["name"], "label": e["label"]})
    try:
        mse_spec = _mse_spectral_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: mse_spectral manifest not found ({e}) — skipping")
    else:
        for entry in mse_spec["models"]:
            series.append(
                {"manifest": mse_spec, "model": entry["name"], "label": entry["label"]}
            )
    try:
        la = _loss_ablation_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: loss_ablation manifest not found ({e}) — skipping")
    else:
        for name, label in [
            ("cfno_shift8_l1_only", "L1 only (no spectral)"),
            ("cfno_shift8_mse_only", "MSE only (no spectral)"),
        ]:
            try:
                entry = get_model_entry(la, name)
            except KeyError:
                print(f"  Warning: '{name}' not in loss_ablation manifest — skipping")
                continue
            series.append(
                {"manifest": la, "model": name, "label": label or entry["label"]}
            )
    return series


def _build_baselines_vs_edsr_norm_skip(
    bm: dict, ln: dict, l1s: dict
) -> list[dict]:
    """All three CFNO baselines (A, B, C) vs both EDSR norm-skip runs
    + trilinear baseline for context."""
    series = []
    # CFNO Baseline C — L1+spectral no-skip w_minor (from l1_spectral_weighting)
    for e in l1s["models"]:
        if e.get("name") == "cfno_shift8_l1_spectral_w_minor":
            series.append(
                {"manifest": l1s, "model": e["name"], "label": "Baseline C (L1+spec, no skip)"}
            )
    # CFNO Baseline B — MSE+spectral w_minor with skip (from mse_spectral_weighting)
    try:
        mse_spec = _mse_spectral_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: mse_spectral manifest not found ({e}) — skipping")
    else:
        for e in mse_spec["models"]:
            if e.get("weight_name") == "w_minor":
                series.append(
                    {"manifest": mse_spec, "model": e["name"], "label": "Baseline B (MSE+spec, skip)"}
                )
    # CFNO Baseline A — L1+spectral skip trilinear (from l1_spectral_skip_connection)
    try:
        skip = _l1_spectral_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: l1_spectral_skip manifest not found ({e}) — skipping")
    else:
        for e in skip["models"]:
            if e.get("name") == "cfno_shift8_skip_trilinear":
                series.append(
                    {"manifest": skip, "model": e["name"], "label": "Baseline A (L1+spec, skip)"}
                )
    # EDSR norm-skip runs
    try:
        ens = _edsr_norm_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: edsr_norm_skip manifest not found ({e}) — skipping")
    else:
        for e in ens["models"]:
            series.append({"manifest": ens, "model": e["name"], "label": e["label"]})
    # Trilinear baseline
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_baseline_b_500_vs_edsr(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """Baseline B (500 epochs) vs both EDSR norm-skip runs + trilinear."""
    series = []
    try:
        b5 = _baseline_b_500_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: baseline_b_500 manifest not found ({e}) — skipping")
    else:
        for e in b5["models"]:
            series.append(
                {"manifest": b5, "model": e["name"], "label": e["label"]}
            )
    try:
        ens = _edsr_norm_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: edsr_norm_skip manifest not found ({e}) — skipping")
    else:
        for e in ens["models"]:
            series.append({"manifest": ens, "model": e["name"], "label": e["label"]})
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_ufno_mse_spectral_500(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """UFNO (best config, MSE+spectral, 500e clip_p10) vs Baseline B vs
    Baseline D + trilinear for context.

    - UFNO run: the single ``ufno_shift8_mse_spectral_w_minor_500e_clip_p10``
      entry from the ``ufno_mse_spectral_500e`` experiment.
    - Baseline B: the ``cfno_shift8_mse_spectral_w_minor_500e_clip_p10`` entry
      from ``baseline_b_500_epochs`` (the canonical stabilized Baseline B).
    - Baseline D: the ``edsr_norm_skip_on`` entry from ``edsr_norm_skip``.
    """
    series = []
    # UFNO (this experiment)
    try:
        uf = _ufno_mse_spectral_500_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: ufno_mse_spectral_500 manifest not found ({e}) — skipping")
    else:
        for e in uf["models"]:
            series.append(
                {"manifest": uf, "model": e["name"], "label": e["label"]}
            )
    # Baseline B (clip_p10 — the canonical stabilized Baseline B)
    try:
        b5 = _baseline_b_500_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: baseline_b_500 manifest not found ({e}) — skipping")
    else:
        for e in b5["models"]:
            if e.get("name") == "cfno_shift8_mse_spectral_w_minor_500e_clip_p10":
                series.append(
                    {
                        "manifest": b5,
                        "model": e["name"],
                        "label": "Baseline B (500e, clip=1.0, p=10)",
                    }
                )
    # Baseline D (EDSR norm-on, skip on)
    try:
        ens = _edsr_norm_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: edsr_norm_skip manifest not found ({e}) — skipping")
    else:
        for e in ens["models"]:
            if e.get("name") == "edsr_norm_skip_on":
                series.append(
                    {
                        "manifest": ens,
                        "model": e["name"],
                        "label": "Baseline D (EDSR norm, skip on)",
                    }
                )
    # Trilinear baseline for context
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_comparing_best_models_mse(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """MSE-only CFNO (canonical stabilized Baseline B config, 500e clip_p30)
    vs the best MSE-only EDSR (edsr_norm_skip_on, "Baseline D") + trilinear
    interpolation for context. Isolates the loss function (MSE only) across
    the two flagship architectures at their canonical stabilized settings."""
    series = []
    try:
        cbm = _comparing_best_models_mse_manifest()
    except FileNotFoundError as e:
        print(
            f"  Warning: comparing_best_models_mse manifest not found "
            f"({e}) — skipping CFNO"
        )
    else:
        for e in cbm["models"]:
            series.append({"manifest": cbm, "model": e["name"], "label": e["label"]})
    try:
        ens = _edsr_norm_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: edsr_norm_skip manifest not found ({e}) — skipping EDSR")
    else:
        for e in ens["models"]:
            if e.get("name") == "edsr_norm_skip_on":
                series.append(
                    {
                        "manifest": ens,
                        "model": e["name"],
                        "label": "Baseline D (EDSR norm, skip on)",
                    }
                )
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_mse_loss_combinations(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """MSE-based loss combinations on the canonical stabilized CFNO (shift=8,
    skip=trilinear, 500e clip_p30):

      - MSE + light spectral          (from the mse_loss_combinations manifest)
      - MSE + light L1               (from the mse_loss_combinations manifest)
      - MSE + light spectral + L1    (from the mse_loss_combinations manifest)
      - MSE only                     (reused from the comparing_best_models_mse
                                      manifest — not retrained here)

    Compares against the original MSE-only run and the trilinear baseline to
    isolate the effect of adding a light auxiliary term (spectral and/or L1)
    to the MSE objective."""
    series = []
    try:
        mlc = _mse_loss_combinations_manifest()
    except FileNotFoundError as e:
        print(
            f"  Warning: mse_loss_combinations manifest not found ({e}) "
            f"— skipping combos"
        )
    else:
        for e in mlc["models"]:
            series.append({"manifest": mlc, "model": e["name"], "label": e["label"]})
    try:
        cbm = _comparing_best_models_mse_manifest()
    except FileNotFoundError as e:
        print(
            f"  Warning: comparing_best_models_mse manifest not found ({e}) "
            f"— skipping MSE-only"
        )
    else:
        for e in cbm["models"]:
            series.append({"manifest": cbm, "model": e["name"], "label": e["label"]})
    series.append({"manifest": bm, "model": "trilinear", "label": "Trilinear"})
    return series


def _build_model_zoo(bm: dict, ln: dict, l1s: dict) -> list[dict]:
    """All entries from the repo-local model zoo (the published models plus the
    trilinear baseline). Works without access to the scratch run dirs."""
    series = []
    zoo = _model_zoo_manifest()
    for e in zoo["models"]:
        series.append({"manifest": zoo, "model": e["name"], "label": e["label"]})
    return series


_PRESET_BUILDERS = {
    "baseline": _build_baseline,
    "loss_norm": _build_loss_norm,
    "loss_norm_cfno": _build_loss_norm_cfno,
    "best_performing": _build_best_performing,
    "l1_spectral_best_vs_best_performing": (_build_l1_spectral_best_vs_best_performing),
    "l1_spectral_skip": _build_l1_spectral_skip,
    "ufno": _build_ufno,
    "loss_ablation_mse_vs_edsr": _build_loss_ablation_mse_vs_edsr,
    "mse_spectral": _build_mse_spectral,
    "edsr_norm_skip": _build_edsr_norm_skip,
    "edsr_vs_cfno_baseline_a": _build_edsr_vs_cfno_baseline_a,
    "spectral_weighting_l1_vs_mse": _build_spectral_weighting_l1_vs_mse,
    "baselines_vs_edsr_norm_skip": _build_baselines_vs_edsr_norm_skip,
    "baseline_b_500_vs_edsr": _build_baseline_b_500_vs_edsr,
    "ufno_mse_spectral_500": _build_ufno_mse_spectral_500,
    "comparing_best_models_mse": _build_comparing_best_models_mse,
    "mse_loss_combinations": _build_mse_loss_combinations,
    "model_zoo": _build_model_zoo,
}


# =====================================================================
# Helpers
# =====================================================================


def _parse_filter(filter_str: str) -> dict:
    """Parse 'col1=val1,col2=val2' into a dict (values coerced to int/float/str)."""
    result = {}
    for part in filter_str.split(","):
        key, val = part.split("=", 1)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        result[key] = val
    return result


def _load_series_data(
    series: list[dict], upsample_factor: int = 4
) -> pd.DataFrame:
    """Load metric values for each series from its manifest's benchmark CSV.

    Returns a DataFrame indexed by series label with one column per metric.
    ``upsample_factor`` selects which evaluation-scale row to read (models
    whose benchmark CSV has no row at that scale are skipped with a warning).
    """
    rows = {}
    for s in series:
        manifest = s["manifest"]
        csv_path = manifest["benchmark_csv"]
        if not csv_path.exists():
            print(f"  Warning: CSV not found, skipping '{s['label']}': {csv_path}")
            continue
        entry = get_model_entry(manifest, s["model"])
        row_key = manifest["benchmark_row_key"]
        row_value = entry[manifest["benchmark_row_value_field"]]

        df = pd.read_csv(csv_path)
        if row_key not in df.columns:
            print(
                f"  Warning: column '{row_key}' not in {csv_path.name}, "
                f"skipping '{s['label']}'"
            )
            continue
        matches = df[df[row_key] == row_value]
        if "upsample_factor" in df.columns:
            matches = matches[matches["upsample_factor"] == upsample_factor]
        if matches.empty:
            print(
                f"  Warning: no row matched {row_key}={row_value!r} "
                f"in {csv_path.name}, skipping '{s['label']}'"
            )
            continue
        if len(matches) > 1:
            print(
                f"  Warning: {len(matches)} rows matched for "
                f"'{s['label']}', using first"
            )
        row = matches.iloc[0]
        rows[s["label"]] = {m: row.get(m, np.nan) for m, _ in METRICS}
    if not rows:
        raise RuntimeError(
            "No series data could be loaded — check manifest benchmark_csv paths."
        )
    return pd.DataFrame(rows).T


def _load_csv_series(csv_path: Path, filter_dict: dict, label: str) -> pd.DataFrame:
    """Legacy single-CSV series loader (for ``--csv`` / ``--filter``)."""
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    mask = pd.Series([True] * len(df))
    for col, val in filter_dict.items():
        if col not in df.columns:
            raise RuntimeError(f"column '{col}' not in {csv_path.name}")
        mask &= df[col] == val
    matches = df[mask]
    if matches.empty:
        raise RuntimeError(f"no row matched {filter_dict} in {csv_path.name}")
    rows = {label: {m: matches.iloc[0].get(m, np.nan) for m, _ in METRICS}}
    return pd.DataFrame(rows).T


# =====================================================================
# Plotting
# =====================================================================


def _plot_bars(data: pd.DataFrame, save_path: Path, title: str) -> None:
    n_metrics = len(METRICS)
    n_series = len(data)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_series))
    bar_width = 0.8 / max(n_series, 1)

    for idx, (metric, direction) in enumerate(METRICS):
        ax = axes[idx]
        if metric not in data.columns:
            ax.set_visible(False)
            continue
        values = data[metric].values
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            ax.set_visible(False)
            continue

        for i, (label, val) in enumerate(zip(data.index, values)):
            if not np.isfinite(val):
                continue
            ax.bar(i * bar_width, val, bar_width, color=colors[i], label=label)

        ax.set_xticks([])
        ax.set_title(f"{metric} {direction}", fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        vals_finite = values[finite_mask]
        vmin, vmax = float(vals_finite.min()), float(vals_finite.max())
        margin = (vmax - vmin) * 0.15 if vmax > vmin else abs(vmax) * 0.1 + 0.01
        ax.set_ylim(max(0, vmin - margin), vmax + margin)

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(n_series, 6), fontsize=8
        )

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="baseline",
        help="Which preset series configuration to use.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to the experiment folder (containing manifest.json).",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated entry names from the manifest (with --manifest). "
        "Defaults to all entries.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a benchmark CSV (legacy ad-hoc single-series mode).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help=(
            "Comma-separated column=value filters for the custom CSV "
            "(e.g. model=CFNO (shift=8),upsample_factor=4)."
        ),
    )
    parser.add_argument(
        "--label", type=str, default="custom", help="Label for the custom series."
    )
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=4,
        help=(
            "Evaluation scale row to read from each benchmark CSV "
            "(default: 4). Models without a row at this scale are skipped "
            "with a warning. Output filename/title gain an '_x<N>' / '@ xN' "
            "suffix when this is not 4."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        help=(
            "Output PNG path (default: "
            "evaluation/comparing_models_bar_chart_<preset>.png)."
        ),
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        filter_dict = _parse_filter(args.filter) if args.filter else {}
        data = _load_csv_series(csv_path, filter_dict, args.label)
        title = f"Model comparison — {args.label}"
        out_name = "comparing_models_bar_chart_custom.png"
    elif args.manifest:
        manifest = load_manifest(args.manifest)
        names = args.models.split(",") if args.models else None
        series = []
        for e in manifest["models"]:
            if names and e["name"] not in names:
                continue
            series.append(
                {"manifest": manifest, "model": e["name"], "label": e["label"]}
            )
        if not series:
            raise RuntimeError(f"No matching models in manifest (names={names}).")
        title = f"Model comparison — {manifest.get('experiment', 'custom')}"
        out_name = "comparing_models_bar_chart_custom.png"
        data = _load_series_data(series, args.upsample_factor)
    else:
        print("Discovering manifests …")

        def _try(discover):
            try:
                return discover()
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                return None

        # Presets that only need the repo-local model zoo work scratch-free;
        # other presets fail downstream if their manifest is missing (as before).
        bm = _try(_best_models_manifest)
        ln = _try(_loss_norm_manifest)
        l1s = _try(_l1_spectral_manifest)
        series = _PRESET_BUILDERS[args.preset](bm, ln, l1s)
        scale_tag = f" @ x{args.upsample_factor}" if args.upsample_factor != 4 else ""
        title = f"Model comparison — {args.preset} preset{scale_tag}"
        suffix = f"_x{args.upsample_factor}" if args.upsample_factor != 4 else ""
        out_name = f"comparing_models_bar_chart_{args.preset}{suffix}.png"
        data = _load_series_data(series, args.upsample_factor)

    out_path = Path(args.output) if args.output else OUTPUT_DIR / out_name

    print(f"Series: {len(data)}  |  Output: {out_path}")
    print(f"Labels: {list(data.index)}")
    _plot_bars(data, out_path, title)


if __name__ == "__main__":
    main()
