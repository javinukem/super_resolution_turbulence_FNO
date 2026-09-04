"""
Bar-chart comparison of model metrics across experiments.

Reads benchmark CSV files referenced by experiment manifests and produces a
grouped bar chart with one subplot per metric (MSE, per-channel losses,
spectral, perceptual, PSNR, SSIM).

The script is configuration-driven: choose a ``--preset`` (which references one
or more manifests and picks entries by name) or pass ``--manifest`` (optionally
with ``--models name1,name2``) for a custom comparison.  Each series loads one
row from a manifest's ``benchmark_csv`` by filtering on the manifest's
``benchmark_row_key`` column.

Presets cover the retained published-model-backed experiments:

  - ``edsr_norm_skip``       — the two EDSR norm-skip runs + trilinear
  - ``comparing_best_models_mse`` — MSE-only CFNO vs EDSR + trilinear
  - ``mse_loss_combinations``     — MSE-based loss combos + MSE-only + trilinear

Usage
-----
    # a retained experiment's comparison
    python evaluation/comparing_models_bar_chart.py --preset edsr_norm_skip

    # all published models + trilinear (repo-local manifest, no scratch needed)
    python evaluation/comparing_models_bar_chart.py --manifest experiments \
        --output experiments/comparing_models_bar_chart_published_models.png

    # custom: pick models from a manifest
    python evaluation/comparing_models_bar_chart.py --manifest experiments \
        --models edsr,trilinear

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
    "edsr_norm_skip",
    "comparing_best_models_mse",
    "mse_loss_combinations",
]


# =====================================================================
# Manifest discovery
# =====================================================================


def _edsr_norm_skip_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "edsr_norm_skip_*"))


def _comparing_best_models_mse_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "comparing_best_models_mse_*"))


def _mse_loss_combinations_manifest() -> dict:
    return load_manifest(discover_latest(SCRATCH_BASE, "mse_loss_combinations_*"))


def _published_manifest() -> dict:
    """The repo-local published-models manifest at ``experiments/manifest.json``
    (no scratch discovery needed)."""
    return load_manifest(ROOT / "experiments")


def _trilinear_series() -> dict:
    """Trilinear context row, sourced from the published-models manifest (whose
    benchmark CSV carries trilinear rows at both eval scales)."""
    pub = _published_manifest()
    return {"manifest": pub, "model": "trilinear", "label": "Trilinear"}


# =====================================================================
# Preset builders — each returns a list of series specs
# =====================================================================


def _build_edsr_norm_skip() -> list[dict]:
    """Both EDSR norm-on runs (skip off/on) + trilinear baseline (for context)."""
    series = []
    try:
        ens = _edsr_norm_skip_manifest()
    except FileNotFoundError as e:
        print(f"  Warning: edsr_norm_skip manifest not found ({e}) — skipping")
        return series
    for e in ens["models"]:
        series.append({"manifest": ens, "model": e["name"], "label": e["label"]})
    series.append(_trilinear_series())
    return series


def _build_comparing_best_models_mse() -> list[dict]:
    """MSE-only CFNO (canonical stabilized Baseline B config, 500e clip_p30)
    vs the best MSE-only EDSR (edsr, "Baseline D") + trilinear
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
            if e.get("name") == "edsr":
                series.append(
                    {
                        "manifest": ens,
                        "model": e["name"],
                        "label": "Baseline D (EDSR norm, skip on)",
                    }
                )
    series.append(_trilinear_series())
    return series


def _build_mse_loss_combinations() -> list[dict]:
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
    series.append(_trilinear_series())
    return series


_PRESET_BUILDERS = {
    "edsr_norm_skip": _build_edsr_norm_skip,
    "comparing_best_models_mse": _build_comparing_best_models_mse,
    "mse_loss_combinations": _build_mse_loss_combinations,
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

    if not (args.csv or args.manifest or args.preset):
        parser.error("one of --preset, --manifest or --csv is required")

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
        series = _PRESET_BUILDERS[args.preset]()
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
