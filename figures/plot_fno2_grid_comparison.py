"""
Plot line plots comparing different losses and statistics from benchmark_metrics.csv
for the FNO2 grid experiment.

This script:
1. Loads benchmark_metrics.csv from the grid experiment output
2. Creates line plots comparing metrics across different model configurations
3. Groups comparisons by: modes, interpolation, skip connection, head variant, loss combo
"""

from autocvd import autocvd

autocvd(num_gpus=1)

import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Path to the grid experiment output
# Update this to point to your specific experiment folder:
# /export/scratch/jalegria/experiments/train_fno2_grid_modes_interp_skip_losses_refine/<timestamp>/benchmark_metrics.csv
BENCHMARK_CSV_PATH = Path(
    "/export/scratch/jalegria/experiments/train_fno2_grid_modes_interp_skip_losses_refine"
    "/fno2_grid_modes_interp_skip_losses_refine_2026-03-25_18-01-49/benchmark_metrics.csv"
)

OUTPUT_DIR = Path(
    "/export/scratch/jalegria/experiments/train_fno2_grid_modes_interp_skip_losses_refine"
    "/fno2_grid_modes_interp_skip_losses_refine_2026-03-25_18-01-49/comparison_plots"
)


# ── Parsing model names ───────────────────────────────────────────────


def parse_model_name(model_name: str) -> dict:
    """Parse a model name into its components."""
    # e.g. fno2_m16_shift8_interp-nearest_skip-0_head-mlp_tail_loss-mse
    result = {
        "modes": None,
        "interpolation": None,
        "skip": None,
        "head": None,
        "loss_combo": None,
    }

    try:
        parts = model_name.split("_")

        # Extract modes (m16 -> 16)
        for p in parts:
            if p.startswith("m") and p[1:].isdigit():
                result["modes"] = int(p[1:])
                break

        # Extract interpolation
        for p in parts:
            if p.startswith("interp-"):
                result["interpolation"] = p.split("-")[1]
                break

        # Extract skip
        for p in parts:
            if p.startswith("skip-"):
                result["skip"] = "Yes" if p.split("-")[1] == "1" else "No"
                break

        # Extract head variant (could be "mlp_tail" or "conv_refine_tail")
        head_start = None
        loss_start = None
        for i, p in enumerate(parts):
            if p.startswith("head-"):
                head_start = i
            if p.startswith("loss-"):
                loss_start = i
                break

        if head_start is not None and loss_start is not None:
            head_parts = parts[head_start:loss_start]
            result["head"] = "_".join(head_parts).replace("head-", "")

        # Extract loss combo
        for i, p in enumerate(parts):
            if p.startswith("loss-"):
                loss_parts = parts[i:]
                result["loss_combo"] = "_".join(loss_parts).replace("loss-", "")
                break

    except Exception as e:
        print(f"Warning: Failed to parse {model_name}: {e}")

    return result


def load_and_enrich_data(csv_path: Path) -> pd.DataFrame:
    """Load benchmark CSV and add parsed columns."""
    df = pd.read_csv(csv_path)

    # Parse model names into separate columns
    parsed = df["model"].apply(parse_model_name).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    return df


# ── Plotting functions ────────────────────────────────────────────────


def plot_metric_by_factor(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    output_path: Path,
    title: str,
    filter_col: Optional[str] = None,
    filter_val: Optional[str] = None,
) -> None:
    """
    Create a grouped bar/line plot showing a metric across different configurations.
    """
    plot_df = df.copy()
    if filter_col and filter_val:
        plot_df = plot_df[plot_df[filter_col] == filter_val]

    if plot_df.empty:
        print(f"Warning: No data for {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by the specified column and upsample_factor
    unique_factors = sorted(plot_df["upsample_factor"].unique())
    unique_groups = sorted(plot_df[group_col].dropna().unique())

    x = np.arange(len(unique_groups))
    width = 0.35

    for i, factor in enumerate(unique_factors):
        factor_df = plot_df[plot_df["upsample_factor"] == factor]
        means = []
        stds = []
        for g in unique_groups:
            group_data = factor_df[factor_df[group_col] == g][metric]
            means.append(group_data.mean() if len(group_data) > 0 else 0)
            stds.append(group_data.std() if len(group_data) > 1 else 0)

        offset = (i - len(unique_factors) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=f"x{factor}",
            yerr=stds,
            capsize=3,
        )

    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_groups, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    row_col: str,
    col_col: str,
    output_path: Path,
    title: str,
    upsample_factor: int = 4,
) -> None:
    """Create a heatmap showing metric values across two categorical variables."""
    plot_df = df[df["upsample_factor"] == upsample_factor].copy()

    if plot_df.empty:
        print(f"Warning: No data for {title}")
        return

    # Pivot to create heatmap data
    pivot = plot_df.pivot_table(
        values=metric,
        index=row_col,
        columns=col_col,
        aggfunc="mean",
    )

    if pivot.empty:
        print(f"Warning: Empty pivot table for {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",  # Lower is better for losses
        ax=ax,
        cbar_kws={"label": metric},
    )
    ax.set_title(f"{title} (x{upsample_factor})")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_all_metrics_comparison(
    df: pd.DataFrame,
    output_path: Path,
    upsample_factor: int = 4,
) -> None:
    """Create a comprehensive comparison of all metrics."""
    metrics = [
        "MSE",
        "Loss_density",
        "Loss_pressure",
        "Loss_v_norm",
        "Loss_vorticity",
        "Spectral_MSE",
        "PSNR",
        "SSIM",
    ]

    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns]

    plot_df = df[df["upsample_factor"] == upsample_factor].copy()
    if plot_df.empty:
        print(f"Warning: No data for upsample_factor={upsample_factor}")
        return

    # Create a figure with subplots for each metric
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]

        # Group by loss_combo and compute mean metric
        grouped = (
            plot_df.groupby("loss_combo")[metric].agg(["mean", "std"]).reset_index()
        )

        colors = plt.cm.Set2(np.linspace(0, 1, len(grouped)))
        bars = ax.bar(
            grouped["loss_combo"],
            grouped["mean"],
            yerr=grouped["std"],
            capsize=3,
            color=colors,
        )

        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    # Remove empty subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Metrics Comparison by Loss Function (x{upsample_factor})", fontsize=14
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scatter_metrics(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_col: str,
    output_path: Path,
    title: str,
    upsample_factor: int = 4,
) -> None:
    """Create a scatter plot comparing two metrics."""
    plot_df = df[df["upsample_factor"] == upsample_factor].copy()

    if plot_df.empty:
        print(f"Warning: No data for {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_colors = plot_df[color_col].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_colors)))

    for c_val, color in zip(unique_colors, colors):
        subset = plot_df[plot_df[color_col] == c_val]
        ax.scatter(
            subset[x_metric],
            subset[y_metric],
            label=str(c_val),
            color=color,
            s=100,
            alpha=0.7,
        )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_line_by_modes(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    """Create line plots showing metric vs modes for different configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, factor in zip(axes, [4, 2]):
        plot_df = df[df["upsample_factor"] == factor].copy()

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        # Group by modes and loss_combo
        for loss_combo in plot_df["loss_combo"].dropna().unique():
            subset = plot_df[plot_df["loss_combo"] == loss_combo]
            grouped = subset.groupby("modes")[metric].agg(["mean", "std"]).reset_index()
            grouped = grouped.sort_values("modes")

            ax.errorbar(
                grouped["modes"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                label=loss_combo,
                capsize=3,
            )

        ax.set_xlabel("Modes")
        ax.set_ylabel(metric)
        ax.set_title(f"x{factor} upsampling")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    """Create a summary table of best models per metric."""
    metrics = [
        "MSE",
        "Loss_vorticity",
        "Spectral_MSE",
        "PSNR",
        "SSIM",
    ]
    available_metrics = [m for m in metrics if m in df.columns]

    summary_rows = []
    for factor in df["upsample_factor"].unique():
        factor_df = df[df["upsample_factor"] == factor]

        for metric in available_metrics:
            # For PSNR and SSIM, higher is better
            if metric in ["PSNR", "SSIM"]:
                best_idx = factor_df[metric].idxmax()
            else:
                best_idx = factor_df[metric].idxmin()

            if pd.isna(best_idx):
                continue

            best_row = factor_df.loc[best_idx]
            summary_rows.append(
                {
                    "upsample_factor": factor,
                    "metric": metric,
                    "best_value": best_row[metric],
                    "best_model": best_row["model"],
                    "modes": best_row.get("modes"),
                    "interpolation": best_row.get("interpolation"),
                    "skip": best_row.get("skip"),
                    "head": best_row.get("head"),
                    "loss_combo": best_row.get("loss_combo"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    if not BENCHMARK_CSV_PATH.exists():
        print(f"Error: Benchmark CSV not found at {BENCHMARK_CSV_PATH}")
        print("Please update BENCHMARK_CSV_PATH to point to your benchmark_metrics.csv")
        return

    print(f"Loading data from: {BENCHMARK_CSV_PATH}")
    df = load_and_enrich_data(BENCHMARK_CSV_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique models: {df['model'].nunique()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Bar plots by factor (modes, interpolation, skip, head, loss_combo)
    for group_col in ["modes", "interpolation", "skip", "head", "loss_combo"]:
        if group_col not in df.columns or df[group_col].isna().all():
            print(f"Skipping {group_col}: no data")
            continue

        for metric in ["MSE", "Loss_vorticity", "Spectral_MSE", "PSNR"]:
            if metric not in df.columns:
                continue
            plot_metric_by_factor(
                df,
                metric,
                group_col,
                OUTPUT_DIR / f"bar_{metric}_by_{group_col}.png",
                f"{metric} by {group_col.replace('_', ' ').title()}",
            )

    # 2. Heatmaps
    for metric in ["MSE", "Loss_vorticity", "PSNR"]:
        if metric not in df.columns:
            continue
        for factor in [4, 2]:
            plot_metric_heatmap(
                df,
                metric,
                "interpolation",
                "skip",
                OUTPUT_DIR / f"heatmap_{metric}_interp_skip_x{factor}.png",
                f"{metric}: Interpolation vs Skip",
                upsample_factor=factor,
            )
            plot_metric_heatmap(
                df,
                metric,
                "head",
                "loss_combo",
                OUTPUT_DIR / f"heatmap_{metric}_head_loss_x{factor}.png",
                f"{metric}: Head vs Loss",
                upsample_factor=factor,
            )

    # 3. All metrics comparison by loss function
    for factor in [4, 2]:
        plot_all_metrics_comparison(
            df,
            OUTPUT_DIR / f"all_metrics_by_loss_x{factor}.png",
            upsample_factor=factor,
        )

    # 4. Scatter plots
    if "MSE" in df.columns and "Loss_vorticity" in df.columns:
        for factor in [4, 2]:
            plot_scatter_metrics(
                df,
                "MSE",
                "Loss_vorticity",
                "loss_combo",
                OUTPUT_DIR / f"scatter_mse_vorticity_x{factor}.png",
                f"MSE vs Vorticity Loss (x{factor})",
                upsample_factor=factor,
            )

    if "MSE" in df.columns and "PSNR" in df.columns:
        for factor in [4, 2]:
            plot_scatter_metrics(
                df,
                "MSE",
                "PSNR",
                "loss_combo",
                OUTPUT_DIR / f"scatter_mse_psnr_x{factor}.png",
                f"MSE vs PSNR (x{factor})",
                upsample_factor=factor,
            )

    # 5. Line plots by modes
    for metric in ["MSE", "Loss_vorticity", "Spectral_MSE", "PSNR"]:
        if metric not in df.columns:
            continue
        plot_line_by_modes(
            df,
            metric,
            OUTPUT_DIR / f"line_{metric}_by_modes.png",
            f"{metric} vs Modes",
        )

    # 6. Summary table
    create_summary_table(df, OUTPUT_DIR / "best_models_summary.csv")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
