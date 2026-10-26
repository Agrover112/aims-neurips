#!/usr/bin/env python3
"""Aggregate evaluation results into summary tables and visualizations.

Reads all parquet files from offline testing and creates:
  - Summary metrics table (AUC, Brier, ECE, LogLoss by model/benchmark/seed)
  - Training curves plots
  - Calibration diagrams
  - Result tables for README

Usage:
  python aggregate_results.py --metrics-only
  python aggregate_results.py --all                # everything
  python aggregate_results.py --plots-only         # skip metrics, do plots
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path("/data/results/data-analysis")  # Modal volume
OUT_PDF = REPO_ROOT / "model-results-summary.pdf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation pipeline results"
    )

    parser.add_argument(
        "--all", action="store_true",
        help="Do everything: metrics + plots"
    )

    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Compute offline metrics only"
    )

    parser.add_argument(
        "--plots-only", action="store_true",
        help="Generate plots only (assumes metrics already computed)"
    )

    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Results directory (default: {RESULTS_DIR})"
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done"
    )

    return parser.parse_args()


def compute_metrics(args):
    """Run modal_offline_metrics.py to aggregate metrics."""

    print("\n" + "="*70)
    print("Computing offline metrics (AUC, Brier, ECE, LogLoss)...")
    print("="*70)

    cmd = f"modal run --detach {REPO_ROOT}/organized/testing/modal_offline_metrics.py"

    if args.dry_run:
        print(f"Would run: {cmd}")
        return True

    try:
        result = subprocess.run(cmd, shell=True, cwd=REPO_ROOT, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def generate_plots(args):
    """Generate training curves, calibration, and acquisition plots."""

    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)

    utils_dir = REPO_ROOT / "organized" / "utils"

    plots = [
        {
            "name": "Training Curves",
            "script": utils_dir / "plot_training_curves.py",
            "args": f"{RESULTS_DIR} {REPO_ROOT}/training-curves.pdf"
        },
        {
            "name": "Calibration Curves",
            "script": utils_dir / "plot_calibration_curves.py",
            "args": f"{RESULTS_DIR} {REPO_ROOT}/calibration-curves.pdf"
        },
        {
            "name": "Online Acquisition",
            "script": utils_dir / "plot_online_acquisition.py",
            "args": f"{RESULTS_DIR} {REPO_ROOT}/online-acquisition.pdf"
        },
    ]

    for plot_info in plots:
        print(f"\n{plot_info['name']}...", end=" ")

        cmd = f"python {plot_info['script']} {plot_info['args']}"

        if args.dry_run:
            print(f"Would run: {cmd}")
            continue

        try:
            result = subprocess.run(cmd, shell=True, cwd=REPO_ROOT, timeout=120)
            if result.returncode == 0:
                print("✓")
            else:
                print(f"✗ (code {result.returncode})")
        except Exception as e:
            print(f"✗ ({e})")


def summarize_metrics(args):
    """Print summary of computed metrics."""

    metrics_file = args.results_dir / "offline-metrics-summary.parquet"

    if not metrics_file.exists():
        print(f"\nMetrics file not found: {metrics_file}")
        print("Run with --metrics-only first")
        return

    print("\n" + "="*70)
    print("Metrics Summary")
    print("="*70)

    try:
        df = pd.read_parquet(metrics_file)

        print(f"\nTotal results: {len(df)} rows")
        print(f"Models: {df['model'].nunique()}")
        print(f"Benchmarks: {df['benchmark_id'].nunique()}")
        print(f"Seeds: {df['seed'].nunique()}")

        # Summary by model
        print("\n" + "-"*70)
        print("Per-Model Summary (mean AUC across benchmarks/seeds):")
        print("-"*70)

        if "auc" in df.columns:
            model_auc = df.groupby("model")["auc"].agg(["mean", "std", "count"])
            print(model_auc.to_string())
        else:
            print("(No pooled AUC; metrics computed per-instance)")
            if "auc_mean" in df.columns:
                model_auc = df.groupby("model")["auc_mean"].mean()
                print(model_auc.to_string())

        # By axis
        print("\n" + "-"*70)
        print("Subject vs Item Generalization:")
        print("-"*70)

        for axis in ["subject", "item"]:
            if axis in df["axis"].values:
                axis_df = df[df["axis"] == axis]
                if "auc" in axis_df.columns:
                    print(f"\n{axis.capitalize()}:")
                    print(f"  AUC: {axis_df['auc'].mean():.3f} ± {axis_df['auc'].std():.3f}")
                    print(f"  Brier: {axis_df['brier'].mean():.3f} ± {axis_df['brier'].std():.3f}")
                    if "ece" in axis_df.columns:
                        print(f"  ECE: {axis_df['ece'].mean():.3f} ± {axis_df['ece'].std():.3f}")

    except Exception as e:
        print(f"Error reading metrics: {e}")


def main():
    args = parse_args()

    # Default to --all if nothing specified
    if not args.metrics_only and not args.plots_only:
        args.all = True

    success = True

    if args.all or args.metrics_only:
        if not compute_metrics(args):
            success = False

    if args.all or args.plots_only:
        generate_plots(args)

    # Summarize
    summarize_metrics(args)

    print("\n" + "="*70)
    if args.dry_run:
        print("DRY RUN: Would have completed aggregation")
    else:
        print("Aggregation complete")
    print("="*70 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
