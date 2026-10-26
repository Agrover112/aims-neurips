"""Plot calibration curves and reliability diagrams from offline metrics.

Reads predictions from generalization test results and plots:
- Calibration curves (predicted vs observed probability)
- Reliability diagrams (bin-based calibration)
- Per-model calibration summary

Usage: python plot_calibration_curves.py <metrics.parquet> <output.pdf>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

if len(sys.argv) < 2:
    print("Usage: python plot_calibration_curves.py <results_dir> <output.pdf>")
    sys.exit(1)

RESULTS_DIR = Path(sys.argv[1])
OUT_PDF = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_DIR.parent / "calibration-curves.pdf"

# Find generalization result files
gen_files = sorted(RESULTS_DIR.glob("*generalization*seed*.parquet"))

if not gen_files:
    print(f"No generalization results found in {RESULTS_DIR}")
    sys.exit(1)

print(f"Found {len(gen_files)} generalization result files")

def plot_reliability_diagram(ax, y_true, y_pred, n_bins=10):
    """Plot reliability diagram with calibration curve."""

    # Compute bin-based calibration
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if not mask.any():
            accuracies.append(np.nan)
            confidences.append(np.nan)
            counts.append(0)
        else:
            acc = (y_true[mask] == (y_pred[mask] >= 0.5)).mean()
            conf = y_pred[mask].mean()
            accuracies.append(acc)
            confidences.append(conf)
            counts.append(mask.sum())

    # Plot bars
    bin_width = 1.0 / n_bins
    for i, (conf, acc, count) in enumerate(zip(confidences, accuracies, counts)):
        if count == 0:
            continue
        color = "#2b6ca3" if not np.isnan(acc) else "#ccc"
        height = acc if not np.isnan(acc) else 0
        ax.add_patch(Rectangle((conf - bin_width/2, 0), bin_width, height,
                              alpha=0.6, facecolor=color, edgecolor="black", linewidth=1))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Empirical calibration curve (smoother)
    valid_conf = [c for c, a in zip(confidences, accuracies) if not np.isnan(a)]
    valid_acc = [a for a in accuracies if not np.isnan(a)]
    if len(valid_conf) > 1:
        ax.plot(valid_conf, valid_acc, "o-", color="#d98324", linewidth=2, markersize=6, label="Empirical")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability (Confidence)")
    ax.set_ylabel("Actual Probability (Accuracy)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()


with PdfPages(OUT_PDF) as pdf:
    for gen_file in gen_files:
        print(f"Processing {gen_file.name}")

        try:
            df = pd.read_parquet(gen_file)
        except Exception as e:
            print(f"  Error loading {gen_file}: {e}")
            continue

        if "label" not in df.columns or "pred_prob" not in df.columns:
            print(f"  No predictions found (label/pred_prob columns missing)")
            continue

        # Create figure with subplots per model/axis
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)))

        if len(models) == 1:
            axes = [axes]

        for model_idx, model in enumerate(models):
            model_df = df[df["model"] == model]

            for axis_idx, axis_name in enumerate(["subject", "item"]):
                ax = axes[model_idx][axis_idx]
                axis_df = model_df[model_df["axis"] == axis_name]

                if axis_df.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    ax.set_title(f"{model} - {axis_name}")
                    continue

                # Pool predictions
                y_true_all = []
                y_pred_all = []

                for _, row in axis_df.iterrows():
                    label = row.get("label")
                    pred = row.get("pred_prob")

                    if label is not None and pred is not None:
                        try:
                            if hasattr(label, "__len__"):
                                y_true_all.extend(np.asarray(label))
                                y_pred_all.extend(np.asarray(pred))
                            else:
                                y_true_all.append(label)
                                y_pred_all.append(pred)
                        except:
                            pass

                if len(y_true_all) > 0:
                    y_true_all = np.asarray(y_true_all)
                    y_pred_all = np.asarray(y_pred_all)
                    plot_reliability_diagram(ax, y_true_all, y_pred_all)
                    ax.set_title(f"{model} - {axis_name} (n={len(y_true_all)})")
                else:
                    ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
                    ax.set_title(f"{model} - {axis_name}")

        fig.suptitle(f"Calibration Curves: {gen_file.stem}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"✓ Saved to {OUT_PDF}")
