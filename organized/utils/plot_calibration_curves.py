"""Plot calibration curves and reliability diagrams from offline metrics.

Reads pooled (label, pred_prob) pairs from generalization test results and
plots a reliability diagram per model and generalization axis.

Usage: python plot_calibration_curves.py <results_dir> [output.pdf]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import INK, MUTED, SUBTLE, TEXTWIDTH, header, style

if len(sys.argv) < 2:
    print("Usage: python plot_calibration_curves.py <results_dir> [output.pdf]")
    sys.exit(1)

RESULTS_DIR = Path(sys.argv[1])
OUT_PDF = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_DIR.parent / "calibration-curves.pdf"

gen_files = sorted(RESULTS_DIR.glob("*generalization*seed*.parquet"))

if not gen_files:
    print(f"No generalization results found in {RESULTS_DIR}")
    sys.exit(1)

print(f"Found {len(gen_files)} generalization result files")

N_BINS = 10


def _binned(y_true, y_pred, n_bins=N_BINS):
    """Observed event rate and mean confidence per equal-width confidence bin."""
    idx = np.clip((y_pred * n_bins).astype(int), 0, n_bins - 1)
    conf, obs, count = [], [], []
    for i in range(n_bins):
        mask = idx == i
        count.append(int(mask.sum()))
        if mask.any():
            conf.append(float(y_pred[mask].mean()))
            # The event rate, NOT classification accuracy. Accuracy against a
            # 0.5 threshold does not lie on the diagonal for a calibrated
            # model, so plotting it mislabels the y axis.
            obs.append(float(y_true[mask].mean()))
        else:
            conf.append(np.nan)
            obs.append(np.nan)
    return np.array(conf), np.array(obs), np.array(count)


def _ece(conf, obs, count):
    keep = count > 0
    w = count[keep] / count[keep].sum()
    return float((w * np.abs(obs[keep] - conf[keep])).sum())


def plot_reliability_diagram(ax, y_true, y_pred, n_bins=N_BINS):
    conf, obs, count = _binned(y_true, y_pred, n_bins)
    keep = count > 0
    width = 0.92 / n_bins
    centres = (np.arange(n_bins) + 0.5) / n_bins

    ax.bar(centres[keep], obs[keep], width=width, color=MUTED["cyan"],
           edgecolor=INK, lw=0.4, zorder=2, label="Observed")
    ax.bar(centres[keep], conf[keep] - obs[keep], width=width, bottom=obs[keep],
           color=MUTED["rose"], alpha=0.45, edgecolor=MUTED["wine"], lw=0.4,
           hatch="////", zorder=2, label="Gap")
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=(0, (4, 2)), zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_aspect("equal")
    style(ax, grid_axis="both")
    return _ece(conf, obs, count)


def _pool(axis_df):
    y_true, y_pred = [], []
    for label, pred in zip(axis_df["label"], axis_df["pred_prob"]):
        if label is None or pred is None:
            continue
        if hasattr(label, "__len__"):
            y_true.extend(np.asarray(label, dtype=float))
            y_pred.extend(np.asarray(pred, dtype=float))
        else:
            y_true.append(float(label))
            y_pred.append(float(pred))
    return np.asarray(y_true), np.asarray(y_pred)


with PdfPages(OUT_PDF) as pdf:
    for gen_file in gen_files:
        print(f"Processing {gen_file.name}")

        try:
            df = pd.read_parquet(gen_file)
        except Exception as exc:
            print(f"  Error loading {gen_file}: {exc}")
            continue

        if "label" not in df.columns or "pred_prob" not in df.columns:
            print("  No predictions found (label/pred_prob columns missing)")
            continue

        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(len(models), 2,
                                 figsize=(TEXTWIDTH, 2.3 * len(models) + 0.4),
                                 squeeze=False)

        for row, model in enumerate(models):
            model_df = df[df["model"] == model]
            for col, axis_name in enumerate(["subject", "item"]):
                ax = axes[row][col]
                axis_df = model_df[model_df["axis"] == axis_name]
                y_true, y_pred = _pool(axis_df) if not axis_df.empty else ([], [])

                if len(y_true) == 0:
                    ax.text(0.5, 0.5, "no predictions", ha="center", va="center",
                            fontsize=7, color=SUBTLE, transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"{model} — {axis_name}", loc="left", fontweight="bold")
                    style(ax, grid_axis="none")
                    continue

                ece = plot_reliability_diagram(ax, y_true, y_pred)
                ax.set_title(f"{model} — {axis_name}", loc="left", fontweight="bold")
                # Upper left is the empty triangle above the diagonal; the
                # lower right is where the bars are.
                ax.text(0.04, 0.96, f"ECE {ece:.3f}\nn={len(y_true):,}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6, color=SUBTLE)

        handles = [plt.Rectangle((0, 0), 1, 1, fc=MUTED["cyan"], ec=INK, lw=0.4),
                   plt.Rectangle((0, 0), 1, 1, fc=MUTED["rose"], alpha=0.45,
                                 ec=MUTED["wine"], lw=0.4, hatch="////"),
                   plt.Line2D([], [], color=INK, lw=0.8, ls=(0, (4, 2)))]
        header(fig, f"Calibration: {gen_file.stem}", f"{len(models)} models, {N_BINS} bins")
        fig.tight_layout(pad=0.3, w_pad=1.2, h_pad=0.8, rect=(0, 0.04, 1, 0.94))
        fig.legend(handles, ["Observed", "Gap to perfect", "Perfect calibration"],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved to {OUT_PDF}")
