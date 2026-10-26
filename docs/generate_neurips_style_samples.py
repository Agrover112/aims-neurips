"""Generate NeurIPS-style sample plots with colorblind-accessible palette."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# Okabe-Ito colorblind-safe palette
COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "red": "#CC78BC",
    "yellow": "#CA9161",
    "green": "#56B4E9",
    "purple": "#029E73",
    "grey": "#999999",
}

DARK = "#333333"
LIGHT_GREY = "#EEEEEE"

def style_axis(ax, remove_spines=True):
    """Apply NeurIPS publication style to axis."""
    if remove_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

def add_header(fig, title, subtitle=""):
    """Add publication-style header."""
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.05, ha="left", y=0.97)
    if subtitle:
        fig.text(0.05, 0.925, subtitle, fontsize=9, color=DARK, va="top", style="italic")

# Create PDF
pdf_path = "/tmp/claude-1000/-home-agrov-aims-neurips/6b98ee46-f61c-48ba-82af-472871f97afc/scratchpad/neurips-sample-plots.pdf"
with PdfPages(pdf_path) as pdf:

    # ========== Page 1: Training Curves ==========
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Training Loss Curves", "Convergence over 300 steps (3 seeds, n≥30 filtered data)")

    models = ["1PL", "2PL", "3PL"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["red"]]

    for idx, (model, color) in enumerate(zip(models, colors)):
        ax = fig.add_subplot(2, 2, idx + 1)

        # Synthetic data: loss curves
        steps = np.arange(0, 300)
        for seed in range(3):
            noise = np.random.RandomState(seed).normal(0, 0.01, len(steps))
            loss = 0.5 * np.exp(-steps / 100) + 0.1 + noise
            ax.plot(steps, loss, linewidth=1.5, color=color, alpha=0.7)

        ax.fill_between(steps, 0, 0.5 * np.exp(-steps / 100) + 0.05, alpha=0.15, color=color)
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(f"{model} Model", fontsize=11, fontweight="bold", loc="left")
        style_axis(ax)

    # Info panel
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    info = "✓ Okabe-Ito colorblind-safe palette\n✓ NeurIPS publication format\n✓ 300 training steps\n✓ 3 seeds shown (transparent overlay)"
    ax.text(0.1, 0.5, info, fontsize=10, verticalalignment="center", family="monospace")

    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.08, hspace=0.3, wspace=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ========== Page 2: Metrics Comparison ==========
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Offline Metrics by Model", "AUC, Brier, ECE aggregated across 7 benchmarks, 3 seeds")

    # Synthetic metrics
    models_list = ["1PL", "2PL", "3PL", "MIRT-D2", "MIRT-D4", "Bayes-D1", "Bayes-D2", "Bayes-D4"]
    auc_means = np.array([0.75, 0.82, 0.81, 0.79, 0.80, 0.78, 0.83, 0.84])
    auc_stds = np.array([0.05, 0.04, 0.04, 0.06, 0.05, 0.06, 0.04, 0.03])

    ax = fig.add_axes([0.08, 0.50, 0.85, 0.35])
    x_pos = np.arange(len(models_list))
    bars = ax.bar(x_pos, auc_means, yerr=auc_stds, capsize=5,
                  color=COLORS["blue"], alpha=0.7, edgecolor=DARK, linewidth=1)

    # Alternate colors for visual grouping
    for i, bar in enumerate(bars):
        if i % 2 == 1:
            bar.set_color(COLORS["orange"])
            bar.set_alpha(0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_list, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AUC (with std)", fontsize=10, fontweight="bold")
    ax.set_ylim([0.7, 0.9])
    ax.set_title("Subject Generalization Performance", fontsize=11, fontweight="bold", loc="left")
    style_axis(ax)

    # Item generalization below
    ax2 = fig.add_axes([0.08, 0.08, 0.85, 0.35])
    auc_item = auc_means * 0.75  # Typically worse on items
    auc_item_stds = auc_stds * 1.5

    bars2 = ax2.bar(x_pos, auc_item, yerr=auc_item_stds, capsize=5,
                   color=COLORS["red"], alpha=0.7, edgecolor=DARK, linewidth=1)

    for i, bar in enumerate(bars2):
        if i % 2 == 1:
            bar.set_color(COLORS["purple"])
            bar.set_alpha(0.7)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_list, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("AUC (with std)", fontsize=10, fontweight="bold")
    ax2.set_ylim([0.5, 0.75])
    ax2.set_title("Item Generalization Performance", fontsize=11, fontweight="bold", loc="left")
    style_axis(ax2)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ========== Page 3: Calibration Curves ==========
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Calibration Analysis", "Reliability diagrams (left) and ECE summary (right)")

    # Left: Reliability diagram
    ax = fig.add_axes([0.08, 0.45, 0.40, 0.45])

    # Synthetic calibration curve
    pred_probs = np.linspace(0, 1, 11)
    accs = 0.8 * pred_probs + 0.1 + np.random.RandomState(42).normal(0, 0.05, len(pred_probs))
    accs = np.clip(accs, 0, 1)

    # Plot bins
    bin_width = 0.08
    for pred, acc in zip(pred_probs, accs):
        rect = Rectangle((pred - bin_width/2, 0), bin_width, acc,
                        facecolor=COLORS["blue"], alpha=0.6, edgecolor=DARK, linewidth=1)
        ax.add_patch(rect)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.plot(pred_probs, accs, "o-", color=COLORS["orange"], linewidth=2, markersize=6, label="Empirical")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability", fontsize=10)
    ax.set_ylabel("Actual Probability", fontsize=10)
    ax.set_title("2PL Model (n≥30)", fontsize=11, fontweight="bold", loc="left")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    style_axis(ax)

    # Right: ECE summary
    ax2 = fig.add_axes([0.55, 0.45, 0.40, 0.45])

    models_ece = ["1PL", "2PL", "3PL", "MIRT", "Bayes"]
    ece_vals = [0.12, 0.08, 0.09, 0.10, 0.07]
    colors_ece = [COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["green"], COLORS["purple"]]

    bars = ax2.barh(models_ece, ece_vals, color=colors_ece, alpha=0.7, edgecolor=DARK, linewidth=1)
    ax2.set_xlabel("Expected Calibration Error", fontsize=10)
    ax2.set_title("ECE Comparison", fontsize=11, fontweight="bold", loc="left")
    ax2.set_xlim(0, 0.15)
    style_axis(ax2)

    # Bottom text
    fig.text(0.05, 0.30,
            "Lower is better. ECE measures difference between predicted and actual probabilities.\n"
            "Reliability diagram (left) shows perfect calibration (dashed line) vs empirical (colored bars).",
            fontsize=9, color=DARK, va="top")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"✓ Saved: {pdf_path}")
print("\nFeatures:")
print("  • Okabe-Ito colorblind-safe palette")
print("  • NeurIPS publication-compliant formatting")
print("  • 11×8.5in paper size (letter)")
print("  • Proper typography and spacing")
print("  • Clear, minimal design")
