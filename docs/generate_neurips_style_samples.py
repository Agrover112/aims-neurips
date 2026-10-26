"""Render docs/neurips-sample-plots.pdf: the house style, on fake data.

Every figure here uses organized/utils/plotstyle.py, so this file doubles as a
worked example of how to use it. The numbers are synthetic -- the point is the
styling, not the result.

    python3 docs/generate_neurips_style_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "organized" / "utils"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from cnp_figures import (  # noqa: E402
    page_acquisition, page_matrix_completion, page_response_function,
)
from plotstyle import (  # noqa: E402
    BRIGHT, MUTED, HIGH_CONTRAST, CYCLE, INK, SUBTLE, HAIRLINE,
    TEXTWIDTH, COLWIDTH, SEQUENTIAL, style, errorbar_kw,
)

OUT = ROOT / "docs" / "neurips-sample-plots.pdf"
MODELS = ["1PL", "2PL", "3PL", "MIRT\nD=2", "MIRT\nD=4", "Bayes\nD=1", "Bayes\nD=2", "Bayes\nD=4"]
rng = np.random.default_rng(0)


def page_palette(pdf):
    fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH, 2.6))
    for ax, (name, pal) in zip(axes, [("Tol muted (default cycle)", MUTED),
                                      ("Tol bright", BRIGHT),
                                      ("Tol high-contrast", HIGH_CONTRAST)]):
        for i, (key, hexval) in enumerate(pal.items()):
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=hexval))
            ax.text(i + 0.46, -0.22, key, ha="center", va="top", fontsize=6)
            ax.text(i + 0.46, -0.52, hexval, ha="center", va="top", fontsize=5, color=SUBTLE)
        ax.set_xlim(-0.1, max(len(p) for p in (MUTED, BRIGHT, HIGH_CONTRAST)))
        ax.set_ylim(-0.8, 1.05)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    pdf.savefig(fig)
    plt.close(fig)


def page_grouped_bars(pdf):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.0))
    subj = np.array([0.748, 0.823, 0.816, 0.789, 0.801, 0.778, 0.831, 0.842])
    subj_e = np.array([0.048, 0.036, 0.041, 0.053, 0.047, 0.058, 0.039, 0.034])
    item, item_e = subj * 0.72, subj_e * 1.4

    x = np.arange(len(MODELS))
    w = 0.38
    ax.bar(x - w / 2, subj, w, yerr=subj_e, error_kw=errorbar_kw(),
           label="Subject holdout", color=MUTED["indigo"], edgecolor=INK, lw=0.4)
    ax.bar(x + w / 2, item, w, yerr=item_e, error_kw=errorbar_kw(),
           label="Item holdout", color=MUTED["sand"], edgecolor=INK, lw=0.4)

    ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.text(-0.55, 0.52, "chance", fontsize=6, color=SUBTLE, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.65, len(MODELS) - 0.35)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.5)
    style(ax)
    fig.tight_layout(pad=0.3)
    pdf.savefig(fig)
    plt.close(fig)


def page_curves(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9), sharey=True)
    steps = np.arange(300)
    curves = {"1PL": (0.42, 95, MUTED["indigo"], "-"),
              "2PL": (0.46, 70, MUTED["rose"], "--"),
              "3PL": (0.47, 60, MUTED["teal"], "-.")}
    for ax, (title, floor) in zip(axes, [("Train", 0.085), ("Validation", 0.105)]):
        for name, (amp, tau, colour, ls) in curves.items():
            base = amp * np.exp(-steps / tau) + floor
            ax.fill_between(steps, base * 0.96, base * 1.04, color=colour, alpha=0.18, lw=0)
            ax.plot(steps, base, color=colour, ls=ls, label=name)
        ax.set_xlabel("Step")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(0, 300)
        ax.set_ylim(0.05, 0.55)
        style(ax, grid_axis="both")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[1].legend(loc="upper right", handlelength=1.6)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    pdf.savefig(fig)
    plt.close(fig)


def page_calibration(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1))

    ax = axes[0]
    edges = np.linspace(0, 1, 11)
    centres = (edges[:-1] + edges[1:]) / 2
    acc = np.clip(0.84 * centres + 0.07 + rng.normal(0, 0.035, 10), 0, 1)
    ax.bar(centres, acc, width=0.092, color=MUTED["cyan"], edgecolor=INK, lw=0.4,
           label="Observed", zorder=2)
    ax.bar(centres, centres - acc, width=0.092, bottom=acc, color=MUTED["rose"],
           alpha=0.45, edgecolor=MUTED["wine"], lw=0.4, hatch="////", label="Gap", zorder=2)
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=(0, (4, 2)), zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("(a) Reliability, 2PL", loc="left", fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", handlelength=1.1)
    style(ax, grid_axis="both")

    ax = axes[1]
    names = ["1PL", "2PL", "3PL", "MIRT D=4", "Bayes D=4"]
    ece = np.array([0.119, 0.075, 0.088, 0.102, 0.064])
    err = np.array([0.011, 0.008, 0.009, 0.013, 0.007])
    y = np.arange(len(names))[::-1]
    ax.hlines(y, 0, ece, color=HAIRLINE, lw=0.8, zorder=1)
    ax.errorbar(ece, y, xerr=err, fmt="o", color=MUTED["wine"], zorder=3, **errorbar_kw())
    best = int(np.argmin(ece))
    ax.plot(ece[best], y[best], "o", ms=5.5, mfc="none", mec=MUTED["green"], mew=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 0.14)
    ax.set_xlabel("Expected calibration error")
    ax.set_title("(b) ECE (lower is better)", loc="left", fontweight="bold")
    style(ax, grid_axis="x")

    fig.tight_layout(pad=0.3, w_pad=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def page_matrix(pdf):
    """Response matrix before/after filtering -- the one figure that needs a colourmap."""
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2))
    cmap = ListedColormap([MUTED["rose"], MUTED["indigo"]])

    raw = rng.random((90, 220))
    difficulty = np.linspace(0.15, 0.9, 220)
    ability = np.linspace(0.2, 0.95, 90)[:, None]
    mat = (raw < (difficulty[None, :] * ability)).astype(float)
    sparse = mat.copy()
    sparse[rng.random(mat.shape) < 0.62] = np.nan

    # Filtering drops columns across the whole difficulty range, so subsample
    # evenly -- slicing a contiguous block would fake a shift in pass rate.
    kept = mat[:, ::2]

    for ax, (grid, label) in zip(axes, [(sparse, "(a) Raw"), (kept, "(b) After $n\\geq30$ filter")]):
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        filled = np.isfinite(grid).mean()
        ax.set_title(f"{label}", loc="left", fontweight="bold")
        ax.set_xlabel(f"{grid.shape[1]} items (hard $\\rightarrow$ easy)")
        ax.set_ylabel(f"{grid.shape[0]} subjects")
        ax.text(0.99, 1.02, f"observed {filled:.0%}", transform=ax.transAxes,
                ha="right", fontsize=6, color=SUBTLE)
        ax.set_xticks([])
        ax.set_yticks([])
        style(ax, grid_axis="none")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=MUTED["indigo"]),
               plt.Rectangle((0, 0), 1, 1, fc=MUTED["rose"]),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec=HAIRLINE)]
    # Reserve the bottom strip first, then drop the legend into it, so it never
    # lands on top of the x labels.
    fig.tight_layout(pad=0.3, w_pad=1.4, rect=(0, 0.13, 1, 1))
    fig.legend(handles, ["correct", "incorrect", "unobserved"], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        page_palette(pdf)
        page_grouped_bars(pdf)
        page_curves(pdf)
        page_calibration(pdf)
        page_matrix(pdf)
        page_response_function(pdf, rng=rng)
        page_matrix_completion(pdf, rng=rng)
        page_acquisition(pdf)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
