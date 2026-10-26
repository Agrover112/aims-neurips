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


# ---------------------------------------------------------------------------
# CNP-style figures, after Garnelo et al. 2018 (Conditional Neural Processes).
#
# Their Figure 2 plots a 1D regression as the context set grows; their Figure 3
# shows an image completion as context/mean/variance rows against a column per
# context size, with the companion panel comparing random against
# highest-variance acquisition.
#
# The analogues here:
#   Fig 2  -> item response function for one held-out subject, context = the
#             items acquired so far. The GP upper bound becomes the oracle 2PL
#             curve fitted on that subject's complete response vector.
#   Fig 3a -> the subject x item response matrix instead of an image: observed
#             entries, posterior mean, posterior spread, per budget.
#   Fig 3b -> our acquisition arms, which is what their random-vs-variance
#             comparison already is.
#
# The posterior below is a real 2PL theta posterior on a grid, not a fitted
# network -- enough to make the figure behave correctly (budget 0 falls back to
# the prior, spread shrinks and localises as context arrives) while staying
# synthetic.
# ---------------------------------------------------------------------------

BUDGETS = [0, 1, 3, 7, 15, 31]
GRID = np.linspace(-4.0, 4.0, 201)


def _icc(a, b, theta):
    """2PL item characteristic curve, broadcast over (theta, item)."""
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def _theta_posterior(a, b, resp, items):
    """Posterior over ability on GRID given the responses to `items`, prior N(0,1)."""
    logp = -0.5 * GRID**2
    for j in items:
        p = np.clip(_icc(a[j], b[j], GRID), 1e-9, 1 - 1e-9)
        logp += np.log(p if resp[j] > 0.5 else 1.0 - p)
    w = np.exp(logp - logp.max())
    return w / w.sum()


def _predict(a, b, post):
    """Posterior predictive mean and epistemic spread per item."""
    p = _icc(a[None, :], b[None, :], GRID[:, None])  # (grid, item)
    mean = post @ p
    second = post @ (p**2)
    return mean, np.sqrt(np.clip(second - mean**2, 0, None))


def page_cnp_response_function(pdf):
    """Fig-2 analogue: the response curve tightening as the budget grows."""
    n_item = 60
    b = np.linspace(-2.5, 2.5, n_item)
    a = np.full(n_item, 1.2)
    subjects = [("Low ability ($\\theta=-0.9$)", -0.9), ("High ability ($\\theta=1.3$)", 1.3)]
    shown = [1, 3, 7, 31]

    fig, axes = plt.subplots(2, len(shown), figsize=(TEXTWIDTH, 2.9),
                             sharex=True, sharey=True)
    for row, (label, theta) in enumerate(subjects):
        truth = _icc(a, b, theta)
        resp = (rng.random(n_item) < truth).astype(float)
        order = rng.permutation(n_item)

        for col, k in enumerate(shown):
            ax = axes[row, col]
            seen = order[:k]
            mean, sd = _predict(a, b, _theta_posterior(a, b, resp, seen))

            ax.fill_between(b, mean - sd, mean + sd, color=MUTED["indigo"],
                            alpha=0.20, lw=0, zorder=2)
            ax.plot(b, mean, color=MUTED["indigo"], lw=1.1, zorder=3)
            ax.plot(b, truth, color=MUTED["rose"], lw=0.9, ls=(0, (4, 2)), zorder=4)
            ax.plot(b[seen], resp[seen], "o", ms=2.6, mfc=INK, mec=INK, zorder=5)

            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-0.08, 1.08)
            ax.set_yticks([0, 0.5, 1])
            if row == 0:
                ax.set_title(f"$k={k}$", loc="center", fontweight="bold")
            if col == 0:
                ax.set_ylabel("P(correct)")
                ax.text(0.03, 0.06, label, transform=ax.transAxes, fontsize=6,
                        color=SUBTLE, ha="left")
            if row == 1:
                ax.set_xlabel("Item difficulty $b$")
            style(ax, grid_axis="both")

    handles = [plt.Line2D([], [], color=MUTED["indigo"], lw=1.1),
               plt.Line2D([], [], color=MUTED["indigo"], lw=5, alpha=0.20),
               plt.Line2D([], [], color=MUTED["rose"], lw=0.9, ls=(0, (4, 2))),
               plt.Line2D([], [], color=INK, marker="o", ms=2.6, ls="none")]
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.7, rect=(0, 0.1, 1, 1))
    fig.legend(handles, ["Posterior mean", "$\\pm1\\sigma$", "Oracle 2PL", "Acquired item"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    pdf.savefig(fig)
    plt.close(fig)


def _max_variance_order(a, b, resp, n_item):
    """Streaming acquisition: at each step, reveal the item with the highest
    posterior-predictive variance under everything seen so far. This is the
    uncertainty-driven arm from Fig. 3b of the CNP paper, applied item-by-item
    instead of pixel-by-pixel, and it is what actually produces the context
    patterns in the row above (whereas a random permutation would not)."""
    seen: list[int] = []
    remaining = list(range(n_item))
    for _ in range(n_item):
        post = _theta_posterior(a, b, resp, seen)
        _, spread = _predict(a, b, post)
        pick = max(remaining, key=lambda j: spread[j])
        seen.append(pick)
        remaining.remove(pick)
    return np.array(seen)


def page_cnp_matrix_completion(pdf):
    """Fig-3a+3b analogue: response-matrix completion under streaming,
    uncertainty-driven acquisition, with the scored metric plotted alongside
    the reconstruction it comes from -- the two halves of their image-regression
    story, shown together instead of as separate figures."""
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import roc_auc_score

    n_subj, n_item = 20, 44
    theta = np.sort(rng.normal(0, 1.1, n_subj))
    b = np.linspace(-2.0, 2.0, n_item)
    a = rng.uniform(0.9, 1.7, n_item)
    truth = _icc(a[None, :], b[None, :], theta[:, None])
    resp = (rng.random(truth.shape) < truth).astype(float)
    # Streaming acquisition order per subject, not a random permutation --
    # this is the online-learning setting, so what gets revealed when is part
    # of the story, not a nuisance.
    orders = np.array([_max_variance_order(a, b, resp[i], n_item) for i in range(n_subj)])

    binary = ListedColormap([MUTED["rose"], MUTED["indigo"]])
    binary.set_bad(HAIRLINE)

    fig = plt.figure(figsize=(TEXTWIDTH, 4.3))
    # Explicit GridSpec extents: tight_layout cannot see the inset colorbars
    # and warns/misplaces if asked to lay this out.
    gs = GridSpec(4, len(BUDGETS), figure=fig, height_ratios=[1, 1, 1, 0.95],
                  hspace=0.32, wspace=0.12,
                  left=0.105, right=0.925, top=0.855, bottom=0.085)
    img_axes = [[fig.add_subplot(gs[r, c]) for c in range(len(BUDGETS))] for r in range(3)]
    metric_ax = fig.add_subplot(gs[3, :])

    auc_by_budget = []
    for col, k in enumerate(BUDGETS):
        context = np.full(resp.shape, np.nan)
        mean = np.empty(resp.shape)
        spread = np.empty(resp.shape)
        for i in range(n_subj):
            seen = orders[i, :k]
            context[i, seen] = resp[i, seen]
            mean[i], spread[i] = _predict(a, b, _theta_posterior(a, b, resp[i], seen))

        # Score only the held-out (not yet acquired) cells -- the quantity the
        # competition actually grades at each budget, not the whole matrix.
        held = np.ones(resp.shape, dtype=bool)
        for i in range(n_subj):
            held[i, orders[i, :k]] = False
        auc_by_budget.append(roc_auc_score(resp[held], mean[held]) if held.any() else np.nan)

        panels = [(np.ma.masked_invalid(context), binary, 0, 1),
                  (mean, SEQUENTIAL, 0, 1),
                  (spread, "magma", 0, 0.45)]
        for row, (grid, cmap, lo, hi) in enumerate(panels):
            ax = img_axes[row][col]
            im = ax.imshow(grid, cmap=cmap, vmin=lo, vmax=hi, aspect="auto",
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$k={k}$", fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(["Context", "Mean", "Spread"][row], fontweight="bold")
            if col == len(BUDGETS) - 1 and row > 0:
                # An inset cax lives outside the axes, so the panel keeps the
                # width of its neighbours; fig.colorbar(ax=...) would shrink it.
                cb = fig.colorbar(im, cax=ax.inset_axes([1.07, 0.0, 0.08, 1.0]))
                cb.ax.tick_params(labelsize=5, width=0.4, length=1.6, pad=1.5)
                cb.outline.set_linewidth(0.4)
            style(ax, grid_axis="none")

    metric_ax.plot(range(len(BUDGETS)), auc_by_budget, color=MUTED["wine"],
                   marker="o", ms=3, lw=1.1)
    metric_ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
    metric_ax.set_xticks(range(len(BUDGETS)))
    metric_ax.set_xticklabels(BUDGETS)
    metric_ax.set_xlabel("Acquisition budget $k$ (items revealed per subject)")
    metric_ax.set_ylabel("AUC, held-out cells")
    metric_ax.set_ylim(0.45, 1.0)
    style(metric_ax, grid_axis="both")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=MUTED["indigo"]),
               plt.Rectangle((0, 0), 1, 1, fc=MUTED["rose"]),
               plt.Rectangle((0, 0), 1, 1, fc=HAIRLINE)]
    fig.suptitle("Online response-matrix completion", x=0.02, ha="left", y=0.985,
                 fontsize=9, fontweight="bold")
    fig.legend(handles, ["correct", "incorrect", "not yet acquired"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=6.5)
    fig.text(0.5, 0.902, "Rows: 20 subjects.  Columns within a panel: 44 items, "
             "hard $\\rightarrow$ easy.  Acquisition: max posterior spread (streaming).",
             ha="center", fontsize=6, color=SUBTLE)
    pdf.savefig(fig)
    plt.close(fig)


def page_cnp_acquisition(pdf):
    """Fig-3b analogue: random against uncertainty-driven acquisition."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.0))
    x = np.arange(len(BUDGETS))
    arms = {
        "Random": (np.array([0.612, 0.648, 0.698, 0.742, 0.778, 0.802]), MUTED["indigo"], "-"),
        "Max spread": (np.array([0.612, 0.671, 0.736, 0.787, 0.818, 0.835]), MUTED["rose"], "--"),
        "Fisher": (np.array([0.612, 0.679, 0.748, 0.799, 0.829, 0.844]), MUTED["teal"], "-."),
    }

    ax = axes[0]
    for name, (vals, colour, ls) in arms.items():
        err = 0.035 * np.exp(-x / 3.5) + 0.008
        ax.fill_between(x, vals - err, vals + err, color=colour, alpha=0.15, lw=0)
        ax.plot(x, vals, color=colour, ls=ls, marker="o", ms=2.8, label=name)
    ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(BUDGETS)
    ax.set_xlabel("Acquisition budget $k$")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 0.9)
    ax.set_title("(a) Accuracy against budget", loc="left", fontweight="bold")
    ax.legend(loc="upper left", handlelength=1.8)
    style(ax, grid_axis="both")

    ax = axes[1]
    for name, (_, colour, ls) in arms.items():
        decay = {"Random": 3.9, "Max spread": 2.5, "Fisher": 2.3}[name]
        ax.plot(x, 0.42 * np.exp(-x / decay) + 0.055, color=colour, ls=ls,
                marker="o", ms=2.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(BUDGETS)
    ax.set_xlabel("Acquisition budget $k$")
    ax.set_ylabel("Mean posterior spread")
    ax.set_ylim(0, 0.5)
    ax.set_title("(b) Uncertainty against budget", loc="left", fontweight="bold")
    style(ax, grid_axis="both")

    fig.tight_layout(pad=0.3, w_pad=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        page_palette(pdf)
        page_grouped_bars(pdf)
        page_curves(pdf)
        page_calibration(pdf)
        page_matrix(pdf)
        page_cnp_response_function(pdf)
        page_cnp_matrix_completion(pdf)
        page_cnp_acquisition(pdf)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
