"""CNP-style figures for the response-prediction setting.

After Garnelo et al. 2018, *Conditional Neural Processes*. Their figures map
onto this problem closely enough to be worth reusing directly:

    their Fig 2   1D regression as the context set grows
                  -> item response function for one subject as the acquisition
                     budget grows; their GP upper bound becomes the oracle 2PL
                     fitted on that subject's complete response vector.

    their Fig 3a  MNIST context / mean / variance against context size
                  -> the subject x item response matrix instead of an image.
                     A response is a pixel and acquisition reveals pixels.

    their Fig 3b  random against highest-variance pixel selection
                  -> our acquisition arms, which is the same comparison.

The one structural difference worth keeping in mind: MNIST has a single image
to reconstruct, whereas here every subject is their own image and they share a
global column structure (item difficulty). That shared structure is the learned
prior, and it is why the k=0 column is not blank -- with no labels at all the
prediction is already the population response curve, which is the analogue of
their "condition on one uninformative pixel, get the average digit".

The posterior here is an exact 2PL theta posterior on a grid, not an amortized
network. It is a reference point rather than a model: a trained CNP should beat
it, since it has to infer item structure that this is simply handed. Pass real
fitted (a, b) to drive these against actual results.

Run directly for a synthetic demo:

    python cnp_figures.py out.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from plotstyle import (  # noqa: E402
    HAIRLINE, INK, MUTED, SEQUENTIAL, SUBTLE, TEXTWIDTH, style,
)

# The competition's budget ladder. Nested, so each is a superset of the last.
BUDGETS = [0, 1, 3, 7, 15, 31]

# Ability grid the posterior is evaluated on. +-4 covers the N(0,1) prior to
# well past any ability the data can identify.
GRID = np.linspace(-4.0, 4.0, 201)


def icc(a, b, theta):
    """2PL item characteristic curve, broadcasting over (theta, item)."""
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def theta_posterior(a, b, resp, items):
    """Posterior over ability on GRID given responses to `items`, prior N(0,1)."""
    logp = -0.5 * GRID**2
    for j in items:
        p = np.clip(icc(a[j], b[j], GRID), 1e-9, 1 - 1e-9)
        logp += np.log(p if resp[j] > 0.5 else 1.0 - p)
    w = np.exp(logp - logp.max())
    return w / w.sum()


def predict(a, b, post):
    """Posterior predictive mean and epistemic spread per item.

    The spread is the std of p over the ability posterior, i.e. the part of the
    uncertainty that acquiring labels can remove. Bernoulli noise at fixed
    ability is irreducible and deliberately excluded.
    """
    p = icc(a[None, :], b[None, :], GRID[:, None])  # (grid, item)
    mean = post @ p
    second = post @ (p**2)
    return mean, np.sqrt(np.clip(second - mean**2, 0, None))


def max_spread_order(a, b, resp):
    """Streaming acquisition order: repeatedly take the item whose predictive
    spread is largest under everything seen so far.

    This is the uncertainty-driven arm of their Fig 3b, item-by-item rather
    than pixel-by-pixel. Note this ranks over the remaining pool, so it is an
    upper bound on what a true streaming policy can do -- deployment has to
    decide on each item as it arrives, without seeing the rest.
    """
    n_item = len(b)
    seen: list[int] = []
    remaining = list(range(n_item))
    for _ in range(n_item):
        _, spread = predict(a, b, theta_posterior(a, b, resp, seen))
        pick = max(remaining, key=lambda j: spread[j])
        seen.append(pick)
        remaining.remove(pick)
    return np.array(seen)


def synthetic_world(rng, n_subj=20, n_item=44):
    """A 2PL world to demo against when real fitted parameters are not to hand."""
    theta = np.sort(rng.normal(0, 1.1, n_subj))
    b = np.linspace(-2.0, 2.0, n_item)
    a = rng.uniform(0.9, 1.7, n_item)
    resp = (rng.random((n_subj, n_item)) < icc(a[None, :], b[None, :], theta[:, None]))
    return a, b, resp.astype(float), theta


def page_response_function(pdf, a=None, b=None, subjects=None, rng=None,
                           shown=(1, 3, 7, 31)):
    """Fig-2 analogue: the response curve tightening as the budget grows.

    `subjects` is a list of (label, theta); with real data, pass fitted
    abilities and the (a, b) they were fitted against.
    """
    rng = rng or np.random.default_rng(0)
    if a is None or b is None:
        n_item = 60
        b = np.linspace(-2.5, 2.5, n_item)
        a = np.full(n_item, 1.2)
    if subjects is None:
        subjects = [("Low ability ($\\theta=-0.9$)", -0.9),
                    ("High ability ($\\theta=1.3$)", 1.3)]

    fig, axes = plt.subplots(len(subjects), len(shown), figsize=(TEXTWIDTH, 2.9),
                             sharex=True, sharey=True, squeeze=False)
    for row, (label, theta) in enumerate(subjects):
        truth = icc(a, b, theta)
        resp = (rng.random(len(b)) < truth).astype(float)
        order = rng.permutation(len(b))

        for col, k in enumerate(shown):
            ax = axes[row][col]
            seen = order[:k]
            mean, sd = predict(a, b, theta_posterior(a, b, resp, seen))

            ax.fill_between(b, mean - sd, mean + sd, color=MUTED["indigo"],
                            alpha=0.20, lw=0, zorder=2)
            ax.plot(b, mean, color=MUTED["indigo"], lw=1.1, zorder=3)
            ax.plot(b, truth, color=MUTED["rose"], lw=0.9, ls=(0, (4, 2)), zorder=4)
            ax.plot(b[seen], resp[seen], "o", ms=2.6, mfc=INK, mec=INK, zorder=5)

            ax.set_xlim(b.min() - 0.1, b.max() + 0.1)
            ax.set_ylim(-0.08, 1.08)
            ax.set_yticks([0, 0.5, 1])
            if row == 0:
                ax.set_title(f"$k={k}$", loc="center", fontweight="bold")
            if col == 0:
                ax.set_ylabel("P(correct)")
                ax.text(0.03, 0.06, label, transform=ax.transAxes, fontsize=6,
                        color=SUBTLE, ha="left")
            if row == len(subjects) - 1:
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


def page_matrix_completion(pdf, a=None, b=None, resp=None, orders=None, rng=None):
    """Fig-3a and 3b fused: response-matrix completion under streaming
    uncertainty-driven acquisition, with the graded metric beneath the
    reconstruction it is computed from."""
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import roc_auc_score

    rng = rng or np.random.default_rng(0)
    if a is None or b is None or resp is None:
        a, b, resp, _ = synthetic_world(rng)
    n_subj, n_item = resp.shape
    if orders is None:
        orders = np.array([max_spread_order(a, b, resp[i]) for i in range(n_subj)])

    binary = ListedColormap([MUTED["rose"], MUTED["indigo"]])
    binary.set_bad(HAIRLINE)

    fig = plt.figure(figsize=(TEXTWIDTH, 4.3))
    # Explicit GridSpec extents: tight_layout cannot see the inset colorbars
    # and warns and misplaces panels if asked to lay this out.
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
            mean[i], spread[i] = predict(a, b, theta_posterior(a, b, resp[i], seen))

        # Score only the not-yet-acquired cells, which is what the competition
        # grades at each budget -- scoring the whole matrix would credit the
        # model for labels it was handed.
        held = np.ones(resp.shape, dtype=bool)
        for i in range(n_subj):
            held[i, orders[i, :k]] = False
        auc_by_budget.append(
            roc_auc_score(resp[held], mean[held])
            if held.any() and len(np.unique(resp[held])) > 1 else np.nan
        )

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
    fig.text(0.5, 0.902,
             f"Rows: {n_subj} subjects.  Columns within a panel: {n_item} items, "
             "hard $\\rightarrow$ easy.  Acquisition: max posterior spread (streaming).",
             ha="center", fontsize=6, color=SUBTLE)
    pdf.savefig(fig)
    plt.close(fig)


def page_acquisition(pdf, arms=None, spreads=None):
    """Fig-3b analogue: random against uncertainty-driven acquisition.

    `arms` maps arm name -> AUC per budget; `spreads` maps arm name -> mean
    posterior spread per budget. Both default to illustrative numbers.
    """
    if arms is None:
        arms = {"Random": np.array([0.612, 0.648, 0.698, 0.742, 0.778, 0.802]),
                "Max spread": np.array([0.612, 0.671, 0.736, 0.787, 0.818, 0.835]),
                "Fisher": np.array([0.612, 0.679, 0.748, 0.799, 0.829, 0.844])}
    x = np.arange(len(BUDGETS))
    if spreads is None:
        spreads = {name: 0.42 * np.exp(-x / d) + 0.055
                   for name, d in (("Random", 3.9), ("Max spread", 2.5), ("Fisher", 2.3))}

    styles = [(MUTED["indigo"], "-"), (MUTED["rose"], "--"), (MUTED["teal"], "-."),
              (MUTED["olive"], ":")]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.0))

    ax = axes[0]
    for i, (name, vals) in enumerate(arms.items()):
        colour, ls = styles[i % len(styles)]
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
    for i, (name, vals) in enumerate(spreads.items()):
        colour, ls = styles[i % len(styles)]
        ax.plot(x, vals, color=colour, ls=ls, marker="o", ms=2.8, label=name)
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
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cnp-figures.pdf")
    rng = np.random.default_rng(0)
    with PdfPages(out) as pdf:
        page_response_function(pdf, rng=rng)
        page_matrix_completion(pdf, rng=rng)
        page_acquisition(pdf)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
