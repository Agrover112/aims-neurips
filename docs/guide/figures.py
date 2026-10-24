"""Generate the figures used in the AIMS field guide.

Run from this directory:  python figures.py
Writes SVG (for the PDF) and PNG (for quick inspection) into figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

INK = "#12241e"
GREEN = "#1d6b50"
GREEN_LIGHT = "#8fb8a7"
AMBER = "#b9771f"
RED = "#a33a2b"
MUTED = "#6d7a74"
GRID = "#ddd8cb"

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "Ubuntu",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#c9c3b4",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "svg.fonttype": "path",
        "lines.solid_capstyle": "round",
    }
)


def save(fig: mpl.figure.Figure, name: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", transparent=True, dpi=220)
    plt.close(fig)
    print(f"wrote {name}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def legend_below(ax, ncol=1, y=-0.34):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.8,
              columnspacing=1.6, handletextpad=0.6)


def fig_sigmoid() -> None:
    z = np.linspace(-6, 6, 500)
    p = sigmoid(z)
    fig, ax = plt.subplots(figsize=(6.5, 2.35))
    ax.axhline(0.5, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    ax.plot(z, p, color=GREEN, lw=2.0, zorder=3)
    for zz, label, dx, dy, ha in [
        (2, "0.88", -0.25, 0.05, "right"),
        (-2, "0.12", 0.25, 0.05, "left"),
    ]:
        ax.scatter([zz], [sigmoid(zz)], s=24, color=AMBER, zorder=4, edgecolor="white", linewidth=1)
        ax.annotate(label, (zz, sigmoid(zz)), xytext=(zz + dx, sigmoid(zz) + dy),
                    fontsize=8.2, color=AMBER, ha=ha, va="center")
    ax.text(-5.6, 0.16, "flat tail", fontsize=7.2, color=MUTED)
    ax.text(3.55, 0.90, "saturates", fontsize=7.2, color=MUTED, ha="center")
    ax.text(0.58, 0.34, "steepest at $z=0$", fontsize=7.2, color=MUTED)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("logit  $z$")
    ax.set_ylabel("probability  $\\sigma(z)$")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y")
    save(fig, "sigmoid")


def fig_nll() -> None:
    p = np.linspace(0.005, 0.995, 900)
    fig, ax = plt.subplots(figsize=(6.5, 2.55))
    ax.plot(p, -np.log(p), color=GREEN, lw=1.9)
    ax.plot(p, -np.log(1 - p), color=AMBER, lw=1.9)
    ax.text(0.26, 1.72, "$y=1$:  $-\\ln p$", color=GREEN, fontsize=8.4)
    ax.text(0.78, 1.15, "$y=0$:  $-\\ln(1-p)$", color=AMBER, fontsize=8.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.8)
    ax.set_xlabel("predicted probability  $p$")
    ax.set_ylabel("loss  (nats)")
    ax.grid(axis="y")
    ax.annotate("confident and wrong\nis punished without limit",
                xy=(0.022, 4.25), xytext=(0.20, 3.9), fontsize=7.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8,
                                connectionstyle="arc3,rad=-0.25"), va="top")
    ax.annotate("0.105", xy=(0.9, 0.105), xytext=(0.90, 0.42), fontsize=7.5, color=GREEN,
                ha="center")
    ax.annotate("2.303", xy=(0.1, 2.303), xytext=(0.155, 2.42), fontsize=7.5, color=AMBER,
                ha="left")
    save(fig, "nll")


def fig_beta() -> None:
    x = np.linspace(0, 1, 600)
    from math import lgamma

    def beta_pdf(x, a, b):
        norm = np.exp(lgamma(a + b) - lgamma(a) - lgamma(b))
        return norm * x ** (a - 1) * (1 - x) ** (b - 1)

    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.plot(x, beta_pdf(x, 2, 2), color=MUTED, lw=1.4, ls="--", label="prior  Beta(2, 2)")
    ax.fill_between(x, beta_pdf(x, 2, 2), color=MUTED, alpha=0.07)
    ax.plot(x, beta_pdf(x, 5, 3), color=GREEN, lw=2.0, label="posterior  Beta(5, 3)")
    ax.fill_between(x, beta_pdf(x, 5, 3), color=GREEN, alpha=0.12)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.axvline(5 / 8, color=AMBER, lw=1.0, ls=":")
    ax.text(0.49, 2.42, "mean 0.50", fontsize=7.4, color=MUTED, ha="right")
    ax.text(0.635, 2.42, "mean 0.625", fontsize=7.4, color=AMBER, ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.55)
    ax.set_xlabel("success probability  $\\pi$")
    ax.set_ylabel("density")
    ax.set_yticks([])
    legend_below(ax, ncol=2, y=-0.36)
    ax.grid(axis="x")
    save(fig, "beta")


def fig_irt() -> None:
    theta = np.linspace(-4, 4, 500)
    fig, ax = plt.subplots(figsize=(6.3, 2.75))
    ax.plot(theta, sigmoid(1.0 * (theta - 0)), color=GREEN, lw=2.0, label="$a=1,\\ b=0$")
    ax.plot(theta, sigmoid(2.0 * (theta - 0)), color=AMBER, lw=2.0, label="$a=2,\\ b=0$")
    ax.plot(theta, sigmoid(1.0 * (theta - 1)), color=RED, lw=1.6, ls="--", label="$a=1,\\ b=1$")
    ax.axhline(0.5, color=GRID, lw=0.7)
    ax.scatter([0, 1], [0.5, 0.5], s=22, color=[GREEN, RED], zorder=4, edgecolor="white")
    ax.annotate("$b$ shifts the curve\nright = harder", xy=(1.04, 0.50), xytext=(2.05, 0.15),
                fontsize=7.4, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("$a$ controls steepness\n(discrimination)", xy=(-1.55, 0.29), xytext=(-3.9, 0.70),
                fontsize=7.4, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("subject ability  $\\theta$")
    ax.set_ylabel("P(success)")
    ax.set_yticks([0, 0.5, 1])
    legend_below(ax, ncol=3, y=-0.38)
    ax.grid(axis="y")
    save(fig, "irt")


def fig_shrink() -> None:
    theta = np.linspace(-6, 6, 900)
    prior = np.exp(-(theta ** 2) / (2 * 4.0)) / np.sqrt(2 * np.pi * 4.0)
    likelihood = sigmoid(theta)
    post = prior * likelihood
    post /= np.trapezoid(post, theta)
    fig, ax = plt.subplots(figsize=(6.4, 2.65))
    ax.plot(theta, prior, color=MUTED, lw=1.3, ls="--", label="prior  $\\mathcal{N}(0, 4)$")
    ax.plot(theta, post, color=GREEN, lw=2.0, label="posterior after one $y=1$")
    ax.fill_between(theta, post, color=GREEN, alpha=0.11)
    ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    ax.axvline(theta[np.argmax(post)], color=AMBER, lw=1.0, ls=":")
    ax.annotate("MAP 1.04", xy=(1.06, 0.345), xytext=(1.25, 0.385), fontsize=7.6, color=AMBER)
    ax.text(3.35, 0.40, "MLE: $\\theta\\to\\infty$  (no prior)", fontsize=7.4, color=RED)
    ax.set_xlim(-4, 6)
    ax.set_ylim(0, 0.43)
    ax.set_xlabel("ability  $\\theta$")
    ax.set_ylabel("density")
    ax.set_yticks([])
    legend_below(ax, ncol=2, y=-0.34)
    save(fig, "shrink")


def fig_alc() -> None:
    budgets = np.array([0, 1, 3, 7, 15, 31])
    x = np.arange(len(budgets))
    model = np.array([0.250, 0.243, 0.231, 0.218, 0.205, 0.199])
    base = np.full(len(x), 0.25)
    weights = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.fill_between(x, model, 0.25, color=GREEN, alpha=0.09)
    ax.plot(x, base, color=MUTED, lw=1.1, ls="--", label="constant 0.5 predictor  (ALC 0.250)")
    ax.plot(x, model, color=GREEN, lw=2.1, marker="o", ms=4.2,
            label="adaptive model  (ALC 0.224)")
    for xi, yi in zip(x, model):
        ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7.1, color=GREEN)
    ax.set_xticks(x, [f"{b}\nw={w:.1f}" for b, w in zip(budgets, weights)])
    ax.set_xlabel("label budget  $B$")
    ax.set_ylabel("Brier score")
    ax.set_ylim(0.185, 0.258)
    ax.grid(axis="y")
    legend_below(ax, ncol=2, y=-0.36)
    save(fig, "alc")


def fig_attention() -> None:
    scores = np.array([0.0, 1.0, 2.0])
    weights = np.exp(scores) / np.exp(scores).sum()
    values = np.array([1.0, 3.0, 2.0])
    fig, ax = plt.subplots(figsize=(6.0, 2.45))
    ax.bar(np.arange(3), weights, color=[GREEN_LIGHT, GREEN, AMBER], width=0.55)
    for i, (w, v) in enumerate(zip(weights, values)):
        ax.annotate(f"$\\alpha_{i + 1}$ = {w:.3f}", (i, w), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.text(i, 0.02, f"value {v:.0f}", ha="center", fontsize=7.4,
                color=INK if i == 0 else "white")
    ax.text(-0.45, 0.86, "$r^* = \\sum_i \\alpha_i v_i = 0.090(1) + 0.245(3) + 0.665(2) = 2.155$",
            fontsize=8.6, color=AMBER)
    ax.set_xticks(np.arange(3), ["context 1\nscore 0", "context 2\nscore 1", "context 3\nscore 2"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("attention weight")
    ax.grid(axis="y")
    save(fig, "attention")


if __name__ == "__main__":
    fig_sigmoid()
    fig_nll()
    fig_beta()
    fig_irt()
    fig_shrink()
    fig_alc()
    fig_attention()
