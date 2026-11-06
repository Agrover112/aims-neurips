#!/usr/bin/env python3
"""Two paper figures that replace tables: effect sizes against the Beta-Binomial, and item-difficulty recovery.

The numbers are the ones printed in the paper's tables (single source: docs/paper-draft.md and
paper/neurips_2026), copied here with their 95% bootstrap CIs; nothing is recomputed.

    python organized/utils/plot_paper_effects.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from plotstyle import MUTED, INK, SUBTLE, TEXTWIDTH, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "neurips_2026" / "figures" / "paper"

C = {"anp": MUTED["indigo"], "rasch": MUTED["teal"], "kernel": MUTED["wine"], "const": MUTED["olive"], "probe": MUTED["rose"]}

# (label, colour, estimate, lo, hi): test Brier ALC minus Beta-Binomial; negative is better.
PANELS = [
    ("(a) Single-pair", [
        ("Constant (training mean)", C["const"], -0.0022, -0.0105, 0.0067),
        ("ANP", C["anp"], 0.0230, 0.0198, 0.0267),
        ("ANP + SAM", C["anp"], 0.0168, 0.0128, 0.0211),
        ("ANP, prefix training", C["anp"], 0.0267, 0.0231, 0.0306),
    ]),
    ("(b) Multi-pair", [
        ("Pooled Rasch without item difficulty", C["rasch"], -0.0020, -0.0029, -0.0011),
        ("Pooled Rasch", C["rasch"], -0.0035, -0.0046, -0.0026),
        ("Clustered Rasch", C["kernel"], -0.0034, -0.0045, -0.0024),
        ("Kernel Rasch", C["kernel"], -0.0049, -0.0060, -0.0038),
    ]),
    ("(c) Multi-pair, one subject per benchmark", [
        ("Pooled Rasch (re-tuned priors)", C["rasch"], -0.0032, -0.0035, -0.0029),
        ("Kernel Rasch", C["kernel"], -0.0024, -0.0027, -0.0021),
    ]),
]


def effects() -> plt.Figure:
    rows_n = [len(r) for _, r in PANELS]
    fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH, 3.1), gridspec_kw={"height_ratios": [n + 0.9 for n in rows_n]})
    fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.10, hspace=0.75)
    for ax, (title, rows) in zip(axes, PANELS):
        ys = list(range(len(rows)))[::-1]
        for y, (lab, col, est, lo, hi) in zip(ys, rows):
            ax.plot([lo, hi], [y, y], color=col, lw=1.7, solid_capstyle="round")
            ax.plot([est], [y], marker="o", ms=4.4, color=col, markeredgecolor=INK, markeredgewidth=0.5)
        ax.axvline(0, color=INK, lw=0.7, ls=(0, (4, 2)))
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=6.8, pad=2, loc="left")
        ax.tick_params(labelsize=5.8)
        style(ax, grid_axis="x")
        lo_all = min(r[3] for r in rows); hi_all = max(r[4] for r in rows)
        span = hi_all - lo_all
        ax.set_xlim(min(lo_all - 0.12 * span, -0.25 * span), max(hi_all + 0.12 * span, 0.25 * span))
        ax.set_xlabel("test Brier ALC minus Beta-Binomial (95% CI over tasks)", fontsize=5.8, labelpad=1.5)
    handles = [Line2D([], [], color=C["const"], lw=2, label="Constant"),
               Line2D([], [], color=C["anp"], lw=2, label="ANP variants"),
               Line2D([], [], color=C["rasch"], lw=2, label="Pooled Rasch"),
               Line2D([], [], color=C["kernel"], lw=2, label="Clustered / kernel Rasch"),
               Line2D([], [], color=INK, lw=0.7, ls=(0, (4, 2)), label="Beta-Binomial (0); left of it is better")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=5.6, frameon=False, bbox_to_anchor=(0.55, 0.995))
    return fig


# (label, ID rho, OOD rho)
DIFFICULTY = [
    ("ANP", 0.10, 0.00), ("ANP + SAM", 0.13, 0.01), ("ANP, prefix training", 0.12, 0.04),
    ("ANP, random-subject split", 0.48, 0.03), ("Probe (item embeddings)", 0.40, 0.09),
]


def difficulty() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.8, 1.85))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.82, bottom=0.2)
    ys = list(range(len(DIFFICULTY)))[::-1]
    for y, (lab, idv, ood) in zip(ys, DIFFICULTY):
        col = C["probe"] if lab.startswith("Probe") else C["anp"]
        ax.plot([ood, idv], [y, y], color=col, lw=1.4, alpha=0.7, zorder=1)
        ax.plot([idv], [y], marker="o", ms=5, color=col, markeredgecolor=INK, markeredgewidth=0.5, zorder=3)
        ax.plot([ood], [y], marker="o", ms=5, color="white", markeredgecolor=col, markeredgewidth=1.4, zorder=3)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in DIFFICULTY], fontsize=6.2)
    ax.set_ylim(-0.6, len(DIFFICULTY) - 0.4)
    ax.set_xlim(-0.05, 0.55)
    ax.set_xlabel("Spearman rank correlation with observed item pass rate (0 = chance)", fontsize=6.2)
    ax.tick_params(labelsize=6)
    style(ax, grid_axis="x")
    handles = [Line2D([], [], marker="o", ls="none", ms=5, color=SUBTLE, markeredgecolor=INK, markeredgewidth=0.5, label="ID (benchmark seen in training)"),
               Line2D([], [], marker="o", ls="none", ms=5, markerfacecolor="white", markeredgecolor=SUBTLE, markeredgewidth=1.4, label="OOD (unseen benchmark)"),
               Line2D([], [], color=INK, lw=0.7, ls=(0, (4, 2)), label="chance")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=5.8, frameon=False, bbox_to_anchor=(0.55, 0.99))
    return fig


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fig in (("effects", effects()), ("difficulty-id-ood", difficulty())):
        fig.savefig(OUT / f"{name}.pdf")
        fig.savefig(Path.home() / "aims-replay-cache" / f"{name}.png", dpi=170)
        print("wrote", OUT / f"{name}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
