"""Render the R8 exact split sensitivity and 17-benchmark diagnostic figures."""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from wp9 import CYCLE, TEXTWIDTH, plt, save_figure, style


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    d = pd.read_csv(args.results / "wp13-split-sensitivity.csv")
    x = d.refit_macro_delta
    orig = x.loc[d.original_test_triple].item()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.65))
    ax.hist(x, bins=24, color=CYCLE[0], alpha=.72)
    ax.axvline(0, color=".45", lw=1, ls=":", label="No change")
    ax.axvline(orig, color=CYCLE[1], lw=1.6, label="Original test benchmarks (refit)")
    ax.set(xlabel="Fitted Beta–Binomial minus Jeffreys (macro Brier ALC)", ylabel="Test triples")
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2)
    ax.text(.98, .91, f"{(x > 0).mean():.1%} of triples worsen", transform=ax.transAxes,
            ha="right", va="top")
    fig.subplots_adjust(left=.11, right=.98, top=.82, bottom=.21)
    save_figure(fig, args.results / "figure5-split-sensitivity")

    d = pd.read_csv(args.results / "wp13-benchmark-explanation.csv")
    d["source"] = d.benchmark.map(dict.fromkeys(["algotune", "cmmlu", "editbench"], "Test")).fillna(
        d.benchmark.map(dict.fromkeys(["ai2d_test", "bird_sql", "financebench"], "Validation"))).fillna("LOBO")
    x = "mean_abs_query_acc_minus_0.52"
    rho = spearmanr(d[x], d.delta).statistic
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.85))
    ax.axhline(0, color=".55", lw=.8)
    for j, (src, grp) in enumerate(d.groupby("source")):
        ax.errorbar(grp[x], grp.delta, yerr=[grp.delta-grp.ci_low, grp.ci_high-grp.delta],
                    fmt=["o", "^", "s"][j], color=CYCLE[j], capsize=2, label=src)
    ax.set(xlabel="Mean absolute distance of evaluation accuracy from 0.52",
           ylabel="Fitted minus Jeffreys (Brier ALC)")
    ax.text(.02, .94, rf"Spearman $\rho={rho:.2f}$", transform=ax.transAxes, va="top")
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3)
    fig.subplots_adjust(left=.14, right=.98, top=.83, bottom=.20)
    save_figure(fig, args.results / "figure4-benchmark-explanation-17")


if __name__ == "__main__":
    main()
