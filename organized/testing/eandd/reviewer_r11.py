"""Plot adaptive-prior harm and the pilot AUC from Person A's committed scores."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wp9 import CYCLE, TEXTWIDTH, plt, save_figure, style


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(a.data / "r3-adaptive-per-benchmark.csv").pivot(
        index="benchmark", columns="model", values="gain")
    b = b.sort_values("bb-eb")
    series = {"bb-eb": "Fitted prior", "bma-s": "Average strength",
              "bma-ms": "Average mean and strength", "mix2": "Two-component mixture"}
    assert len(b) == 17 and b[list(series)].notna().all().all()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 4.35))
    x = np.arange(len(b))
    ax.axvline(0, color=".45", lw=.8)
    for j, (model, name) in enumerate(series.items()):
        ax.scatter(b[model], x + (j - 1.5)*.16, label=name, s=14,
                   marker=["o", "s", "^", "D"][j], color=CYCLE[j])
    ax.set(yticks=x, yticklabels=b.index, xlabel="Brier ALC difference to Jeffreys (positive = harm)")
    ax.invert_yaxis()
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.46, 1.015), ncol=2)
    fig.subplots_adjust(left=.25, right=.98, top=.86, bottom=.13)
    save_figure(fig, a.out / "figure-r11-adaptive-harm")

    auc = pd.read_csv(a.data / "r4-pilot-auc.csv").sort_values("pilot_labels")
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.7))
    x = np.arange(len(auc))
    ax.axhline(.5, color=".55", lw=.8)
    ax.axhline(.7, color=".45", lw=.8, ls=":", label="Planned threshold")
    for j, (col, name) in enumerate([("auc_pooled", "Pooled"),
                                     ("auc_mean_per_benchmark", "Benchmark mean")]):
        vals = auc[col].to_numpy()
        bars = ax.bar(x + (j-.5)*.27, vals-.5, bottom=.5, width=.25, color=CYCLE[j], label=name)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=auc.pilot_labels, xlabel="Pilot labels",
           ylabel="AUC for predicting harm", ylim=(.48, .80))
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=3)
    fig.subplots_adjust(left=.13, right=.98, top=.83, bottom=.21)
    save_figure(fig, a.out / "figure-r11-pilot-auc")


if __name__ == "__main__":
    main()
