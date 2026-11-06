"""Build paper curves from analysis.py CSVs; no model or response access."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
from plotstyle import CYCLE, TEXTWIDTH, style

BUDGETS = (0, 1, 3, 7, 15, 31)
STYLES = ("solid", "dashed", "dotted")


def build(table, shape, panels, series, output):
    """Series labels describe prior sources or fit modes; model IDs remain internal."""
    table = table.loc[table["shape"].eq(shape) & table.budget.astype(str).isin(map(str, BUDGETS))]
    table = table.assign(budget=table.budget.astype(int))
    fig, axs = plt.subplots(1, len(panels), figsize=(TEXTWIDTH, 2.25), squeeze=False, sharey=True)
    maximum = 0.
    for ax, (panel_name, role) in zip(axs[0], panels):
        for j, (label, model) in enumerate(series):
            curve = table.loc[table.role.eq(role) & table.model.eq(model)].sort_values("budget")
            if tuple(curve.budget) != BUDGETS or not curve.brier.between(0, 1).all():
                raise ValueError(f"Expected one valid budget curve for {shape}/{role}/{model}")
            maximum = max(maximum, curve.brier.max())
            ax.plot(np.arange(len(BUDGETS)), curve.brier, color=CYCLE[j],
                    linestyle=STYLES[j], marker="o", label=f"{label} ({STYLES[j]})")
        ax.set_xticks(np.arange(len(BUDGETS)), BUDGETS)
        ax.set_xlabel(f"Acquired labels ({panel_name})")
        ax.set_xlim(-.15, len(BUDGETS)-.85)
        style(ax)
    axs[0][0].set_ylabel("Brier score")
    axs[0][0].set_ylim(0, max(.1, min(1, np.ceil((maximum+.01)*10)/10)))
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(series), frameon=False)
    fig.subplots_adjust(left=.105, right=.98, top=.79, bottom=.23, wspace=.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + ext), bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True, type=Path, nargs="+")
    p.add_argument("--shape", default="competition")
    p.add_argument("--panel", action="append", required=True, help="Panel label=role, e.g. OOD=test")
    p.add_argument("--series", action="append", required=True, help="Prior source or fit mode=model ID")
    p.add_argument("--output", required=True, type=Path, help="Output stem")
    a = p.parse_args()
    if not 1 <= len(a.panel) <= 2 or not 1 <= len(a.series) <= 3:
        p.error("Use one or two panels and at most three series")
    panels = [tuple(x.split("=", 1)) for x in a.panel]
    series = [tuple(x.split("=", 1)) for x in a.series]
    if any(len(pair) != 2 or not all(pair) for pair in [*panels, *series]):
        p.error("Panels and series must use label=value")
    build(pd.concat([pd.read_csv(x) for x in a.summary], ignore_index=True), a.shape, panels, series, a.output)


if __name__ == "__main__":
    main()
