#!/usr/bin/env python3
"""Plot paired full-protocol test curves with between-task spread."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = Path("/tmp/opencode/allbench2/per_task_rows.parquet")
OUT = ROOT / "paper" / "neurips_2026" / "figures" / "offline-paired-raw-curves.png"
BUDGETS = (0, 1, 3, 7, 15, 31)
METHODS = (
    ("beta_binomial", "Beta-Binomial", "black"),
    ("cnp_airt", "CNP-AIRT masked", "#377eb8"),
    ("online_airt_2pl", "Online-AIRT masked", "#e66101"),
)


def main() -> int:
    rows = pd.read_parquet(ROWS)
    rows = rows[(rows["split"] == "test") & (rows["mode"] == "masked")].copy()
    # First average seeds and acquisition orders within each task. The shaded
    # spread therefore represents the 22 paired held-out tasks, not duplicate
    # order/seed observations.
    tasks = (
        rows.groupby(["model", "task_id", "budget"], as_index=False)
        .agg(brier=("brier", "mean"), ece=("ece", "mean"))
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
    x = np.arange(len(BUDGETS))
    for model, label, color in METHODS:
        selected = tasks[tasks["model"] == model]
        for axis, metric in ((axes[0], "brier"), (axes[1], "ece")):
            summary = selected.groupby("budget")[metric].agg(["mean", "std"]).loc[list(BUDGETS)]
            mean = summary["mean"].to_numpy()
            sd = summary["std"].to_numpy()
            axis.fill_between(x, np.maximum(0, mean - sd), mean + sd,
                              color=color, alpha=0.12, linewidth=0)
            axis.plot(x, mean, "o-", color=color, linewidth=2.3, label=label)

    axes[0].set(title="Raw Brier", ylabel="Mean across paired tasks ↓")
    axes[1].set(title="Raw calibration error", ylabel="Mean ECE across paired tasks ↓")
    for axis in axes:
        axis.set_xlabel("Labels acquired")
        axis.set_xticks(x, BUDGETS)
        axis.set_ylim(bottom=0)
    axes[1].legend(fontsize=9)
    fig.suptitle(
        "Offline benchmark-held-out test protocol\n"
        "same 22 tasks and 10 acquisition orders; shading = ±1 SD across tasks"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190)
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
