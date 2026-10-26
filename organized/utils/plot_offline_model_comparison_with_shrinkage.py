#!/usr/bin/env python3
"""Paired offline comparison including validation-selected shrinkage."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = Path("/tmp/opencode/allbench2/per_task_rows.parquet")
EB = Path("/tmp/empirical-bayes-results.json")
BUDGETS = (0, 1, 3, 7, 15, 31)
KAPPA = {0: 4.0, 1: 3.890451449942805, 3: 3.630780547701014,
         7: 3.715352290971724, 15: 3.8018939632056123, 31: 3.981071705534973}


def _summary(frame: pd.DataFrame, metric: str):
    task = frame.groupby(["task_id", "budget"], as_index=False)[metric].mean()
    pivot = task.pivot(index="task_id", columns="budget", values=metric).loc[:, list(BUDGETS)]
    values = pivot.to_numpy()
    return values.mean(0), values.std(0, ddof=1)


def _rate(row: dict) -> float:
    point = row["by_budget"]["1"]
    return (point["brier"] - point["probability"] ** 2) / (1 - 2 * point["probability"])


def _shrinkage_rows() -> pd.DataFrame:
    artifact = json.loads(EB.read_text())
    rows = []
    for row in artifact["test"]["baseline"]["rows"]:
        rate = _rate(row)
        for budget in BUDGETS:
            base_p = row["by_budget"][str(budget)]["probability"]
            successes = round(base_p * (4 + budget) - 2)
            kappa = KAPPA[budget]
            p = (0.5 * kappa + successes) / (kappa + budget)
            rows.append({
                "task_id": row["task_id"], "budget": budget,
                "brier": p * p - 2 * p * rate + rate,
                "ece": abs(p - rate),
            })
    return pd.DataFrame(rows)


def main() -> int:
    rows = pd.read_parquet(ROWS)
    rows = rows[(rows["split"] == "test") & (rows["mode"] == "masked")]
    methods = [
        ("beta_binomial", "Beta-Binomial", "black"),
        ("cnp_airt", "CNP-AIRT", "#377eb8"),
        ("online_airt_2pl", "Online-AIRT", "#e66101"),
    ]
    prepared = []
    for model, label, color in methods:
        frame = rows[rows["model"] == model]
        prepared.append((label, color, _summary(frame, "brier"), _summary(frame, "ece")))
    shrinkage = _shrinkage_rows()
    prepared.insert(1, ("Budget-specific shrinkage", "#4daf4a", _summary(shrinkage, "brier"), _summary(shrinkage, "ece")))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    x = np.arange(len(BUDGETS))
    for label, color, brier, ece in prepared:
        for axis, (mean, sd) in zip(axes, (brier, ece), strict=True):
            axis.fill_between(x, np.maximum(0, mean - sd), mean + sd, color=color, alpha=.09, linewidth=0)
            axis.plot(x, mean, "o-", color=color, lw=2.2, label=label)
    axes[0].set(title="Raw Brier", ylabel="Mean across paired tasks ↓")
    axes[1].set(title="Raw calibration error", ylabel="Mean ECE across paired tasks ↓")
    for axis in axes:
        axis.set(xlabel="Labels acquired", xticks=x, xticklabels=BUDGETS)
        axis.set_ylim(bottom=0)
    axes[1].legend(fontsize=8.5)
    fig.suptitle("Offline benchmark-held-out test: same 22 tasks and 10 acquisition orders; shading = ±1 SD")
    output = ROOT / "results" / "empirical-bayes-v1" / "offline_all_models_with_shrinkage.png"
    fig.savefig(output, dpi=190)
    print(output)


if __name__ == "__main__":
    main()
