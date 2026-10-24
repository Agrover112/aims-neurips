#!/usr/bin/env python3
"""Plot the validation-selected budget-specific shrinkage ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = Path("/tmp/empirical-bayes-results.json")
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (.1, .2, .2, .2, .2, .1)
KAPPA = {0: 4.0, 1: 3.890451449942805, 3: 3.630780547701014,
         7: 3.715352290971724, 15: 3.8018939632056123, 31: 3.981071705534973}


def target_rate(row: dict) -> float:
    point = row["by_budget"]["1"]
    p, brier = point["probability"], point["brier"]
    return (brier - p * p) / (1.0 - 2.0 * p)


def curves(rows: list[dict], schedule: bool):
    per_task: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        rate = target_rate(row)
        task = per_task.setdefault(row["task_id"], {b: [] for b in BUDGETS})
        for budget in BUDGETS:
            if schedule:
                kappa = KAPPA[budget]
                baseline_p = row["by_budget"][str(budget)]["probability"]
                successes = round(baseline_p * (4.0 + budget) - 2.0)
                p = (0.5 * kappa + successes) / (kappa + budget)
            else:
                p = row["by_budget"][str(budget)]["probability"]
            task[budget].append(p * p - 2.0 * p * rate + rate)
    values = np.asarray(
        [[np.mean(per_task[task][budget]) for budget in BUDGETS] for task in sorted(per_task)]
    )
    return values.mean(0), values.std(0, ddof=1), values


def main() -> int:
    result = json.loads(RESULT.read_text())
    data = {}
    for split in ("validation", "test"):
        rows = result[split]["baseline"]["rows"]
        data[split] = (curves(rows, False), curves(rows, True))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    x = np.arange(len(BUDGETS))
    for axis, split, title in zip(axes[:2], ("validation", "test"), ("Validation", "Test"), strict=True):
        (base_m, base_sd, _), (new_m, new_sd, _) = data[split]
        axis.fill_between(x, base_m-base_sd, base_m+base_sd, color="black", alpha=.10)
        axis.fill_between(x, new_m-new_sd, new_m+new_sd, color="#d95f02", alpha=.10)
        axis.plot(x, base_m, "o-", color="black", label="fixed Beta(2,2)")
        axis.plot(x, new_m, "o-", color="#d95f02", label="budget-specific shrinkage")
        axis.set(title=title, xlabel="Labels acquired", ylabel="Brier ↓", xticks=x, xticklabels=BUDGETS)
    axes[1].legend(fontsize=8)
    for split, color in (("validation", "#377eb8"), ("test", "#d95f02")):
        base, new = data[split][0][0], data[split][1][0]
        axes[2].plot(x, (new-base)*1e4, "o-", color=color, label=split)
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set(title="Difference, magnified", xlabel="Labels acquired", ylabel="new − fixed Brier × $10^4$", xticks=x, xticklabels=BUDGETS)
    axes[2].legend()
    fig.suptitle("Paired offline shrinkage ablation; shading = ±1 SD across tasks")
    out = ROOT / "results" / "empirical-bayes-v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "budget_shrinkage_brier_curves.png"
    fig.savefig(path, dpi=190)
    print(path)
    for split in ("validation", "test"):
        base, new = data[split][0][0], data[split][1][0]
        print(split, "fixed", base.tolist(), "new", new.tolist(), "ALC delta", float(np.dot(WEIGHTS, new-base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
