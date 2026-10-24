#!/usr/bin/env python3
"""Uniform validation-grid budget-shrinkage ablation.

The flat baseline is Beta(2, 2), p=(2+s)/(4+n).  The budget-specific arm uses a
Beta(kappa_b/2, kappa_b/2) prior at each budget.  Kappa is selected on the four
training folds of the validation split and evaluated on the held-out fold, so
the reported validation score is out of fold.

The target rate for each task is the evaluation-pool base rate.  This makes the
Brier calculation exact for a constant probability and avoids fitting anything
on the evaluation labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aims_starter.uniform_eval import (
    BUDGETS,
    WEIGHTS,
    task_bootstrap,
    validation_fold_frame,
)

CORPUS = Path("/tmp/opencode/allbench-features-v1")
OUTPUT = Path("/tmp/opencode/uniform-v1/budget-shrinkage")
KAPPA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
FLAT_KAPPA = 4.0


def probability(successes: float, budget: int, kappa: float) -> float:
    return (0.5 * kappa + successes) / (kappa + budget)


def brier(p: float, rate: float) -> float:
    return p * p - 2.0 * p * rate + rate


def main() -> int:
    items = pd.read_parquet(CORPUS / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS / "eval_orders.parquet")
    validation = items[items["split"] == "validation"].copy()
    folds = validation_fold_frame(items, n_folds=5)
    fold_map = dict(zip(folds["task_id"], folds["fold"], strict=True))

    base_rows: list[dict] = []
    for task_id, task_rows in validation.groupby("task_id", sort=True):
        target = task_rows[task_rows["pool"] == "evaluation"]
        candidates = task_rows[task_rows["pool"] == "acquisition"]
        if target.empty or candidates.empty:
            continue
        rate = float(target["label"].mean())
        lookup = candidates.set_index("item_index")["label"]
        for order_seed in range(10):
            order = orders[
                (orders["task_id"] == task_id) & (orders["order_seed"] == order_seed)
            ].sort_values("acquisition_rank")
            labels = lookup.loc[order["item_index"].to_numpy()].to_numpy(dtype=float)
            for budget in BUDGETS:
                successes = float(labels[:budget].sum())
                base_rows.append(
                    {
                        "task_id": str(task_id),
                        "benchmark_id": str(task_rows["benchmark_id"].iloc[0]),
                        "fold": int(fold_map[str(task_id)]),
                        "order_seed": int(order_seed),
                        "budget": int(budget),
                        "successes": successes,
                        "rate": rate,
                    }
                )
    frame = pd.DataFrame(base_rows)

    chosen: dict[int, dict[int, float]] = {}
    for fold in range(5):
        fit = frame[frame["fold"] != fold]
        chosen[fold] = {}
        for budget in BUDGETS:
            group = fit[fit["budget"] == budget]
            losses = []
            for kappa in KAPPA_GRID:
                p = probability(group["successes"].to_numpy(), budget, kappa)
                losses.append(float(np.mean(brier(p, group["rate"].to_numpy()))))
            chosen[fold][budget] = float(KAPPA_GRID[int(np.argmin(losses))])

    rows: list[dict] = []
    for row in frame.itertuples(index=False):
        flat_p = probability(row.successes, row.budget, FLAT_KAPPA)
        tuned_p = probability(row.successes, row.budget, chosen[int(row.fold)][int(row.budget)])
        rows.append(
            {
                "task_id": row.task_id,
                "benchmark_id": row.benchmark_id,
                "fold": int(row.fold),
                "order_seed": int(row.order_seed),
                "budget": int(row.budget),
                "flat_brier": brier(flat_p, row.rate),
                "tuned_brier": brier(tuned_p, row.rate),
                "flat_probability": flat_p,
                "tuned_probability": tuned_p,
                "kappa": chosen[int(row.fold)][int(row.budget)],
            }
        )
    result = pd.DataFrame(rows)

    weights = dict(zip(BUDGETS, WEIGHTS, strict=True))
    result["flat_weighted"] = result["budget"].map(weights) * result["flat_brier"]
    result["tuned_weighted"] = result["budget"].map(weights) * result["tuned_brier"]
    per_task = (
        result.groupby(["task_id", "order_seed"], sort=False)[["flat_weighted", "tuned_weighted"]]
        .sum()
        .groupby("task_id", sort=False)
        .mean()
    )
    delta = (per_task["tuned_weighted"] - per_task["flat_weighted"]).to_numpy(dtype=float)
    summary = {
        "flat_alc": float(per_task["flat_weighted"].mean()),
        "tuned_alc": float(per_task["tuned_weighted"].mean()),
        "delta": task_bootstrap(delta),
        "tasks": int(len(per_task)),
        "orders": 10,
        "folds": 5,
        "kappa_grid": list(KAPPA_GRID),
        "selected_kappa": {str(fold): {str(b): k for b, k in values.items()} for fold, values in chosen.items()},
    }
    by_budget = (
        result.groupby("budget", as_index=False)
        .agg(
            flat_brier=("flat_brier", "mean"),
            tuned_brier=("tuned_brier", "mean"),
            delta_brier=("flat_brier", lambda values: 0.0),
        )
        .drop(columns=["delta_brier"])
    )
    by_budget["delta_brier"] = by_budget["tuned_brier"] - by_budget["flat_brier"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT / "rows.parquet", index=False)
    artifact = {
        "status": "uniform_budget_shrinkage",
        "split": "validation",
        "summary": summary,
        "by_budget": by_budget.to_dict(orient="records"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
