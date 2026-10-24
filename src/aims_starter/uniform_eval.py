"""Shared definitions for the uniform ablation grid.

The report uses one validation grid for every ablation.  This module keeps the
fold assignment and the table schema in one place so that the ladder,
calibration, disjoint, and acquisition scripts cannot drift apart.

Fold assignment is deterministic and stratified by benchmark.  Tasks are
sorted by benchmark and task id, then dealt round-robin within each benchmark.
Every script must use this function rather than its own random split.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
N_FOLDS = 5


def validation_fold_frame(evaluation_items: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Return one row per validation task with a deterministic fold label."""

    validation = evaluation_items[evaluation_items["split"] == "validation"].copy()
    tasks = (
        validation[["task_id", "benchmark_id"]]
        .drop_duplicates()
        .assign(
            task_id=lambda frame: frame["task_id"].astype(str),
            benchmark_id=lambda frame: frame["benchmark_id"].astype(str),
        )
        .sort_values(["benchmark_id", "task_id"])
        .reset_index(drop=True)
    )
    tasks["fold"] = tasks.groupby("benchmark_id", sort=False).cumcount() % int(n_folds)
    if tasks["task_id"].duplicated().any():
        raise ValueError("a validation task appears under more than one benchmark")
    return tasks


def validation_fold_map(evaluation_items: pd.DataFrame, n_folds: int = N_FOLDS) -> dict[str, int]:
    frame = validation_fold_frame(evaluation_items, n_folds=n_folds)
    return {str(task_id): int(fold) for task_id, fold in zip(frame["task_id"], frame["fold"], strict=True)}


def add_fold(records: Iterable[dict], fold_map: dict[str, int]) -> None:
    for record in records:
        record["fold"] = int(fold_map[str(record["task_id"])])


def task_bootstrap(
    values: np.ndarray,
    *,
    draws: int = 10_000,
    seed: int = 20261020,
) -> dict[str, float]:
    """Percentile bootstrap over tasks, paired when values are differences."""

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "draws": int(draws),
    }
