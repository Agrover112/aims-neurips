"""Competition-shaped offline simulations for fast, leakage-safe baselines."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from .data import build_episode_index
from .metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, brier_alc, brier_score, calibration_ece


def simulate_beta_binomial(
    observations,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    prior_mean: float = 0.5,
    prior_strength: float = 4.0,
    seed: int = 20261017,
    minimum_items: int = 80,
    benchmark_split: str | None = "test",
    episode_index=None,
) -> dict[str, Any]:
    """Simulate the official nested-budget protocol with a smoothed mean.

    Distinct items receive a deterministic 50/50 acquisition/evaluation split.
    One deterministic response is revealed for each acquired item, while every
    response row belonging to an evaluation item is scored. Task curves are
    averaged equally, matching the competition's subject--benchmark weighting.
    """

    if not 0.0 <= prior_mean <= 1.0 or prior_strength <= 0.0:
        raise ValueError("prior_mean must be in [0, 1] and prior_strength must be positive")
    budgets = tuple(int(budget) for budget in budgets)
    if not budgets or budgets[0] != 0 or any(a >= b for a, b in pairwise(budgets)):
        raise ValueError("budgets must be strictly increasing and start at zero")
    if len(budgets) != len(weights):
        raise ValueError("budgets and weights must have the same length")

    frame = observations
    if benchmark_split is not None:
        if benchmark_split not in {"train", "validation", "test"}:
            raise ValueError("benchmark_split must be train, validation, test, or None")
        frame = frame[frame["split_by_benchmark"] == benchmark_split]

    assignments = episode_index
    if assignments is None:
        assignments = build_episode_index(
            frame,
            seed=seed,
            minimum_items=minimum_items,
        )
    elif benchmark_split is not None:
        assignments = assignments[
            assignments["split_by_benchmark"] == benchmark_split
        ]

    task_results: list[dict[str, Any]] = []
    observation_lookup = frame.set_index("observation_id", drop=False)
    for task_id, assignment in assignments.groupby("task_id", sort=True):
        task = frame[frame["task_id"].astype(str) == str(task_id)]
        acquisition = assignment[assignment["pool"] == "acquisition"].sort_values(
            "acquisition_rank"
        )
        evaluation_items = set(
            assignment.loc[assignment["pool"] == "evaluation", "item_uid"].astype(str)
        )
        if len(acquisition) < max(budgets) or not evaluation_items:
            continue

        targets = task[task["item_uid"].isin(evaluation_items)]
        truth = targets["label"].astype(float).tolist()
        label_ids = acquisition["label_observation_id"].astype(str).tolist()

        by_budget: dict[str, dict[str, float | int]] = {}
        scores = []
        for budget in budgets:
            context = observation_lookup.loc[label_ids[:budget]]
            successes = float(context["label"].sum())
            probability = (
                prior_mean * prior_strength + successes
            ) / (prior_strength + len(context))
            probabilities = [probability] * len(truth)
            score = brier_score(truth, probabilities)
            scores.append(score)
            by_budget[str(budget)] = {
                "labels_acquired": len(context),
                "probability": float(probability),
                "brier": score,
                "ece": calibration_ece(truth, probabilities),
            }

        task_results.append(
            {
                "task_id": str(task_id),
                "benchmark_id": str(task.iloc[0]["benchmark_id"]),
                "subject_uid": str(task.iloc[0]["subject_uid"]),
                "n_items": int(task["item_uid"].nunique()),
                "n_acquisition_items": len(acquisition),
                "n_evaluation_items": len(evaluation_items),
                "n_evaluation_rows": len(targets),
                "brier_alc": brier_alc(scores, weights),
                "by_budget": by_budget,
            }
        )

    if not task_results:
        split_name = benchmark_split or "all"
        raise ValueError(
            f"No eligible tasks in split={split_name!r}; each needs at least "
            f"{minimum_items} distinct items and {max(budgets)} acquisition items"
        )

    aggregate_by_budget = {}
    for budget in budgets:
        key = str(budget)
        aggregate_by_budget[key] = {
            metric: sum(float(task["by_budget"][key][metric]) for task in task_results)
            / len(task_results)
            for metric in ("brier", "ece")
        }

    return {
        "protocol": {
            "budgets": list(budgets),
            "weights": list(weights),
            "minimum_items": minimum_items,
            "benchmark_split": benchmark_split or "all",
            "item_split": "deterministic balanced 50/50 within task",
            "task_aggregation": "unweighted mean",
        },
        "n_tasks": len(task_results),
        "brier_alc": sum(float(task["brier_alc"]) for task in task_results)
        / len(task_results),
        "by_budget": aggregate_by_budget,
        "tasks": task_results,
    }
