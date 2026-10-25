"""Utilities for probing benchmark-centered item-difficulty transfer."""

from __future__ import annotations

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""

    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_centered_rasch(
    labels: np.ndarray,
    item_codes: np.ndarray,
    task_codes: np.ndarray,
    n_items: int,
    n_tasks: int,
    *,
    rounds: int = 40,
    theta_variance: float = 4.0,
    difficulty_variance: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit task abilities and item difficulties for one benchmark.

    The response model is ``sigmoid(theta_task - difficulty_item)``.  Both
    parameter families have zero-centered Gaussian priors, and the fitted item
    difficulties are re-centered to weighted mean zero after every round.
    Centering removes the arbitrary benchmark-level offset before a text model
    is asked to predict difficulty across benchmarks.
    """

    labels = np.asarray(labels, dtype=float)
    item_codes = np.asarray(item_codes, dtype=np.int64)
    task_codes = np.asarray(task_codes, dtype=np.int64)
    if labels.ndim != 1 or item_codes.shape != labels.shape or task_codes.shape != labels.shape:
        raise ValueError("labels, item_codes, and task_codes must be aligned vectors")
    if n_items < 1 or n_tasks < 1 or labels.size == 0:
        raise ValueError("the Rasch fit needs at least one response, item, and task")
    if item_codes.min() < 0 or item_codes.max() >= n_items:
        raise ValueError("item code outside [0, n_items)")
    if task_codes.min() < 0 or task_codes.max() >= n_tasks:
        raise ValueError("task code outside [0, n_tasks)")

    theta = np.zeros(n_tasks, dtype=float)
    difficulty = np.zeros(n_items, dtype=float)
    item_count = np.bincount(item_codes, minlength=n_items).astype(float)

    for _ in range(int(rounds)):
        for _ in range(3):
            probability = sigmoid(theta[task_codes] - difficulty[item_codes])
            gradient = np.bincount(
                task_codes, weights=labels - probability, minlength=n_tasks
            ) - theta / theta_variance
            curvature = np.bincount(
                task_codes,
                weights=probability * (1.0 - probability),
                minlength=n_tasks,
            ) + 1.0 / theta_variance
            theta = np.clip(theta + gradient / np.maximum(curvature, 1e-9), -6.0, 6.0)

        for _ in range(3):
            probability = sigmoid(theta[task_codes] - difficulty[item_codes])
            gradient = np.bincount(
                item_codes, weights=probability - labels, minlength=n_items
            ) - difficulty / difficulty_variance
            curvature = np.bincount(
                item_codes,
                weights=probability * (1.0 - probability),
                minlength=n_items,
            ) + 1.0 / difficulty_variance
            difficulty = np.clip(
                difficulty + gradient / np.maximum(curvature, 1e-9), -6.0, 6.0
            )

        offset = float(np.average(difficulty, weights=item_count))
        difficulty -= offset
        theta -= offset

    return difficulty, theta


def center_by_group(
    values: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract a (possibly weighted) mean independently within each group."""

    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if values.shape != groups.shape:
        raise ValueError("values and groups must have the same shape")
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != values.shape or np.any(weights < 0.0):
        raise ValueError("weights must be non-negative and aligned with values")
    output = values.copy()
    for group in np.unique(groups):
        mask = groups == group
        if float(weights[mask].sum()) <= 0.0:
            raise ValueError(f"group {group!r} has zero total weight")
        output[mask] -= np.average(values[mask], weights=weights[mask])
    return output


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Return weighted R-squared, RMSE, and Pearson correlation."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("truth and prediction must be non-empty aligned vectors")
    weights = np.ones_like(truth) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != truth.shape or float(weights.sum()) <= 0.0:
        raise ValueError("weights must be aligned and have positive total mass")
    residual = truth - prediction
    mean = float(np.average(truth, weights=weights))
    ss_res = float(np.sum(weights * np.square(residual)))
    ss_tot = float(np.sum(weights * (truth - mean) ** 2))
    rmse = float(np.sqrt(ss_res / weights.sum()))
    if truth.size < 2 or np.std(truth) == 0.0 or np.std(prediction) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(truth, prediction)[0, 1])
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0,
        "rmse": rmse,
        "pearson": correlation,
    }
