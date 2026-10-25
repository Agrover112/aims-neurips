from __future__ import annotations

import numpy as np

from aims_starter.item_difficulty_probe import (
    center_by_group,
    fit_centered_rasch,
    regression_metrics,
    sigmoid,
)


def test_rasch_recovers_centered_difficulty_order() -> None:
    true_difficulty = np.array([-1.0, 0.0, 1.0])
    true_theta = np.linspace(-2.0, 2.0, 21)
    item_codes = np.tile(np.arange(3), len(true_theta))
    task_codes = np.repeat(np.arange(len(true_theta)), 3)
    probability = sigmoid(true_theta[task_codes] - true_difficulty[item_codes])
    labels = (probability >= 0.5).astype(float)

    fitted, _ = fit_centered_rasch(
        labels, item_codes, task_codes, n_items=3, n_tasks=len(true_theta)
    )

    assert np.all(np.diff(fitted) > 0.0)
    assert abs(float(fitted.mean())) < 1e-10


def test_center_by_group_uses_weights() -> None:
    values = np.array([1.0, 3.0, 10.0, 14.0])
    groups = np.array(["a", "a", "b", "b"])
    weights = np.array([1.0, 3.0, 2.0, 2.0])
    centered = center_by_group(values, groups, weights)

    for group in np.unique(groups):
        mask = groups == group
        assert abs(float(np.average(centered[mask], weights=weights[mask]))) < 1e-12


def test_regression_metrics_identify_perfect_prediction() -> None:
    truth = np.array([-1.0, 0.0, 2.0])
    metrics = regression_metrics(truth, truth.copy(), np.array([1.0, 2.0, 1.0]))

    assert metrics == {"r2": 1.0, "rmse": 0.0, "pearson": 1.0}
