from __future__ import annotations

import pandas as pd

from aims_starter.evaluation import simulate_beta_binomial


def _observations(n_items: int = 80):
    rows = []
    for item in range(n_items):
        repeats = 2 if item == 0 else 1
        for trial in range(repeats):
            rows.append(
                {
                    "task_id": "subject::benchmark",
                    "subject_uid": "subject",
                    "benchmark_id": "benchmark",
                    "item_uid": f"benchmark::{item:03d}",
                    "observation_id": f"obs-{item:03d}-{trial}",
                    "label": (item + trial) % 2,
                    "split_by_benchmark": "test",
                }
            )
    return pd.DataFrame(rows)


def test_simulation_matches_competition_shape():
    result = simulate_beta_binomial(_observations(), seed=7)

    assert result["n_tasks"] == 1
    assert result["protocol"]["budgets"] == [0, 1, 3, 7, 15, 31]
    task = result["tasks"][0]
    assert task["n_acquisition_items"] == 40
    assert task["n_evaluation_items"] == 40
    assert task["by_budget"]["31"]["labels_acquired"] == 31
    assert 40 <= task["n_evaluation_rows"] <= 41
    assert 0.0 <= result["brier_alc"] <= 1.0


def test_simulation_rejects_ineligible_tasks():
    try:
        simulate_beta_binomial(_observations(79))
    except ValueError as exc:
        assert "No eligible tasks" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ineligible task failure")
