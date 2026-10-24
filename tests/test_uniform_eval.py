"""Tests for the shared uniform ablation grid.

``uniform_eval`` assigns the five validation folds that every out-of-fold
calibration in the ablation protocol depends on. ``docs/uniform-audit.md``
asserts that the folds cover every validation task exactly once; these tests
make that claim enforceable rather than aspirational.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aims_starter.uniform_eval import (
    N_FOLDS,
    add_fold,
    task_bootstrap,
    validation_fold_frame,
    validation_fold_map,
)


def _evaluation_items(
    benchmark_sizes: dict[str, int],
    *,
    split: str = "validation",
    rows_per_task: int = 3,
) -> pd.DataFrame:
    """Build a synthetic eval_items frame with repeated rows per task."""

    records = []
    for benchmark, size in benchmark_sizes.items():
        for index in range(size):
            for _ in range(rows_per_task):
                records.append(
                    {
                        "split": split,
                        "benchmark_id": benchmark,
                        "task_id": f"{benchmark}-task-{index:03d}",
                    }
                )
    return pd.DataFrame(records)


def test_every_validation_task_appears_exactly_once() -> None:
    items = _evaluation_items({"alpha": 11, "beta": 7, "gamma": 1})
    frame = validation_fold_frame(items)

    expected = items["task_id"].nunique()
    assert len(frame) == expected
    assert not frame["task_id"].duplicated().any()
    assert set(frame["task_id"]) == set(items["task_id"])


def test_every_fold_label_is_within_range() -> None:
    frame = validation_fold_frame(_evaluation_items({"alpha": 23, "beta": 9}))
    assert frame["fold"].between(0, N_FOLDS - 1).all()


def test_folds_are_balanced_within_each_benchmark() -> None:
    frame = validation_fold_frame(_evaluation_items({"alpha": 23, "beta": 9}))
    for _, group in frame.groupby("benchmark_id"):
        # Reindex across the full grid: an unused fold must count as zero,
        # otherwise collapsing every task into one fold would look "balanced".
        counts = group["fold"].value_counts().reindex(range(N_FOLDS), fill_value=0)
        assert counts.max() - counts.min() <= 1


def test_folds_are_actually_spread_across_the_grid() -> None:
    """A benchmark with at least N_FOLDS tasks must use every fold."""

    frame = validation_fold_frame(_evaluation_items({"alpha": 23}))
    assert set(frame["fold"]) == set(range(N_FOLDS))


def test_consecutive_tasks_are_dealt_to_different_folds() -> None:
    """Round-robin dealing, not block assignment."""

    frame = validation_fold_frame(_evaluation_items({"alpha": 10}))
    folds = list(frame.sort_values("task_id")["fold"])
    assert folds == [index % N_FOLDS for index in range(10)]


def test_raises_when_a_task_spans_two_benchmarks() -> None:
    items = pd.DataFrame(
        [
            {"split": "validation", "benchmark_id": "alpha", "task_id": "shared"},
            {"split": "validation", "benchmark_id": "beta", "task_id": "shared"},
        ]
    )
    with pytest.raises(ValueError, match="more than one benchmark"):
        validation_fold_frame(items)


def test_fold_assignment_is_invariant_to_input_row_order() -> None:
    items = _evaluation_items({"alpha": 13, "beta": 6})
    shuffled = items.sample(frac=1.0, random_state=7).reset_index(drop=True)

    first = validation_fold_frame(items)
    second = validation_fold_frame(shuffled)
    pd.testing.assert_frame_equal(first, second)


def test_non_validation_rows_are_excluded() -> None:
    validation = _evaluation_items({"alpha": 4})
    other = _evaluation_items({"alpha": 4}, split="test")
    other["task_id"] = other["task_id"] + "-test"

    frame = validation_fold_frame(pd.concat([validation, other], ignore_index=True))
    assert len(frame) == 4
    assert not frame["task_id"].str.endswith("-test").any()


def test_validation_fold_map_agrees_with_frame() -> None:
    items = _evaluation_items({"alpha": 9, "beta": 5})
    frame = validation_fold_frame(items)
    mapping = validation_fold_map(items)

    assert set(mapping) == set(frame["task_id"])
    assert all(isinstance(key, str) for key in mapping)
    assert all(isinstance(value, int) for value in mapping.values())
    for task_id, fold in zip(frame["task_id"], frame["fold"], strict=True):
        assert mapping[task_id] == int(fold)


def test_add_fold_mutates_records_in_place() -> None:
    items = _evaluation_items({"alpha": 6})
    mapping = validation_fold_map(items)
    records = [{"task_id": task_id} for task_id in mapping]

    assert add_fold(records, mapping) is None
    for record in records:
        assert record["fold"] == mapping[record["task_id"]]


def test_add_fold_raises_on_unknown_task_id() -> None:
    """Pins today's unguarded failure mode: an unmapped task raises KeyError."""

    with pytest.raises(KeyError):
        add_fold([{"task_id": "never-seen"}], {"alpha-task-000": 0})


def test_task_bootstrap_ci_brackets_the_mean() -> None:
    values = np.array([0.1, -0.2, 0.05, 0.3, -0.15, 0.22, 0.0, 0.11])
    result = task_bootstrap(values, draws=2_000)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert result["draws"] == 2_000


def test_task_bootstrap_is_deterministic_given_seed() -> None:
    values = np.linspace(-0.5, 0.5, 17)
    first = task_bootstrap(values, draws=1_000, seed=123)
    second = task_bootstrap(values, draws=1_000, seed=123)
    assert first == second


def test_task_bootstrap_zero_variance_collapses_interval() -> None:
    values = np.full(12, 0.25)
    result = task_bootstrap(values, draws=500)
    assert result["mean"] == pytest.approx(0.25)
    assert result["ci_low"] == pytest.approx(0.25)
    assert result["ci_high"] == pytest.approx(0.25)
