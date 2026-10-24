from __future__ import annotations

import pandas as pd
import pytest

from aims_starter.data import (
    _episode_order,
    _split,
    _stable_hash,
    build_acquisition_order_index,
    build_episode_index,
    iter_episodes,
    prepare_measurement_data,
)


def _observations(task_item_counts: dict[str, int]) -> pd.DataFrame:
    """Build a minimal observations frame with one response row per item."""

    records = []
    for task_id, count in task_item_counts.items():
        for index in range(count):
            records.append(
                {
                    "task_id": task_id,
                    "subject_uid": f"{task_id}-subject",
                    "benchmark_id": "bench",
                    "item_uid": f"{task_id}-i{index:03d}",
                    "observation_id": f"{task_id}-o{index:03d}",
                    "split_by_benchmark": "train",
                }
            )
    return pd.DataFrame(records)


def test_prepare_and_episode_generation(tmp_path):
    source = tmp_path / "raw" / "toy" / "formatted_tables"
    source.mkdir(parents=True)
    pd.DataFrame(
        [{"benchmark_id": "bench", "response_type": "binary", "granularity": "item"}]
    ).to_csv(source / "benchmarks.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": "s1", "normalized_name": "model-a", "provider": "org"},
            {"subject_id": "s2", "normalized_name": "model-b", "provider": "org"},
        ]
    ).to_csv(source / "subjects.csv", index=False)
    pd.DataFrame(
        [
            {"item_id": "i1", "content": "one", "item_features": "kind=a"},
            {"item_id": "i2", "content": "two", "item_features": "kind=b"},
        ]
    ).to_csv(source / "items.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": "s1", "item_id": "i1", "trial": 0, "response": 1},
            {"subject_id": "s1", "item_id": "i1", "trial": 1, "response": 0},
            {"subject_id": "s1", "item_id": "i2", "trial": 0, "response": 1},
            {"subject_id": "s2", "item_id": "i1", "trial": 0, "response": 0},
            {"subject_id": "s2", "item_id": "i2", "trial": 0, "response": None},
        ]
    ).to_csv(source / "responses.csv", index=False)

    result = prepare_measurement_data(tmp_path / "raw", tmp_path / "processed")
    assert result.manifest["rows"]["observations"] == 4
    assert result.manifest["rows"]["pairs"] == 3
    assert result.manifest["rows"]["tasks"] == 2
    assert result.manifest["dropped_observations"][0]["rows"] == 1

    observations = pd.read_parquet(result.output / "observations.parquet")
    repeated = observations[observations["item_id"] == "i1"]
    s1_repeated = repeated[repeated["subject_id"] == "s1"]
    assert s1_repeated["split_by_input"].nunique() == 1
    assert observations["item_uid"].str.startswith("bench::").all()

    episodes = list(iter_episodes(observations, budgets=(0, 1)))
    assert {episode["budget"] for episode in episodes} == {0, 1}
    assert all(set(ep["context"].index).isdisjoint(ep["target"].index) for ep in episodes)


def test_acquisition_orders_are_shared_reproducible_permutations():
    items = pd.DataFrame(
        {
            "task_id": ["a"] * 4 + ["b"] * 4,
            "item_uid": [f"i{x}" for x in range(4)] * 2,
            "pool": ["acquisition"] * 3 + ["evaluation"] + ["acquisition"] * 3 + ["evaluation"],
            "split_by_benchmark": ["test"] * 8,
        }
    )
    first = build_acquisition_order_index(items, order_seeds=(0, 1))
    second = build_acquisition_order_index(items, order_seeds=(0, 1))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert set(first["item_uid"]) == {"i0", "i1", "i2"}
    ranks = first.groupby(["task_id", "order_seed"])["acquisition_rank"].apply(list)
    assert all(value == [1, 2, 3] for value in ranks)


def test_single_item_task_yields_an_empty_evaluation_pool() -> None:
    """Pins a degenerate split: a one-item task has no held-out evaluation row.

    ``acquisition_count = (len(item_ids) + 1) // 2`` rounds up, so a task with
    a single distinct item puts that item in the acquisition pool and leaves
    the evaluation pool empty. Such a task contributes no scoring signal, and
    downstream consumers must not assume every task has evaluation rows.
    """

    result = build_episode_index(_observations({"solo": 1}), minimum_items=1)

    assert len(result) == 1
    assert list(result["pool"]) == ["acquisition"]
    assert (result["pool"] == "evaluation").sum() == 0


def test_two_item_task_splits_one_and_one() -> None:
    result = build_episode_index(_observations({"pair": 2}), minimum_items=1)

    assert (result["pool"] == "acquisition").sum() == 1
    assert (result["pool"] == "evaluation").sum() == 1


def test_pools_partition_every_item_without_overlap() -> None:
    observations = _observations({"task": 9})
    result = build_episode_index(observations, minimum_items=1)

    acquisition = set(result.loc[result["pool"] == "acquisition", "item_uid"])
    evaluation = set(result.loc[result["pool"] == "evaluation", "item_uid"])

    assert acquisition & evaluation == set()
    assert acquisition | evaluation == set(observations["item_uid"])
    # (9 + 1) // 2 == 5 items are acquired.
    assert len(acquisition) == 5


def test_acquisition_ranks_are_consecutive_from_one() -> None:
    result = build_episode_index(_observations({"task": 8}), minimum_items=1)
    ranks = sorted(
        int(rank) for rank in result.loc[result["pool"] == "acquisition", "acquisition_rank"]
    )

    assert ranks == list(range(1, len(ranks) + 1))


def test_tasks_below_minimum_items_are_dropped() -> None:
    observations = _observations({"small": 3, "large": 12})
    result = build_episode_index(observations, minimum_items=10)

    assert set(result["task_id"]) == {"large"}


def test_build_episode_index_is_deterministic() -> None:
    observations = _observations({"task": 11})
    first = build_episode_index(observations, minimum_items=1)
    second = build_episode_index(observations, minimum_items=1)

    pd.testing.assert_frame_equal(first, second)


def test_build_episode_index_rejects_a_nonpositive_minimum() -> None:
    with pytest.raises(ValueError, match="minimum_items must be positive"):
        build_episode_index(_observations({"task": 4}), minimum_items=0)


def test_build_episode_index_reports_missing_columns() -> None:
    observations = _observations({"task": 4}).drop(columns=["observation_id"])
    with pytest.raises(ValueError, match="missing episode columns"):
        build_episode_index(observations, minimum_items=1)


def test_stable_hash_is_deterministic_and_seed_sensitive() -> None:
    assert _stable_hash("a", "b", seed=1) == _stable_hash("a", "b", seed=1)
    assert _stable_hash("a", "b", seed=1) != _stable_hash("a", "b", seed=2)
    assert _stable_hash("a", "b", seed=1) != _stable_hash("b", "a", seed=1)
    assert len(_stable_hash("a", seed=0, length=12)) == 12


def test_episode_order_is_deterministic_and_full_length() -> None:
    first = _episode_order("task", "item", "pool", seed=7)
    assert first == _episode_order("task", "item", "pool", seed=7)
    assert len(first) == 64


def test_split_is_deterministic_and_covers_every_bucket() -> None:
    assert _split("subject-1", 0) == _split("subject-1", 0)

    labels = {_split(f"subject-{index}", 0) for index in range(500)}
    assert labels == {"train", "validation", "test"}
