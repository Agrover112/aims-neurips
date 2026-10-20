from __future__ import annotations

import pandas as pd

from aims_starter.data import (
    build_acquisition_order_index,
    iter_episodes,
    prepare_measurement_data,
)


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
