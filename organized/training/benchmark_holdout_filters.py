"""Leakage-safe response filtering and splitting for benchmark transfer.

This is deliberately separate from the within-benchmark ``n>=30`` filter.
That filter answers whether IRT can recover item parameters *inside* a dense
matrix; applying it globally before a benchmark split left no validation or
test benchmarks.  Here every quality rule is evaluated per benchmark, then a
new deterministic benchmark split is made from the benchmarks that survive.

The output never fills a missing response.  It contains only observed binary
responses and a ``benchmark_split`` column with train/validation/test roles.
Exact duplicate content that crosses roles is removed from every role before
the final iterative core filter, preventing direct question leakage.
"""

from __future__ import annotations

from collections.abc import Mapping


DEFAULT_MIN_ITEM_RESPONDERS = 15
DEFAULT_MIN_ITEMS_PER_SUBJECT = 80
DEFAULT_MIN_TASKS_PER_BENCHMARK = 10
DEFAULT_SPLIT_SEED = 20261018


def _stage_counts(frame, *, by_split: bool = False) -> dict:
    """Compact audit counts; intentionally never inspects unobserved cells."""
    if frame.empty:
        return {"rows": 0, "benchmarks": 0, "subjects": 0, "items": 0}
    result = {
        "rows": int(len(frame)),
        "benchmarks": int(frame["benchmark_id"].nunique()),
        "subjects": int(frame[["benchmark_id", "subject_uid"]].drop_duplicates().shape[0]),
        "items": int(frame[["benchmark_id", "item_uid"]].drop_duplicates().shape[0]),
    }
    if by_split:
        result["by_split"] = {
            str(role): {
                "rows": int(len(group)),
                "benchmarks": int(group["benchmark_id"].nunique()),
                "subjects": int(group[["benchmark_id", "subject_uid"]].drop_duplicates().shape[0]),
                "items": int(group[["benchmark_id", "item_uid"]].drop_duplicates().shape[0]),
            }
            for role, group in frame.groupby("benchmark_split", sort=True)
        }
    return result


def _canonical_observations(observations, items):
    """Aggregate repeated answers and canonicalise same-benchmark duplicates."""
    import pandas as pd

    required_obs = {"subject_uid", "item_uid", "benchmark_id", "label"}
    required_items = {"item_uid", "content_hash"}
    if missing := required_obs.difference(observations.columns):
        raise ValueError(f"observations missing columns: {sorted(missing)}")
    if missing := required_items.difference(items.columns):
        raise ValueError(f"items missing columns: {sorted(missing)}")

    # One binary outcome per observed subject/question/benchmark cell.  This
    # resolves duplicate grading rows without inventing a label for an absent
    # cell.  The original training filter used this same majority threshold.
    d = (observations.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)
         ["label"].mean())
    d = d.merge(items[["item_uid", "content_hash"]].drop_duplicates("item_uid"),
                on="item_uid", how="left", validate="many_to_one")
    if d["content_hash"].isna().any():
        # A missing content hash cannot be used to claim a cross-benchmark
        # duplicate. Treat it as a unique question rather than deleting data.
        d["content_hash"] = d["content_hash"].fillna(
            "__missing_content_hash__:" + d["item_uid"].astype(str)
        )
    d["content_hash"] = d["content_hash"].astype(str)

    # Different item IDs can represent the same question inside a benchmark.
    # Collapse them to one stable item ID before computing response support;
    # otherwise duplicate text artificially inflates an item's evidence.
    canonical = (d.groupby(["benchmark_id", "content_hash"], as_index=False)["item_uid"]
                   .min().rename(columns={"item_uid": "canonical_item_uid"}))
    d = (d.groupby(["subject_uid", "benchmark_id", "content_hash"], as_index=False)
         ["label"].mean().merge(canonical, on=["benchmark_id", "content_hash"],
                                 how="left", validate="many_to_one"))
    d = d.rename(columns={"canonical_item_uid": "item_uid"})
    d["label"] = (d["label"] >= 0.5).astype("int8")
    return d[["subject_uid", "item_uid", "benchmark_id", "content_hash", "label"]]


def _iterative_bipartite_core(frame, *, min_item_responders: int,
                              min_items_per_subject: int):
    """Filter every benchmark until both response-support constraints hold."""
    if min_item_responders < 2:
        raise ValueError("min_item_responders must be at least 2")
    if min_items_per_subject < 2:
        raise ValueError("min_items_per_subject must be at least 2")

    d = frame.copy()
    while True:
        before = len(d)
        if not before:
            return d
        item_stats = d.groupby(["benchmark_id", "item_uid"])["label"].agg(
            responders="size", pass_rate="mean"
        )
        good_items = item_stats.index[
            (item_stats["responders"] >= min_item_responders)
            & (item_stats["pass_rate"] > 0.0)
            & (item_stats["pass_rate"] < 1.0)
        ]
        d = d.set_index(["benchmark_id", "item_uid"])
        d = d.loc[d.index.isin(good_items)].reset_index()
        if d.empty:
            return d

        subject_counts = d.groupby(["benchmark_id", "subject_uid"])["item_uid"].nunique()
        good_subjects = subject_counts.index[
            subject_counts >= min_items_per_subject
        ]
        d = d.set_index(["benchmark_id", "subject_uid"])
        d = d.loc[d.index.isin(good_subjects)].reset_index()
        if len(d) == before:
            return d


def _eligible_benchmarks(frame, *, min_tasks_per_benchmark: int) -> set[str]:
    if frame.empty:
        return set()
    tasks = frame.groupby("benchmark_id")["subject_uid"].nunique()
    return set(tasks[tasks >= min_tasks_per_benchmark].index.astype(str))


def _split_benchmarks(benchmarks: set[str], *, seed: int,
                      validation_fraction: float, test_fraction: float) -> Mapping[str, str]:
    """Create a fixed split with at least one validation and test benchmark."""
    import numpy as np

    ordered = sorted(map(str, benchmarks))
    if len(ordered) < 3:
        raise RuntimeError(
            f"need at least three eligible benchmarks for train/validation/test; got {len(ordered)}"
        )
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must lie in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    shuffled = np.random.default_rng(seed).permutation(ordered).tolist()
    n_validation = max(1, round(len(shuffled) * validation_fraction))
    n_test = max(1, round(len(shuffled) * test_fraction))
    while len(shuffled) - n_validation - n_test < 1:
        if n_test > 1:
            n_test -= 1
        elif n_validation > 1:
            n_validation -= 1
        else:
            raise RuntimeError("split fractions leave no training benchmark")
    return {
        **{benchmark: "validation" for benchmark in shuffled[:n_validation]},
        **{benchmark: "test" for benchmark in shuffled[n_validation:n_validation + n_test]},
        **{benchmark: "train" for benchmark in shuffled[n_validation + n_test:]},
    }


def prepare_benchmark_holdout(
    observations,
    items,
    *,
    min_item_responders: int = DEFAULT_MIN_ITEM_RESPONDERS,
    min_items_per_subject: int = DEFAULT_MIN_ITEMS_PER_SUBJECT,
    min_tasks_per_benchmark: int = DEFAULT_MIN_TASKS_PER_BENCHMARK,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
):
    """Return a clean observed-response frame and an auditable split manifest."""
    audit = {
        "parameters": {
            "min_item_responders": int(min_item_responders),
            "min_items_per_subject": int(min_items_per_subject),
            "min_tasks_per_benchmark": int(min_tasks_per_benchmark),
            "split_seed": int(split_seed),
            "validation_fraction": float(validation_fraction),
            "test_fraction": float(test_fraction),
            "missing_response_policy": "masked_not_imputed",
            "cross_split_duplicate_policy": "remove_all_exact_content_hash_overlaps",
        }
    }
    d = _canonical_observations(observations, items)
    audit["after_cell_and_within_benchmark_duplicate_collapse"] = _stage_counts(d)
    d = _iterative_bipartite_core(
        d, min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject,
    )
    audit["after_per_benchmark_iterative_core"] = _stage_counts(d)
    initial_benchmarks = _eligible_benchmarks(d, min_tasks_per_benchmark=min_tasks_per_benchmark)
    d = d[d["benchmark_id"].astype(str).isin(initial_benchmarks)].copy()
    audit["after_minimum_tasks_per_benchmark"] = _stage_counts(d)

    split_map = _split_benchmarks(
        initial_benchmarks, seed=split_seed,
        validation_fraction=validation_fraction, test_fraction=test_fraction,
    )
    d["benchmark_split"] = d["benchmark_id"].astype(str).map(split_map)

    # Exact content shared by train and evaluation is direct item leakage.  It
    # is safer to remove it from all roles than to retain a privileged copy in
    # one role.  This is a filtering decision, not a train-time augmentation.
    content_roles = d.groupby("content_hash")["benchmark_split"].nunique()
    overlapping = set(content_roles[content_roles > 1].index)
    d = d[~d["content_hash"].isin(overlapping)].copy()
    audit["cross_split_duplicate_content_hashes_removed"] = int(len(overlapping))
    audit["after_cross_split_duplicate_removal"] = _stage_counts(d, by_split=True)

    d = _iterative_bipartite_core(
        d, min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject,
    )
    final_benchmarks = _eligible_benchmarks(d, min_tasks_per_benchmark=min_tasks_per_benchmark)
    d = d[d["benchmark_id"].astype(str).isin(final_benchmarks)].copy()
    audit["final"] = _stage_counts(d, by_split=True)
    final_roles = set(d["benchmark_split"].astype(str).unique())
    required_roles = {"train", "validation", "test"}
    if final_roles != required_roles:
        raise RuntimeError(
            "benchmark filtering left an empty split role; adjust support thresholds or split seed. "
            f"final roles={sorted(final_roles)} audit={audit['final']}"
        )
    audit["benchmark_split_map"] = {
        benchmark: split_map[benchmark] for benchmark in sorted(final_benchmarks)
    }
    return d.reset_index(drop=True), audit


def load_benchmark_holdout_filtered(processed_dir, **kwargs):
    """Load source parquet files and run :func:`prepare_benchmark_holdout`."""
    import pandas as pd

    observations = pd.read_parquet(
        processed_dir / "observations.parquet",
        columns=["subject_uid", "item_uid", "benchmark_id", "label"],
    )
    items = pd.read_parquet(
        processed_dir / "items.parquet", columns=["item_uid", "content_hash"]
    )
    return prepare_benchmark_holdout(observations, items, **kwargs)
