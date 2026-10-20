"""Canonical Measurement Data Bank preprocessing and episodic sampling.

Raw files remain untouched. The processor namespaces item identifiers, derives
stable subject signatures, preserves trial-level observations, aggregates
repeated inputs, and creates deterministic group splits for several evaluation
questions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TABLE_ALIASES = {
    "benchmarks": ("benchmarks",),
    "subjects": ("subjects", "models"),
    "items": ("items", "test_items"),
    "responses": ("responses", "response", "ground_truth"),
}


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("Data dependencies are missing. Run: pip install -e '.[data]'") from exc
    return pd


def _stable_hash(*values: object, seed: int = 0, length: int = 20) -> str:
    payload = "\x1f".join(str(value or "").strip() for value in values)
    return hashlib.sha256(f"{seed}\x1e{payload}".encode()).hexdigest()[:length]


def _split(value: object, seed: int) -> str:
    bucket = int(_stable_hash(value, seed=seed, length=8), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _read_table(directory: Path, names: Sequence[str]):
    pd = _pandas()
    for name in names:
        for suffix in (".parquet", ".csv"):
            path = directory / f"{name}{suffix}"
            if path.exists():
                if suffix == ".parquet":
                    return pd.read_parquet(path), path
                return pd.read_csv(path), path
    return None, None


def discover_benchmark_directories(source: str | Path) -> list[Path]:
    """Find directories containing canonical benchmark tables."""

    root = Path(source).resolve()
    found: set[Path] = set()
    for suffix in ("parquet", "csv"):
        for path in root.rglob(f"benchmarks.{suffix}"):
            found.add(path.parent)
    if not found:
        raise FileNotFoundError(
            f"No benchmarks.parquet or benchmarks.csv tables found under {root}"
        )
    return sorted(found)


def _clean_string_columns(frame) -> None:
    for column in frame.columns:
        if str(frame[column].dtype) in {"object", "string"}:
            frame[column] = frame[column].fillna("").map(lambda value: str(value).strip())


def _subject_signature(row: Any) -> str:
    fields = (
        "normalized_name",
        "provider",
        "release_date",
        "harness",
        "reasoning_effort",
        "harness_version",
        "subject_features_extra",
    )
    values = [getattr(row, field, "") for field in fields]
    if not any(str(value or "").strip() for value in values):
        values = [getattr(row, "subject_id", "")]
    return "subject_" + _stable_hash(*values)


def _is_eligible(benchmark, responses) -> bool:
    if "response_type" in benchmark and str(benchmark["response_type"]).lower() != "binary":
        return False
    if "granularity" in benchmark and str(benchmark["granularity"]).lower() != "item":
        return False
    values = set(responses["response"].dropna().astype(float).unique())
    return values.issubset({0.0, 1.0})


@dataclass(frozen=True)
class PreparedData:
    output: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PreparedEpisodes:
    output: Path
    manifest: dict[str, Any]


def _episode_order(*values: object, seed: int) -> str:
    """Return a deterministic full-length key for episode assignments."""

    return _stable_hash(*values, seed=seed, length=64)


def build_episode_index(
    observations,
    *,
    seed: int = 20261017,
    minimum_items: int = 80,
):
    """Freeze competition-shaped acquisition/evaluation assignments.

    Assignment is by distinct source item, so all repeated response rows for an
    item stay in one pool. Acquisition receives one deterministic observed row
    per item; evaluation retains every response row. This reproduces the public
    protocol's shape but not Codabench's private persistent random split.
    """

    pd = _pandas()
    if minimum_items < 1:
        raise ValueError("minimum_items must be positive")
    required = {
        "task_id",
        "subject_uid",
        "benchmark_id",
        "item_uid",
        "observation_id",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"observations missing episode columns: {missing}")

    # Collapse response rows once. This keeps the large-corpus path linear in
    # the number of source rows instead of repeatedly filtering each task.
    result = observations.groupby(
        ["task_id", "item_uid"], as_index=False, sort=True, observed=True
    ).agg(
        subject_uid=("subject_uid", "first"),
        benchmark_id=("benchmark_id", "first"),
        label_observation_id=("observation_id", "min"),
        n_response_rows=("observation_id", "size"),
        split_by_benchmark=("split_by_benchmark", "first"),
    )
    eligible_tasks = (
        result.groupby("task_id", sort=False, observed=True)["item_uid"]
        .size()
        .loc[lambda counts: counts >= minimum_items]
        .index
    )
    result = result[result["task_id"].isin(eligible_tasks)].copy()
    result["item_uid"] = result["item_uid"].astype(str)
    result["pool"] = "evaluation"
    result["acquisition_rank"] = pd.array([None] * len(result), dtype="Int64")

    task_groups = result.groupby("task_id", sort=True, observed=True).groups
    for task_number, (task_id, task_indices) in enumerate(task_groups.items(), start=1):
        if task_number == 1 or task_number % 500 == 0:
            print(
                f"episode assignments: task {task_number}/{len(task_groups)}",
                flush=True,
            )
        item_ids = result.loc[task_indices, "item_uid"].tolist()
        pool_order = sorted(
            item_ids,
            key=lambda item_uid: _episode_order(task_id, item_uid, "pool", seed=seed),
        )
        acquisition_count = (len(item_ids) + 1) // 2
        acquisition_items = set(pool_order[:acquisition_count])
        acquisition_order = sorted(
            acquisition_items,
            key=lambda item_uid: _episode_order(task_id, item_uid, "acquisition", seed=seed),
        )
        acquisition_rank = {
            item_uid: rank for rank, item_uid in enumerate(acquisition_order, start=1)
        }
        result.loc[task_indices, "pool"] = [
            "acquisition" if item_uid in acquisition_items else "evaluation"
            for item_uid in item_ids
        ]
        result.loc[task_indices, "acquisition_rank"] = [
            acquisition_rank.get(item_uid) for item_uid in item_ids
        ]

    if not result.empty:
        result.loc[result["pool"] == "evaluation", "label_observation_id"] = ""
        result["acquisition_rank"] = result["acquisition_rank"].astype("Int64")
        result = result.sort_values(["task_id", "pool", "acquisition_rank", "item_uid"])
        result = result.reset_index(drop=True)
    return result


def prepare_episode_index(
    observations,
    output: str | Path,
    *,
    seed: int = 20261017,
    minimum_items: int = 80,
) -> PreparedEpisodes:
    """Persist a frozen episode index and an auditable protocol manifest."""

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_root}")

    assignments = build_episode_index(
        observations,
        seed=seed,
        minimum_items=minimum_items,
    )
    if assignments.empty:
        raise ValueError(f"No tasks contain at least {minimum_items} distinct items")

    assignments.to_parquet(output_root / "episode_items.parquet", index=False)
    task_summary = (
        assignments.groupby(
            ["task_id", "subject_uid", "benchmark_id", "split_by_benchmark"],
            as_index=False,
        )
        .agg(
            n_items=("item_uid", "size"),
            n_acquisition_items=("pool", lambda values: int((values == "acquisition").sum())),
            n_evaluation_items=("pool", lambda values: int((values == "evaluation").sum())),
            n_evaluation_rows=(
                "n_response_rows",
                lambda values: int(
                    values[assignments.loc[values.index, "pool"] == "evaluation"].sum()
                ),
            ),
        )
        .sort_values("task_id")
        .reset_index(drop=True)
    )
    task_summary.to_parquet(output_root / "episode_tasks.parquet", index=False)
    manifest = {
        "format_version": 1,
        "seed": seed,
        "minimum_items": minimum_items,
        "budgets": [0, 1, 3, 7, 15, 31],
        "assignment_unit": "distinct item_uid within task_id",
        "acquisition_label": "one deterministic observation per acquisition item",
        "evaluation_rows": "all observations of evaluation items",
        "private_split_warning": (
            "Deterministic offline simulation; it does not reproduce Codabench's private "
            "persistent split or candidate order."
        ),
        "rows": {
            "episode_items": len(assignments),
            "episode_tasks": len(task_summary),
        },
        "pool_counts": {
            str(pool): int(count) for pool, count in assignments["pool"].value_counts().items()
        },
        "split_task_counts": {
            str(split): int(count)
            for split, count in task_summary["split_by_benchmark"].value_counts().items()
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PreparedEpisodes(output=output_root, manifest=manifest)


def build_acquisition_order_index(
    episode_items,
    *,
    order_seeds: Sequence[int] = tuple(range(10)),
):
    """Create shared deterministic reveal orders without changing item pools."""

    pd = _pandas()
    required = {"task_id", "item_uid", "pool", "split_by_benchmark"}
    missing = sorted(required - set(episode_items.columns))
    if missing:
        raise ValueError(f"episode_items missing order columns: {missing}")
    acquisition = episode_items[episode_items["pool"] == "acquisition"].copy()
    parts = []
    for order_seed in order_seeds:
        ordered = acquisition[["task_id", "item_uid", "split_by_benchmark"]].copy()
        ordered["order_seed"] = int(order_seed)
        ordered["order_key"] = [
            _stable_hash(task_id, item_uid, "acquisition-order", seed=int(order_seed))
            for task_id, item_uid in zip(ordered["task_id"], ordered["item_uid"], strict=True)
        ]
        ordered = ordered.sort_values(["task_id", "order_key", "item_uid"])
        ordered["acquisition_rank"] = ordered.groupby("task_id", sort=False).cumcount() + 1
        parts.append(ordered.drop(columns="order_key"))
    if not parts:
        return pd.DataFrame(
            columns=["task_id", "item_uid", "split_by_benchmark", "order_seed", "acquisition_rank"]
        )
    return pd.concat(parts, ignore_index=True)


def prepare_measurement_data(
    source: str | Path,
    output: str | Path,
    *,
    seed: int = 20261017,
) -> PreparedData:
    """Convert per-benchmark Measurement DB tables into shared Parquet tables."""

    pd = _pandas()
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_root}")

    all_benchmarks = []
    all_subjects = []
    all_items = []
    all_observations = []
    skipped: list[dict[str, str]] = []
    dropped_observations: list[dict[str, Any]] = []
    source_files: list[str] = []

    for directory in discover_benchmark_directories(source_root):
        tables = {}
        paths = {}
        for table, aliases in TABLE_ALIASES.items():
            tables[table], paths[table] = _read_table(directory, aliases)
        if any(tables[name] is None for name in TABLE_ALIASES):
            missing = [name for name in TABLE_ALIASES if tables[name] is None]
            skipped.append({"directory": str(directory), "reason": f"missing {missing}"})
            continue

        benchmark_frame = tables["benchmarks"].copy()
        subject_frame = tables["subjects"].copy()
        item_frame = tables["items"].copy()
        response_frame = tables["responses"].copy()
        for frame in (benchmark_frame, subject_frame, item_frame, response_frame):
            _clean_string_columns(frame)

        if "response" not in response_frame and "label" in response_frame:
            response_frame = response_frame.rename(columns={"label": "response"})
        required = {
            "subjects": {"subject_id"},
            "items": {"item_id"},
            "responses": {"subject_id", "item_id", "response"},
        }
        normalized_tables = {
            "subjects": subject_frame,
            "items": item_frame,
            "responses": response_frame,
        }
        absent = {
            name: sorted(columns - set(normalized_tables[name].columns))
            for name, columns in required.items()
            if columns - set(normalized_tables[name].columns)
        }
        if absent:
            skipped.append({"directory": str(directory), "reason": f"missing columns {absent}"})
            continue

        if benchmark_frame.empty:
            skipped.append({"directory": str(directory), "reason": "empty benchmark table"})
            continue
        benchmark = benchmark_frame.iloc[0]
        benchmark_id = str(benchmark.get("benchmark_id") or directory.parent.name)
        if not _is_eligible(benchmark, response_frame):
            skipped.append({"directory": str(directory), "reason": "not binary item-level data"})
            continue

        numeric_response = pd.to_numeric(response_frame["response"], errors="coerce")
        missing_response = int(numeric_response.isna().sum())
        if missing_response:
            dropped_observations.append(
                {
                    "benchmark_id": benchmark_id,
                    "source_directory": str(directory.relative_to(source_root)),
                    "reason": "missing response",
                    "rows": missing_response,
                }
            )
            response_frame = response_frame.loc[numeric_response.notna()].copy()
            numeric_response = numeric_response.loc[numeric_response.notna()]
        if response_frame.empty:
            skipped.append({"directory": str(directory), "reason": "no observed responses"})
            continue
        response_frame["response"] = numeric_response.astype(float)

        for frame in (benchmark_frame, subject_frame, item_frame, response_frame):
            frame["benchmark_id"] = benchmark_id
            frame["source_directory"] = str(directory.relative_to(source_root))

        if "content" not in item_frame and "item_content" in item_frame:
            item_frame = item_frame.rename(columns={"item_content": "content"})
        for column in ("content", "item_features"):
            if column not in item_frame:
                item_frame[column] = ""
        if "interactors" not in response_frame:
            response_frame["interactors"] = ""
        if "test_condition" not in response_frame:
            response_frame["test_condition"] = ""
        if "trial" not in response_frame:
            response_frame["trial"] = 0

        item_frame["item_uid"] = benchmark_id + "::" + item_frame["item_id"].astype(str)
        item_frame["model_text"] = (
            item_frame["content"].fillna("").astype(str)
            + "\n[ITEM_FEATURES] "
            + item_frame["item_features"].fillna("").astype(str)
        ).str.strip()
        item_frame["content_chars"] = item_frame["content"].str.len()
        item_frame["content_missing"] = item_frame["content"].eq("")

        subject_frame["subject_uid"] = [
            _subject_signature(row) for row in subject_frame.itertuples(index=False)
        ]
        subject_lookup = subject_frame.set_index("subject_id")["subject_uid"]
        item_lookup = item_frame.set_index("item_id")["item_uid"]
        response_frame["subject_uid"] = response_frame["subject_id"].map(subject_lookup)
        response_frame["item_uid"] = response_frame["item_id"].map(item_lookup)

        orphan_subjects = int(response_frame["subject_uid"].isna().sum())
        orphan_items = int(response_frame["item_uid"].isna().sum())
        if orphan_subjects or orphan_items:
            raise ValueError(
                f"{directory}: {orphan_subjects} orphan subjects and {orphan_items} orphan items"
            )

        response_frame["label"] = response_frame["response"].astype(float).astype("int8")
        response_frame["task_id"] = (
            response_frame["subject_uid"] + "::" + response_frame["benchmark_id"]
        )
        response_frame["condition_uid"] = [
            "condition_" + _stable_hash(test, inter)
            for test, inter in zip(
                response_frame["test_condition"], response_frame["interactors"], strict=True
            )
        ]
        response_frame["input_id"] = (
            response_frame["task_id"]
            + "::"
            + response_frame["item_uid"]
            + "::"
            + response_frame["condition_uid"]
        )
        response_frame["observation_id"] = [
            "obs_" + _stable_hash(input_id, trial, index)
            for index, (input_id, trial) in enumerate(
                zip(response_frame["input_id"], response_frame["trial"], strict=True)
            )
        ]

        all_benchmarks.append(benchmark_frame)
        all_subjects.append(subject_frame)
        all_items.append(item_frame)
        all_observations.append(response_frame)
        source_files.extend(str(path.relative_to(source_root)) for path in paths.values())

    if not all_observations:
        raise ValueError("No eligible benchmark tables were prepared")

    benchmarks = pd.concat(all_benchmarks, ignore_index=True, sort=False)
    subjects = pd.concat(all_subjects, ignore_index=True, sort=False)
    items = pd.concat(all_items, ignore_index=True, sort=False)
    observations = pd.concat(all_observations, ignore_index=True, sort=False)

    subjects = subjects.drop_duplicates(["benchmark_id", "subject_id"], keep="first")
    items = items.drop_duplicates("item_uid", keep="first")
    benchmarks = benchmarks.drop_duplicates("benchmark_id", keep="first")

    for column, key in (
        ("split_by_benchmark", "benchmark_id"),
        ("split_by_subject", "subject_uid"),
        ("split_by_task", "task_id"),
        ("split_by_input", "input_id"),
    ):
        observations[column] = observations[key].map(lambda value: _split(value, seed))

    input_groups = ["input_id", "task_id", "subject_uid", "benchmark_id", "item_uid"]
    pairs = (
        observations.groupby(input_groups, dropna=False, as_index=False)
        .agg(
            n_trials=("label", "size"),
            successes=("label", "sum"),
            response_mean=("label", "mean"),
            test_condition=("test_condition", "first"),
            interactors=("interactors", "first"),
            split_by_benchmark=("split_by_benchmark", "first"),
            split_by_subject=("split_by_subject", "first"),
            split_by_task=("split_by_task", "first"),
            split_by_input=("split_by_input", "first"),
        )
        .sort_values(["task_id", "input_id"])
        .reset_index(drop=True)
    )
    tasks = (
        observations.groupby(["task_id", "subject_uid", "benchmark_id"], as_index=False)
        .agg(
            n_observations=("label", "size"),
            n_inputs=("input_id", "nunique"),
            n_items=("item_uid", "nunique"),
            response_mean=("label", "mean"),
            split_by_benchmark=("split_by_benchmark", "first"),
            split_by_subject=("split_by_subject", "first"),
            split_by_task=("split_by_task", "first"),
        )
        .sort_values("task_id")
        .reset_index(drop=True)
    )

    outputs = {
        "benchmarks": benchmarks,
        "subjects": subjects,
        "items": items,
        "observations": observations,
        "pairs": pairs,
        "tasks": tasks,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output_root / f"{name}.parquet", index=False)

    nulls = {
        name: {column: int(count) for column, count in frame.isna().sum().items() if count}
        for name, frame in outputs.items()
    }
    manifest = {
        "format_version": 1,
        "source": str(source_root),
        "seed": seed,
        "eligible_benchmarks": sorted(benchmarks["benchmark_id"].astype(str).unique()),
        "skipped": skipped,
        "dropped_observations": dropped_observations,
        "source_files": sorted(set(source_files)),
        "rows": {name: len(frame) for name, frame in outputs.items()},
        "null_counts": nulls,
        "split_policy": "sha256 group hashing: 80% train, 10% validation, 10% test",
        "recommended_split": "split_by_benchmark for unseen-benchmark evaluation; split_by_task for held-out subject-benchmark episodes",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PreparedData(output=output_root, manifest=manifest)


def iter_episodes(
    observations,
    *,
    budgets: Sequence[int] = (0, 1, 3, 7, 15, 31),
    seed: int = 20261017,
    minimum_items: int = 1,
    episode_index=None,
) -> Iterator[dict[str, Any]]:
    """Yield nested, distinct-item episodes from a shared frozen assignment."""

    assignments = episode_index
    if assignments is None:
        assignments = build_episode_index(
            observations,
            seed=seed,
            minimum_items=minimum_items,
        )

    observation_lookup = observations.set_index("observation_id", drop=False)
    for task_id, task_assignment in assignments.groupby("task_id", sort=True):
        acquisition = task_assignment[task_assignment["pool"] == "acquisition"].sort_values(
            "acquisition_rank"
        )
        evaluation_items = set(
            task_assignment.loc[task_assignment["pool"] == "evaluation", "item_uid"].astype(str)
        )
        targets = observations[
            (observations["task_id"].astype(str) == str(task_id))
            & (observations["item_uid"].astype(str).isin(evaluation_items))
        ]
        if targets.empty:
            continue

        label_ids = acquisition["label_observation_id"].astype(str).tolist()
        for budget in budgets:
            selected_ids = label_ids[: min(int(budget), len(label_ids))]
            context = (
                observation_lookup.loc[selected_ids].copy()
                if selected_ids
                else observations.iloc[0:0]
            )
            if not context.empty and getattr(context, "ndim", 2) == 1:
                context = context.to_frame().T
            yield {
                "task_id": str(task_id),
                "budget": int(budget),
                "context": context,
                "target": targets,
            }
