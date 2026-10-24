#!/usr/bin/env python3
"""Per-benchmark error analysis with within-benchmark task clustering.

The uniform ablation tables report one number per arm averaged over all
validation tasks. That hides where the error actually lives: a benchmark where
the forecaster is badly miscalibrated can be cancelled out by one where it is
fine. This script breaks the same ``-rows.parquet`` artifacts down by benchmark
and, inside each benchmark, clusters tasks by their budget-wise error profile
so that systematic failure modes are visible instead of averaged away.

Three levels of readout:

1. **Per benchmark.** Brier ALC, ECE ALC, and mean AUC per arm, with the paired
   delta against a baseline arm and a task bootstrap interval.
2. **Per benchmark x budget.** Where on the budget curve a benchmark goes wrong
   -- a benchmark that is only bad at budget 0 is a prior problem, one that is
   bad at budget 31 is a modelling problem.
3. **Within benchmark.** Tasks clustered on their per-budget calibrated Brier
   profile, so distinct failure modes inside one benchmark separate out. The
   worst individual tasks are listed per benchmark.

Clustering uses k-means over the length-6 budget profile, with k chosen per
benchmark as ``min(max_clusters, n_tasks // min_cluster_size)``. The profile is
the quantity being predicted, so clusters group tasks that fail the same way,
not tasks that merely look similar in feature space.

Inputs are the artifacts written by ``modal_cnp_ladder.py`` (and any other app
emitting the same schema):

    arm, split, fold, benchmark_id, task_id, order_seed, budget,
    brier_raw, brier_cal, ece_raw, ece_cal, auc_raw, auc_cal

Usage:

    python scripts/analyze_benchmark_errors.py /path/to/*-rows.parquet
    python scripts/analyze_benchmark_errors.py rows/*.parquet --split test
    python scripts/analyze_benchmark_errors.py rows/*.parquet --arm cnp_full \\
        --baseline beta --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
METRICS = ("brier_raw", "brier_cal", "ece_raw", "ece_cal", "auc_raw", "auc_cal")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261022


def load_rows(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = sorted(
            {"arm", "split", "benchmark_id", "task_id", "order_seed", "budget", *METRICS}
            - set(frame.columns)
        )
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frame["source"] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("no input files")
    return pd.concat(frames, ignore_index=True)


def task_alc(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one weighted ALC row per (source, arm, benchmark, task).

    Orders are averaged within a task, matching the uniform protocol: the task
    is the independent unit, not the task-order pair.
    """

    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame = frame.copy()
    frame["weight"] = frame["budget"].map(weight)
    if frame["weight"].isna().any():
        unknown = sorted(set(frame.loc[frame["weight"].isna(), "budget"]))
        raise ValueError(f"budgets outside the official grid: {unknown}")

    for metric in METRICS:
        frame[f"w_{metric}"] = frame["weight"] * frame[metric]

    keys = ["source", "arm", "benchmark_id", "task_id"]
    per_order = frame.groupby([*keys, "order_seed"], sort=False)[
        [f"w_{metric}" for metric in METRICS]
    ].sum()
    collapsed = per_order.groupby(keys, sort=False).mean().reset_index()
    return collapsed.rename(
        columns={f"w_{metric}": f"alc_{metric}" for metric in METRICS}
    )


def budget_profile(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per task, one column per budget, averaged over orders."""

    profile = (
        frame.groupby(["arm", "benchmark_id", "task_id", "budget"], sort=True)[metric]
        .mean()
        .unstack("budget")
    )
    profile.columns = [f"budget_{int(column)}" for column in profile.columns]
    return profile.reset_index()


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        point = float(values.mean()) if len(values) else float("nan")
        return {"mean": point, "ci_low": point, "ci_high": point, "tasks": len(values)}
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[index].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "tasks": len(values),
    }


def benchmark_table(
    alc: pd.DataFrame, baseline: str, metric: str, draws: int
) -> list[dict[str, Any]]:
    """Per-benchmark ALC per arm, with a paired delta against the baseline."""

    column = f"alc_{metric}"
    rows: list[dict[str, Any]] = []
    baseline_lookup = (
        alc[alc["arm"] == baseline].set_index(["benchmark_id", "task_id"])[column]
        if baseline in set(alc["arm"])
        else None
    )

    for (benchmark, arm), group in alc.groupby(["benchmark_id", "arm"], sort=True):
        entry: dict[str, Any] = {
            "benchmark_id": benchmark,
            "arm": arm,
            "tasks": int(group["task_id"].nunique()),
            "alc": float(group[column].mean()),
        }
        for other in METRICS:
            entry[f"alc_{other}"] = float(group[f"alc_{other}"].mean())

        if baseline_lookup is not None and arm != baseline:
            paired = group.set_index(["benchmark_id", "task_id"])[column]
            shared = paired.index.intersection(baseline_lookup.index)
            if len(shared):
                delta = (paired.loc[shared] - baseline_lookup.loc[shared]).to_numpy()
                summary = _bootstrap(delta, draws, BOOTSTRAP_SEED)
                entry["delta_vs_baseline"] = summary["mean"]
                entry["delta_ci_low"] = summary["ci_low"]
                entry["delta_ci_high"] = summary["ci_high"]
                entry["wins"] = int((delta < 0.0).sum())
                entry["verdict"] = (
                    "better"
                    if summary["ci_high"] < 0.0
                    else "worse"
                    if summary["ci_low"] > 0.0
                    else "tie"
                )
        rows.append(entry)
    return rows


def budget_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per benchmark x arm x budget means, orders averaged within task first."""

    per_task = frame.groupby(
        ["arm", "benchmark_id", "task_id", "budget"], sort=True
    )[list(METRICS)].mean()
    grouped = per_task.groupby(["arm", "benchmark_id", "budget"], sort=True)
    table = grouped.mean().reset_index()
    table["tasks"] = grouped.size().to_numpy()
    return table.to_dict(orient="records")


def _kmeans(points: np.ndarray, k: int, seed: int, iterations: int = 100):
    """Small deterministic k-means++ to avoid a scikit-learn dependency."""

    rng = np.random.default_rng(seed)
    n = len(points)
    centres = [points[rng.integers(n)]]
    for _ in range(1, k):
        distance = np.min(
            [((points - centre) ** 2).sum(axis=1) for centre in centres], axis=0
        )
        total = distance.sum()
        if total <= 0:
            centres.append(points[rng.integers(n)])
            continue
        centres.append(points[rng.choice(n, p=distance / total)])
    centroids = np.array(centres, dtype=float)

    assignment = np.zeros(n, dtype=int)
    for _ in range(iterations):
        distance = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        updated = distance.argmin(axis=1)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for index in range(k):
            member = points[assignment == index]
            if len(member):
                centroids[index] = member.mean(axis=0)
    return assignment, centroids


def cluster_within_benchmark(
    profile: pd.DataFrame,
    arm: str,
    *,
    max_clusters: int,
    min_cluster_size: int,
    seed: int,
    baseline_alc: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Cluster tasks inside each benchmark on their budget-wise error profile.

    ``baseline_alc`` is the baseline arm's ALC indexed by task_id. Reporting it
    per cluster answers the question the aggregate number cannot: does the model
    win on the clusters that are *hard* for the baseline, or only on the ones
    that were already easy?
    """

    budget_columns = [column for column in profile.columns if column.startswith("budget_")]
    selected = profile[profile["arm"] == arm]
    output: list[dict[str, Any]] = []

    def _baseline_for(task_ids) -> float:
        if baseline_alc is None:
            return float("nan")
        shared = baseline_alc.index.intersection(pd.Index(task_ids))
        return float(baseline_alc.loc[shared].mean()) if len(shared) else float("nan")

    for benchmark, group in selected.groupby("benchmark_id", sort=True):
        points = group[budget_columns].to_numpy(dtype=float)
        points = np.nan_to_num(points, nan=float(np.nanmean(points)) if points.size else 0.0)
        n_tasks = len(group)
        k = min(max_clusters, max(1, n_tasks // min_cluster_size))
        if n_tasks < 2 or k < 2:
            output.append(
                {
                    "benchmark_id": benchmark,
                    "cluster": 0,
                    "tasks": int(n_tasks),
                    "mean_alc": float(np.average(points.mean(axis=1))) if points.size else float("nan"),
                    "baseline_alc": _baseline_for(group["task_id"]),
                    "profile": [float(value) for value in points.mean(axis=0)]
                    if points.size
                    else [],
                    "note": "too few tasks to cluster",
                    "example_tasks": group["task_id"].head(3).tolist(),
                }
            )
            continue

        assignment, centroids = _kmeans(points, k, seed)
        weights = np.asarray(WEIGHTS[: len(budget_columns)], dtype=float)
        for index in range(k):
            member = assignment == index
            if not member.any():
                continue
            member_points = points[member]
            member_tasks = group.loc[member, "task_id"]
            alc = float((member_points * weights).sum(axis=1).mean())
            worst = (
                group.loc[member]
                .assign(alc=(member_points * weights).sum(axis=1))
                .nlargest(3, "alc")["task_id"]
                .tolist()
            )
            output.append(
                {
                    "benchmark_id": benchmark,
                    "cluster": index,
                    "tasks": int(member.sum()),
                    "mean_alc": alc,
                    "baseline_alc": _baseline_for(member_tasks),
                    "profile": [float(value) for value in centroids[index]],
                    "budget_columns": budget_columns,
                    "worst_tasks": worst,
                    "example_tasks": member_tasks.head(3).tolist(),
                }
            )
    return output


def worst_tasks(alc: pd.DataFrame, arm: str, metric: str, top: int) -> list[dict[str, Any]]:
    column = f"alc_{metric}"
    selected = alc[alc["arm"] == arm]
    rows = []
    for benchmark, group in selected.groupby("benchmark_id", sort=True):
        for record in group.nlargest(top, column).itertuples(index=False):
            rows.append(
                {
                    "benchmark_id": benchmark,
                    "task_id": record.task_id,
                    "alc": float(getattr(record, column)),
                }
            )
    return rows


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _format(value: float, width: int = 9) -> str:
    if _is_missing(value):
        return "n/a".rjust(width)
    return f"{value:>{width}.6f}"


def print_report(report: dict[str, Any], metric: str, baseline: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"PER-BENCHMARK ALC  ({metric}, split={report['split']}, baseline={baseline})")
    print("=" * 78)
    table = pd.DataFrame(report["benchmarks"])
    if table.empty:
        print("no rows")
        return
    for benchmark, group in table.groupby("benchmark_id", sort=True):
        tasks = int(group["tasks"].iloc[0])
        print(f"\n{benchmark}  ({tasks} tasks)")
        print(f"  {'arm':<26}{'ALC':>10}{'delta':>11}{'95% CI':>24}  verdict")
        for record in group.sort_values("alc").itertuples(index=False):
            delta = getattr(record, "delta_vs_baseline", float("nan"))
            low = getattr(record, "delta_ci_low", float("nan"))
            high = getattr(record, "delta_ci_high", float("nan"))
            interval = "" if _is_missing(low) else f"[{low:+.6f}, {high:+.6f}]"
            shown_delta = "" if _is_missing(delta) else f"{delta:+.6f}"
            verdict = "baseline" if record.arm == baseline else getattr(record, "verdict", "")
            if _is_missing(verdict):
                verdict = ""
            print(
                f"  {record.arm:<26}{_format(record.alc, 10)}"
                f"{shown_delta:>11}{interval:>24}  {verdict}"
            )

    print(f"\n{'=' * 78}")
    print(f"WITHIN-BENCHMARK CLUSTERS  (arm={report['cluster_arm']}, {metric} by budget)")
    print("=" * 78)
    for entry in report["clusters"]:
        label = f"{entry['benchmark_id']}/c{entry['cluster']}"
        profile = " ".join(f"{value:.3f}" for value in entry["profile"])
        note = entry.get("note", "")
        baseline_value = entry.get("baseline_alc", float("nan"))
        if _is_missing(baseline_value):
            comparison = ""
        else:
            delta = entry["mean_alc"] - baseline_value
            comparison = f"  base={baseline_value:.6f}  delta={delta:+.6f}"
        print(
            f"  {label:<28}{entry['tasks']:>4} tasks  ALC={entry['mean_alc']:.6f}"
            f"{comparison}  profile=[{profile}] {note}"
        )
        if entry.get("worst_tasks"):
            print(f"      worst: {', '.join(entry['worst_tasks'][:2])}")

    print(f"\n{'=' * 78}")
    print(f"WORST TASKS PER BENCHMARK  (arm={report['cluster_arm']})")
    print("=" * 78)
    for record in report["worst_tasks"]:
        print(
            f"  {record['benchmark_id']:<20}{record['alc']:>10.6f}  {record['task_id']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="*-rows.parquet artifacts")
    parser.add_argument("--split", default="validation", help="split to analyze")
    parser.add_argument(
        "--metric",
        default="brier_cal",
        choices=METRICS,
        help="metric driving the ALC, delta, and clustering",
    )
    parser.add_argument("--baseline", default="beta", help="arm for paired deltas")
    parser.add_argument(
        "--arm", default="cnp_full", help="arm to cluster and list worst tasks for"
    )
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=6)
    parser.add_argument("--top", type=int, default=3, help="worst tasks per benchmark")
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    arguments = parser.parse_args()

    paths: list[Path] = []
    for path in arguments.paths:
        paths.extend(sorted(path.parent.glob(path.name)) if "*" in path.name else [path])
    frame = load_rows(paths)

    frame = frame[frame["split"] == arguments.split]
    if frame.empty:
        print(f"no rows for split={arguments.split}", file=sys.stderr)
        return 1

    available = sorted(frame["arm"].unique())
    cluster_arm = arguments.arm if arguments.arm in available else available[0]
    if cluster_arm != arguments.arm:
        print(
            f"warning: arm {arguments.arm!r} absent; clustering {cluster_arm!r} instead",
            file=sys.stderr,
        )

    alc = task_alc(frame)
    profile = budget_profile(frame, arguments.metric)
    baseline_rows = alc[alc["arm"] == arguments.baseline]
    baseline_alc = (
        baseline_rows.set_index("task_id")[f"alc_{arguments.metric}"]
        if not baseline_rows.empty
        else None
    )

    report = {
        "split": arguments.split,
        "metric": arguments.metric,
        "baseline": arguments.baseline,
        "cluster_arm": cluster_arm,
        "arms": available,
        "sources": sorted(frame["source"].unique()),
        "benchmarks": benchmark_table(
            alc, arguments.baseline, arguments.metric, arguments.draws
        ),
        "by_budget": budget_table(frame),
        "clusters": cluster_within_benchmark(
            profile,
            cluster_arm,
            max_clusters=arguments.max_clusters,
            min_cluster_size=arguments.min_cluster_size,
            seed=arguments.seed,
            baseline_alc=baseline_alc,
        ),
        "worst_tasks": worst_tasks(alc, cluster_arm, arguments.metric, arguments.top),
    }

    print_report(report, arguments.metric, arguments.baseline)
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
