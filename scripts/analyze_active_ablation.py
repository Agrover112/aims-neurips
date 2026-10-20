#!/usr/bin/env python3
"""Aggregate the active-acquisition ablation at the task level.

The Modal simulator writes one row per arm, task, stream, and budget. Streams
are repeated measures, so the task is the independent unit. This script first
averages streams within a task, then reports mean, standard deviation, and
standard error across the 104 validation tasks. Paired deltas use the same
task-level averages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)

STUDY = {
    "β-R": "Priority-1: Beta predictor",
    "β-Uβ": "Priority-1: Beta predictor",
    "β-Vβ": "Priority-1: Beta predictor",
    "C-R": "Priority-1: CNP predictor",
    "C-UC": "Priority-1: CNP predictor",
    "C-Uβ": "Priority-2: decoupled acquisition",
    "C-Vβ": "Priority-2: decoupled acquisition",
    "C-D": "Priority-2: decoupled acquisition",
}


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "sd": sd,
        "se": float(sd / np.sqrt(n)) if n > 1 else 0.0,
    }


def _paired(frame: pd.DataFrame, arm: str, baseline: str) -> dict[str, Any] | None:
    left = frame[frame["arm"] == arm].set_index("task_id")["brier_alc"]
    right = frame[frame["arm"] == baseline].set_index("task_id")["brier_alc"]
    joined = pd.concat([left.rename("arm"), right.rename("baseline")], axis=1, join="inner")
    if joined.empty:
        return None
    delta = (joined["arm"] - joined["baseline"]).to_numpy(dtype=float)
    stats = _summary(delta)
    return {
        "arm": arm,
        "baseline": baseline,
        "delta_mean": stats["mean"],
        "delta_sd": stats["sd"],
        "delta_se": stats["se"],
        "wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "n": stats["n"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_parquet(args.rows)
    frame["weight"] = frame["budget"].map(dict(zip(BUDGETS, WEIGHTS, strict=True)))
    frame["weighted_brier"] = frame["brier"] * frame["weight"]
    frame["weighted_ece"] = frame["ece"] * frame["weight"]

    per_run = (
        frame.groupby(["arm", "task_id", "stream_seed"], sort=False)
        .agg(brier_alc=("weighted_brier", "sum"), ece_alc=("weighted_ece", "sum"))
        .reset_index()
    )
    per_task = (
        per_run.groupby(["arm", "task_id"], sort=False)
        .agg(brier_alc=("brier_alc", "mean"), ece_alc=("ece_alc", "mean"))
        .reset_index()
    )

    summary_rows: list[dict[str, Any]] = []
    for arm, group in per_task.groupby("arm", sort=False):
        brier = _summary(group["brier_alc"].to_numpy())
        ece = _summary(group["ece_alc"].to_numpy())
        labels_31 = frame[(frame["arm"] == arm) & (frame["budget"] == 31)][
            "labels_acquired"
        ].mean()
        summary_rows.append(
            {
                "arm": arm,
                "study": STUDY.get(arm, ""),
                "brier_alc_mean": brier["mean"],
                "brier_alc_sd": brier["sd"],
                "brier_alc_se": brier["se"],
                "ece_alc_mean": ece["mean"],
                "ece_alc_sd": ece["sd"],
                "ece_alc_se": ece["se"],
                "labels_acquired_31": float(labels_31),
                "tasks": brier["n"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("brier_alc_mean").reset_index(drop=True)

    deltas: list[dict[str, Any]] = []
    for arm in summary["arm"]:
        predictor = "beta" if arm.startswith("β") else "cnp"
        random_baseline = "β-R" if predictor == "beta" else "C-R"
        deployed_baseline = "β-Uβ" if predictor == "beta" else "C-UC"
        for label, baseline in (
            ("random", random_baseline),
            ("deployed", deployed_baseline),
            ("beta_active", "β-Uβ"),
        ):
            result = _paired(per_task, arm, baseline)
            if result is not None:
                result["comparison"] = label
                deltas.append(result)

    by_budget_rows: list[dict[str, Any]] = []
    for (arm, budget), group in frame.groupby(["arm", "budget"], sort=False):
        task_means = group.groupby("task_id", sort=False)["brier"].mean().to_numpy()
        ece_means = group.groupby("task_id", sort=False)["ece"].mean().to_numpy()
        brier = _summary(task_means)
        ece = _summary(ece_means)
        labels = group.groupby("task_id", sort=False)["labels_acquired"].mean()
        by_budget_rows.append(
            {
                "arm": arm,
                "budget": int(budget),
                "brier_mean": brier["mean"],
                "brier_se": brier["se"],
                "ece_mean": ece["mean"],
                "ece_se": ece["se"],
                "labels_acquired_mean": float(labels.mean()),
            }
        )

    acquisition_rows: list[dict[str, Any]] = []
    for (arm, budget), group in frame.groupby(["arm", "budget"], sort=False):
        rank = group.groupby("task_id", sort=False)["mean_selected_rank"].mean().to_numpy()
        length = group.groupby("task_id", sort=False)["stream_length"].mean().to_numpy()
        fraction = np.divide(rank, np.maximum(length - 1.0, 1.0))
        acquisition_rows.append(
            {
                "arm": arm,
                "budget": int(budget),
                "mean_selected_rank": float(rank.mean()),
                "mean_stream_position": float(fraction.mean()),
                "mean_stream_length": float(length.mean()),
            }
        )

    artifact = {
        "status": "active_acquisition_ablation_task_level",
        "source_rows": str(args.rows),
        "summary": summary.to_dict(orient="records"),
        "paired_deltas": deltas,
        "by_budget": by_budget_rows,
        "acquisition": acquisition_rows,
    }
    tasks = int(summary["tasks"].iloc[0]) if not summary.empty else 0
    print(f"\nTask-level summary ({tasks} tasks; streams averaged within task)\n")
    print("| arm | study | Brier ALC | ECE ALC |")
    print("|---|---|---:|---:|")
    for row in summary.itertuples(index=False):
        print(
            f"| {row.arm} | {row.study} | {row.brier_alc_mean:.5f} "
            f"({row.brier_alc_se:.5f}) | {row.ece_alc_mean:.5f} ({row.ece_alc_se:.5f}) |"
        )
    print("\nPaired task-level deltas (negative is better)\n")
    print("| arm | comparison | delta | wins | tasks |")
    print("|---|---|---:|---:|---:|")
    for row in deltas:
        print(
            f"| {row['arm']} | {row['comparison']} | {row['delta_mean']:+.5f} "
            f"({row['delta_se']:.5f}) | {row['wins']}/{row['n']} | {row['n']} |"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
