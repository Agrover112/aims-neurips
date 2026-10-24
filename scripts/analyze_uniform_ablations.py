#!/usr/bin/env python3
"""Aggregate the uniform validation-grid ablations into one audited table.

Every input row is a per-task, per-order, per-budget score on the same 104
validation tasks.  This script checks the task grid, computes weighted ALC,
and forms paired task-level deltas.  It keeps the task bootstrap and the
benchmark-cluster bootstrap in separate fields; the benchmark interval is a
sensitivity check, never the primary interval.

Expected local layout after downloading the Modal artifacts:

    /tmp/opencode/uniform-v1/ladder/*-rows.parquet
    /tmp/opencode/uniform-v1/calibration/*-rows.parquet
    /tmp/opencode/uniform-v1/disjoint/*-rows.parquet
    /tmp/opencode/uniform-v1/acquisition/*-rows.parquet
    /tmp/opencode/uniform-v1/budget-shrinkage/summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aims_starter.uniform_eval import BUDGETS, WEIGHTS, task_bootstrap

ROOT = Path("/tmp/opencode/uniform-v1")
OUTPUT = ROOT / "analysis"
EVAL_ITEMS = Path("/tmp/opencode/allbench-features-v1/eval_items.parquet")
SELECTED_SEED = 20261018
SELECTED_MODE = "masked"
WEIGHT = dict(zip(BUDGETS, WEIGHTS, strict=True))
DRAWS = 10_000
BOOTSTRAP_SEED = 20261020
EXPECTED_TASKS = 104
EXPECTED_ORDERS = tuple(range(10))
EXPECTED_BUDGETS = tuple(BUDGETS)


def _benchmark_map(split: str) -> dict[str, str]:
    frame = pd.read_parquet(EVAL_ITEMS)
    frame = frame[frame["split"] == split]
    frame = (
        frame[["task_id", "benchmark_id"]]
        .drop_duplicates()
        .assign(task_id=lambda value: value["task_id"].astype(str))
    )
    return dict(zip(frame["task_id"], frame["benchmark_id"], strict=True))


BENCHMARK = {
    "validation": _benchmark_map("validation"),
    "test": _benchmark_map("test"),
}


def _config(path: Path) -> tuple[int, str]:
    match = re.search(r"-(2026\d+)-(masked|visible)", path.stem)
    if not match:
        raise ValueError(f"cannot parse seed/mode from {path.name}")
    return int(match.group(1)), match.group(2)


def _load(pattern: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(pattern.parent.glob(pattern.name)):
        frame = pd.read_parquet(path)
        seed, mode = _config(path)
        frame["seed"] = seed
        frame["mode"] = mode
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no {name} rows")
    return pd.concat(frames, ignore_index=True)


def load_ladder() -> pd.DataFrame:
    return _load(ROOT / "ladder" / "cnp-ladder-*-rows.parquet", "ladder")


def load_calibration() -> pd.DataFrame:
    return _load(ROOT / "calibration" / "*-rows.parquet", "calibration")


def load_disjoint() -> pd.DataFrame:
    return _load(ROOT / "disjoint" / "disjoint-*-rows.parquet", "disjoint")


def load_acquisition() -> pd.DataFrame:
    frames = []
    for path in sorted((ROOT / "acquisition").glob("seed-*-rows.parquet")):
        frame = pd.read_parquet(path)
        match = re.search(r"seed-(2026\d+)-rows", path.name)
        if not match:
            raise ValueError(f"cannot parse seed from {path.name}")
        frame["seed"] = int(match.group(1))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no acquisition rows")
    return pd.concat(frames, ignore_index=True)


def brier_task_alc(frame: pd.DataFrame, value: str) -> pd.Series:
    """Weighted ALC per task, averaging the frozen orders within a task."""

    working = frame.copy()
    working["weighted"] = working["budget"].map(WEIGHT) * working[value]
    per_order = working.groupby(["task_id", "order_seed"], sort=False)["weighted"].sum()
    return per_order.groupby("task_id", sort=False).mean()


def weighted_alc_task_alc(frame: pd.DataFrame, value: str) -> pd.Series:
    """ALC per task when the input column is already budget-weighted."""

    per_order = frame.groupby(["task_id", "order_seed"], sort=False)[value].sum()
    return per_order.groupby("task_id", sort=False).mean()


def _benchmark_bootstrap(delta: pd.Series, split: str) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "task_id": delta.index.astype(str),
            "delta": delta.to_numpy(dtype=float),
        }
    )
    frame["benchmark"] = frame["task_id"].map(BENCHMARK[split])
    groups = {
        benchmark: group["delta"].to_numpy(dtype=float)
        for benchmark, group in frame.groupby("benchmark", sort=True)
    }
    names = list(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(DRAWS, dtype=float)
    for index in range(DRAWS):
        sampled = rng.choice(names, size=len(names), replace=True)
        means[index] = float(np.concatenate([groups[name] for name in sampled]).mean())
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "benchmarks": int(len(names)),
    }


def paired_delta(first: pd.Series, second: pd.Series, *, split: str) -> dict[str, Any]:
    """Paired task-level difference with both uncertainty summaries."""

    joined = pd.concat([first.rename("first"), second.rename("second")], axis=1, join="inner")
    if joined.empty:
        raise RuntimeError("paired comparison has no shared tasks")
    delta = joined["first"] - joined["second"]
    task = task_bootstrap(delta.to_numpy(dtype=float), draws=DRAWS, seed=BOOTSTRAP_SEED)
    benchmark = _benchmark_bootstrap(delta, split)
    return {
        "mean": float(delta.mean()),
        "task_ci_low": float(task["ci_low"]),
        "task_ci_high": float(task["ci_high"]),
        "benchmark_ci_low": float(benchmark["ci_low"]),
        "benchmark_ci_high": float(benchmark["ci_high"]),
        "wins": int((delta < 0.0).sum()),
        "tasks": int(len(delta)),
        "benchmarks": int(benchmark["benchmarks"]),
        "draws": int(DRAWS),
        "first_mean": float(joined["first"].mean()),
        "second_mean": float(joined["second"].mean()),
    }


def task_set(frame: pd.DataFrame) -> set[str]:
    return set(frame["task_id"].astype(str))


def select(frame: pd.DataFrame, seed: int, mode: str) -> pd.DataFrame:
    return frame[(frame["seed"] == seed) & (frame["mode"] == mode)].copy()


def config_series(frame: pd.DataFrame, seed: int, mode: str, value: str) -> pd.Series:
    return brier_task_alc(select(frame, seed, mode), value)


def mean_config_series(frame: pd.DataFrame, value: str) -> pd.Series:
    series = [
        brier_task_alc(group, value)
        for _, group in frame.groupby(["seed", "mode"], sort=True)
    ]
    indexes = [set(series_item.index.astype(str)) for series_item in series]
    if any(index != indexes[0] for index in indexes[1:]):
        raise RuntimeError("seed/mode configurations do not share one task set")
    return pd.concat(series, axis=1).mean(axis=1)


def _check_grid(
    frame: pd.DataFrame,
    name: str,
    *,
    split: str | None = None,
    expected_tasks: int = EXPECTED_TASKS,
) -> dict[str, Any]:
    working = frame.copy()
    if split is not None:
        working = working[working["split"] == split]
    tasks = task_set(working)
    if len(tasks) != expected_tasks:
        raise RuntimeError(f"{name}: expected {expected_tasks} tasks, found {len(tasks)}")
    orders = tuple(sorted(working["order_seed"].unique().tolist()))
    if orders != EXPECTED_ORDERS:
        raise RuntimeError(f"{name}: orders {orders}")
    budgets = tuple(sorted(working["budget"].unique().tolist()))
    if budgets != EXPECTED_BUDGETS:
        raise RuntimeError(f"{name}: budgets {budgets}")
    key = ["seed", "mode", "task_id", "order_seed", "budget"]
    for column in ("arm", "model", "variant"):
        if column in working.columns:
            key.append(column)
    if working.duplicated(key).any():
        raise RuntimeError(f"{name}: duplicate rows on {key}")
    per_task = (
        working.groupby("task_id", sort=False)
        .agg(orders=("order_seed", "nunique"), budgets=("budget", "nunique"))
        .reset_index()
    )
    if not (per_task["orders"] == len(EXPECTED_ORDERS)).all():
        raise RuntimeError(f"{name}: some task lacks an order")
    if not (per_task["budgets"] == len(EXPECTED_BUDGETS)).all():
        raise RuntimeError(f"{name}: some task lacks a budget")
    folds = working.groupby("task_id", sort=False)["fold"].nunique()
    if not (folds == 1).all():
        raise RuntimeError(f"{name}: a task has more than one fold")
    if not set(working["fold"].unique()).issubset({-1, 0, 1, 2, 3, 4}):
        raise RuntimeError(f"{name}: unexpected fold label")
    return {
        "rows": int(len(working)),
        "tasks": int(len(tasks)),
        "orders": int(len(orders)),
        "budgets": list(budgets),
        "folds": sorted(int(value) for value in working["fold"].unique()),
        "duplicate_rows": int(working.duplicated(key).sum()),
    }


def _check_calibration_folds(frame: pd.DataFrame) -> dict[str, Any]:
    validation = frame[frame["split"] == "validation"]
    checks = {}
    for (seed, mode, model, variant), group in validation.groupby(
        ["seed", "mode", "model", "variant"], sort=True
    ):
        key = ["task_id", "order_seed", "budget"]
        duplicates = int(group.duplicated(key).sum())
        if duplicates:
            raise RuntimeError(
                f"calibration {seed}/{mode}/{model}/{variant}: {duplicates} duplicates"
            )
        folds = group.groupby("task_id", sort=False)["fold"].nunique()
        if not (folds == 1).all():
            raise RuntimeError(
                f"calibration {seed}/{mode}/{model}/{variant}: task fold drift"
            )
        if set(group["fold"].unique()) != {0, 1, 2, 3, 4}:
            raise RuntimeError(f"calibration {seed}/{mode}/{model}/{variant}: missing fold")
        checks[f"{seed}/{mode}/{model}/{variant}"] = {
            "rows": int(len(group)),
            "tasks": int(group["task_id"].nunique()),
            "folds": sorted(int(value) for value in group["fold"].unique()),
        }
    if not (frame[frame["split"] == "test"]["fold"] == -1).all():
        raise RuntimeError("calibration test rows do not carry fold -1")
    return checks


def _check_acquisition(frame: pd.DataFrame) -> dict[str, Any]:
    tasks = task_set(frame)
    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"acquisition: expected {EXPECTED_TASKS} tasks, found {len(tasks)}")
    arms = sorted(frame["arm"].unique())
    if arms != ["C-D", "C-F", "C-R", "C-U", "beta-R"]:
        raise RuntimeError(f"acquisition: unexpected arms {arms}")
    budgets = tuple(sorted(frame["budget"].unique().tolist()))
    if budgets != EXPECTED_BUDGETS:
        raise RuntimeError(f"acquisition: budgets {budgets}")
    streams = tuple(sorted(frame["stream_seed"].unique().tolist()))
    if streams != EXPECTED_ORDERS:
        raise RuntimeError(f"acquisition: streams {streams}")
    for (seed, arm), group in frame.groupby(["seed", "arm"], sort=True):
        if set(group["task_id"].astype(str)) != tasks:
            raise RuntimeError(f"acquisition seed {seed} arm {arm}: task set differs")
        if group["stream_seed"].nunique() != len(streams):
            raise RuntimeError(f"acquisition seed {seed} arm {arm}: stream count differs")
    # At budget zero no arm has revealed a label, so the four CNP arms must
    # produce the same target predictions.  Beta-R uses a different predictor.
    zero = frame[(frame["budget"] == 0) & (frame["arm"] != "beta-R")]
    spread = zero.groupby(["seed", "task_id", "stream_seed"])["brier"].agg(np.ptp)
    if float(spread.max()) > 1e-12:
        raise RuntimeError(f"acquisition budget-0 arms differ: {float(spread.max())}")
    return {
        "rows": int(len(frame)),
        "tasks": int(len(tasks)),
        "streams": list(streams),
        "arms": arms,
        "budget_zero_max_spread": float(spread.max()),
    }


def _summarize_ladder(ladder: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    validation = ladder[ladder["split"] == "validation"].copy()
    selected = select(validation, SELECTED_SEED, SELECTED_MODE)
    beta = brier_task_alc(selected[selected["arm"] == "beta"], "brier_raw")
    if beta.empty:
        raise RuntimeError("Beta baseline is missing from the ladder")

    def rows_for(frame: pd.DataFrame, baseline: pd.Series, family: str) -> list[dict]:
        output = []
        for arm, group in frame.groupby("arm", sort=True):
            raw = brier_task_alc(group, "brier_raw")
            cal = brier_task_alc(group, "brier_cal")
            output.append(
                {
                    "family": family,
                    "arm": arm,
                    "brier_alc_raw": float(raw.mean()),
                    "brier_alc_cal": float(cal.mean()),
                    "delta_cal_vs_beta": paired_delta(cal, baseline, split="validation"),
                    "delta_raw_vs_beta": paired_delta(raw, baseline, split="validation"),
                }
            )
        return output

    selected_rows = rows_for(selected, beta, "encoder_ladder_selected")
    mean_beta = mean_config_series(validation[validation["arm"] == "beta"], "brier_raw")
    mean_rows = []
    for arm, group in validation.groupby("arm", sort=True):
        raw = mean_config_series(group, "brier_raw")
        cal = mean_config_series(group, "brier_cal")
        mean_rows.append(
            {
                "family": "encoder_ladder_mean",
                "arm": arm,
                "brier_alc_raw": float(raw.mean()),
                "brier_alc_cal": float(cal.mean()),
                "delta_cal_vs_beta": paired_delta(cal, mean_beta, split="validation"),
                "delta_raw_vs_beta": paired_delta(raw, mean_beta, split="validation"),
            }
        )
    return selected_rows, mean_rows


def _summarize_calibration(calibration: pd.DataFrame) -> dict[str, Any]:
    validation = calibration[calibration["split"] == "validation"].copy()
    selected = select(validation, SELECTED_SEED, SELECTED_MODE)
    output: dict[str, Any] = {"selected": {}, "mean": {}, "by_budget": {}}
    for model in sorted(selected["model"].unique()):
        model_rows = selected[selected["model"] == model]
        baseline = brier_task_alc(model_rows[model_rows["variant"] == "full_bce"], "brier")
        rows = []
        for variant, group in model_rows.groupby("variant", sort=True):
            values = brier_task_alc(group, "brier")
            rows.append(
                {
                    "family": "calibration_selected",
                    "model": model,
                    "variant": variant,
                    "brier_alc": float(values.mean()),
                    "delta_vs_full_bce": paired_delta(values, baseline, split="validation"),
                }
            )
        output["selected"][model] = rows

    for model in sorted(validation["model"].unique()):
        def mean_values(variant: str) -> pd.Series:
            return pd.concat(
                [
                    brier_task_alc(group, "brier")
                    for _, group in validation[validation["model"] == model].groupby(
                        ["seed", "mode"], sort=True
                    )
                    for group in [group[group["variant"] == variant]]
                ],
                axis=1,
            ).mean(axis=1)

        baseline = mean_values("full_bce")
        rows = []
        for variant in sorted(validation["variant"].unique()):
            values = mean_values(variant)
            rows.append(
                {
                    "family": "calibration_mean",
                    "model": model,
                    "variant": variant,
                    "brier_alc": float(values.mean()),
                    "delta_vs_full_bce": paired_delta(values, baseline, split="validation"),
                }
            )
        output["mean"][model] = rows

    key_variants = (
        "raw",
        "context",
        "anchor_bce",
        "full_bce",
        "full_brier",
        "full_brier_task",
    )
    for model in sorted(selected["model"].unique()):
        rows = []
        for variant in key_variants:
            group = selected[(selected["model"] == model) & (selected["variant"] == variant)]
            for budget, budget_group in group.groupby("budget", sort=True):
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "budget": int(budget),
                        "brier": float(budget_group["brier"].mean()),
                        "ece": float(budget_group["ece"].mean()),
                    }
                )
        output["by_budget"][model] = rows
    return output


def _summarize_disjoint(disjoint: pd.DataFrame, ladder: pd.DataFrame) -> dict[str, list[dict]]:
    validation = disjoint[disjoint["split"] == "validation"].copy()
    ladder_validation = ladder[ladder["split"] == "validation"].copy()
    selected = select(validation, SELECTED_SEED, SELECTED_MODE)
    selected_original = select(ladder_validation, SELECTED_SEED, SELECTED_MODE)
    beta = brier_task_alc(
        selected_original[selected_original["arm"] == "beta"], "brier_raw"
    )
    original = brier_task_alc(
        selected_original[selected_original["arm"] == "cnp_full"], "brier_cal"
    )
    output = {"selected": [], "mean": []}
    for column, arm, label in (
        ("calibrated_alc", "disjoint_cal", "calibrated"),
        ("raw_alc", "disjoint_raw", "raw"),
    ):
        values = weighted_alc_task_alc(selected, column)
        output["selected"].append(
            {
                "family": "disjoint_selected",
                "arm": arm,
                "brier_alc": float(values.mean()),
                "delta_vs_beta": paired_delta(values, beta, split="validation"),
                "delta_vs_original_cnp": paired_delta(
                    values, original if label == "calibrated" else brier_task_alc(
                        selected_original[selected_original["arm"] == "cnp_full"], "brier_raw"
                    ), split="validation"
                ),
            }
        )

    configs = [(seed, mode) for seed in (20261018, 20261019) for mode in ("masked", "visible")]
    mean_beta = mean_config_series(
        ladder_validation[ladder_validation["arm"] == "beta"], "brier_raw"
    )
    mean_original_cal = mean_config_series(
        ladder_validation[ladder_validation["arm"] == "cnp_full"], "brier_cal"
    )
    mean_original_raw = mean_config_series(
        ladder_validation[ladder_validation["arm"] == "cnp_full"], "brier_raw"
    )
    for column, arm in (("calibrated_alc", "disjoint_cal"), ("raw_alc", "disjoint_raw")):
        series = [
            weighted_alc_task_alc(select(validation, seed, mode), column)
            for seed, mode in configs
        ]
        values = pd.concat(series, axis=1).mean(axis=1)
        output["mean"].append(
            {
                "family": "disjoint_mean",
                "arm": arm,
                "brier_alc": float(values.mean()),
                "delta_vs_beta": paired_delta(values, mean_beta, split="validation"),
                "delta_vs_original_cnp": paired_delta(
                    values,
                    mean_original_cal if arm == "disjoint_cal" else mean_original_raw,
                    split="validation",
                ),
            }
        )
    return output


def _summarize_acquisition(acquisition: pd.DataFrame) -> list[dict]:
    work = acquisition.copy()
    work["weighted"] = work["budget"].map(WEIGHT) * work["brier"]
    per_run = (
        work.groupby(["seed", "arm", "task_id", "stream_seed"], sort=False)["weighted"]
        .sum()
        .reset_index()
    )
    rows = []
    for seed, group in per_run.groupby("seed", sort=True):
        per_task = (
            group.groupby(["arm", "task_id"], sort=False)["weighted"].mean().reset_index()
        )
        baseline = (
            per_task[per_task["arm"] == "C-R"][["task_id", "weighted"]]
            .rename(columns={"weighted": "baseline"})
        )
        for arm, arm_group in per_task.groupby("arm", sort=True):
            joined = arm_group.merge(baseline, on="task_id", how="inner")
            delta = pd.Series(
                (joined["weighted"] - joined["baseline"]).to_numpy(dtype=float),
                index=joined["task_id"].astype(str),
            )
            rows.append(
                {
                    "family": "acquisition",
                    "seed": int(seed),
                    "arm": arm,
                    "brier_alc": float(arm_group["weighted"].mean()),
                    "delta_vs_C_R": paired_delta(
                        pd.Series(
                            arm_group["weighted"].to_numpy(dtype=float),
                            index=arm_group["task_id"].astype(str),
                        ),
                        pd.Series(
                            baseline["baseline"].to_numpy(dtype=float),
                            index=baseline["task_id"].astype(str),
                        ),
                        split="validation",
                    ),
                }
            )
    # Pooled seed mean, for a compact robustness row.
    per_task_seed = (
        per_run.groupby(["seed", "arm", "task_id"], sort=False)["weighted"].mean().reset_index()
    )
    mean_per_task = (
        per_task_seed.groupby(["arm", "task_id"], sort=False)["weighted"].mean().reset_index()
    )
    baseline = (
        mean_per_task[mean_per_task["arm"] == "C-R"][["task_id", "weighted"]]
        .rename(columns={"weighted": "baseline"})
    )
    for arm, arm_group in mean_per_task.groupby("arm", sort=True):
        joined = arm_group.merge(baseline, on="task_id", how="inner")
        rows.append(
            {
                "family": "acquisition_seed_mean",
                "seed": "mean",
                "arm": arm,
                "brier_alc": float(arm_group["weighted"].mean()),
                "delta_vs_C_R": paired_delta(
                    pd.Series(
                        arm_group["weighted"].to_numpy(dtype=float),
                        index=arm_group["task_id"].astype(str),
                    ),
                    pd.Series(
                        baseline["baseline"].to_numpy(dtype=float),
                        index=baseline["task_id"].astype(str),
                    ),
                    split="validation",
                ),
            }
        )
    return rows


def _summarize_final_test(ladder: pd.DataFrame) -> dict[str, Any]:
    test = ladder[ladder["split"] == "test"].copy()
    selected = select(test, SELECTED_SEED, SELECTED_MODE)
    beta = brier_task_alc(selected[selected["arm"] == "beta"], "brier_raw")
    output: dict[str, Any] = {"selected": [], "two_seed_masked": []}
    for arm in ("beta", "cnp_full"):
        group = selected[selected["arm"] == arm]
        raw = brier_task_alc(group, "brier_raw")
        cal = brier_task_alc(group, "brier_cal")
        output["selected"].append(
            {
                "family": "final_test",
                "arm": arm,
                "brier_alc_raw": float(raw.mean()),
                "brier_alc_cal": float(cal.mean()),
                "delta_cal_vs_beta": paired_delta(cal, beta, split="test"),
                "delta_raw_vs_beta": paired_delta(raw, beta, split="test"),
            }
        )
    mean_beta = pd.concat(
        [
            brier_task_alc(
                test[(test["seed"] == seed) & (test["mode"] == "masked") & (test["arm"] == "beta")],
                "brier_raw",
            )
            for seed in (20261018, 20261019)
        ],
        axis=1,
    ).mean(axis=1)
    for arm in ("beta", "cnp_full"):
        raw = pd.concat(
            [
                brier_task_alc(
                    test[
                        (test["seed"] == seed)
                        & (test["mode"] == "masked")
                        & (test["arm"] == arm)
                    ],
                    "brier_raw",
                )
                for seed in (20261018, 20261019)
            ],
            axis=1,
        ).mean(axis=1)
        cal = pd.concat(
            [
                brier_task_alc(
                    test[
                        (test["seed"] == seed)
                        & (test["mode"] == "masked")
                        & (test["arm"] == arm)
                    ],
                    "brier_cal",
                )
                for seed in (20261018, 20261019)
            ],
            axis=1,
        ).mean(axis=1)
        output["two_seed_masked"].append(
            {
                "family": "final_test_two_seed_masked",
                "arm": arm,
                "brier_alc_raw": float(raw.mean()),
                "brier_alc_cal": float(cal.mean()),
                "delta_cal_vs_beta": paired_delta(cal, mean_beta, split="test"),
                "delta_raw_vs_beta": paired_delta(raw, mean_beta, split="test"),
            }
        )
    return output


def _flatten_rows(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    output = []
    for row in rows:
        flat = {key: row[key] for key in keys}
        flat["brier_alc"] = row.get("brier_alc", row.get("brier_alc_cal"))
        for prefix in ("delta_cal_vs_beta", "delta_raw_vs_beta", "delta_vs_beta",
                       "delta_vs_full_bce", "delta_vs_original_cnp", "delta_vs_C_R"):
            if prefix in row and row[prefix] is not None:
                delta = row[prefix]
                flat[f"{prefix}_mean"] = delta["mean"]
                flat[f"{prefix}_task_ci_low"] = delta["task_ci_low"]
                flat[f"{prefix}_task_ci_high"] = delta["task_ci_high"]
                flat[f"{prefix}_benchmark_ci_low"] = delta["benchmark_ci_low"]
                flat[f"{prefix}_benchmark_ci_high"] = delta["benchmark_ci_high"]
                flat[f"{prefix}_wins"] = delta["wins"]
        output.append(flat)
    return output


def main() -> int:
    ladder = load_ladder()
    calibration = load_calibration()
    disjoint = load_disjoint()
    acquisition = load_acquisition()

    audit: dict[str, Any] = {
        "protocol": {
            "split": "validation",
            "tasks": EXPECTED_TASKS,
            "orders": len(EXPECTED_ORDERS),
            "budgets": list(EXPECTED_BUDGETS),
            "folds": 5,
            "selected_seed": SELECTED_SEED,
            "selected_mode": SELECTED_MODE,
            "metric": "Brier ALC",
        },
        "grids": {},
    }
    for name, frame in (("ladder", ladder), ("disjoint", disjoint)):
        for split in ("validation", "test"):
            audit["grids"][f"{name}/{split}"] = _check_grid(
                frame,
                f"{name}/{split}",
                split=split,
                expected_tasks=EXPECTED_TASKS if split == "validation" else 22,
            )
    # Calibration has both split values in every file and repeated rows per
    # variant/model, so validate its own key instead of the shared grid helper.
    audit["grids"]["calibration/validation"] = {
        "models": sorted(calibration["model"].unique().tolist()),
        "variants": sorted(calibration["variant"].unique().tolist()),
        "rows": int(len(calibration[calibration["split"] == "validation"])),
        "tasks": int(calibration[calibration["split"] == "validation"]["task_id"].nunique()),
    }
    audit["calibration_folds"] = _check_calibration_folds(calibration)
    audit["acquisition"] = _check_acquisition(acquisition)

    # All families must share the same validation task set.
    expected_tasks = task_set(
        ladder[(ladder["split"] == "validation") & (ladder["seed"] == SELECTED_SEED)
               & (ladder["mode"] == SELECTED_MODE)]
    )
    for name, values in (
        ("calibration", task_set(calibration[calibration["split"] == "validation"])),
        ("disjoint", task_set(disjoint[disjoint["split"] == "validation"])),
        ("acquisition", task_set(acquisition)),
    ):
        if values != expected_tasks:
            raise RuntimeError(f"{name} task set differs from ladder: {len(values)}")
    audit["shared_validation_tasks"] = len(expected_tasks)
    if len(expected_tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"expected {EXPECTED_TASKS} validation tasks")

    ladder_selected, ladder_mean = _summarize_ladder(ladder)
    calibration_summary = _summarize_calibration(calibration)
    disjoint_summary = _summarize_disjoint(disjoint, ladder)
    acquisition_summary = _summarize_acquisition(acquisition)
    final_test = _summarize_final_test(ladder)
    budget = json.loads((ROOT / "budget-shrinkage" / "summary.json").read_text())
    budget_row = {
        "family": "budget_shrinkage",
        "flat_alc": budget["summary"]["flat_alc"],
        "tuned_alc": budget["summary"]["tuned_alc"],
        "delta_vs_flat": budget["summary"]["delta"],
        "selected_kappa": budget["summary"]["selected_kappa"],
    }

    summary = {
        "protocol": audit["protocol"],
        "audit": audit,
        "encoder_ladder_selected": ladder_selected,
        "encoder_ladder_mean": ladder_mean,
        "calibration_selected": calibration_summary["selected"],
        "calibration_mean": calibration_summary["mean"],
        "calibration_by_budget": calibration_summary["by_budget"],
        "disjoint": disjoint_summary,
        "acquisition": acquisition_summary,
        "budget_shrinkage": budget_row,
        "final_test": final_test,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    pd.DataFrame(_flatten_rows(ladder_selected, ("family", "arm"))).to_csv(
        OUTPUT / "ladder_selected.csv", index=False
    )
    pd.DataFrame(_flatten_rows(ladder_mean, ("family", "arm"))).to_csv(
        OUTPUT / "ladder_mean.csv", index=False
    )
    for model, rows in calibration_summary["selected"].items():
        pd.DataFrame(_flatten_rows(rows, ("family", "model", "variant"))).to_csv(
            OUTPUT / f"calibration_selected_{model}.csv", index=False
        )
    pd.DataFrame(
        _flatten_rows(
            calibration_summary["selected"]["cnp_airt"],
            ("family", "model", "variant"),
        )
    ).to_csv(OUTPUT / "calibration_selected.csv", index=False)
    pd.DataFrame(_flatten_rows(disjoint_summary["selected"], ("family", "arm"))).to_csv(
        OUTPUT / "disjoint_selected.csv", index=False
    )
    pd.DataFrame(_flatten_rows(disjoint_summary["mean"], ("family", "arm"))).to_csv(
        OUTPUT / "disjoint_mean.csv", index=False
    )
    pd.DataFrame(_flatten_rows(acquisition_summary, ("family", "seed", "arm"))).to_csv(
        OUTPUT / "acquisition.csv", index=False
    )
    pd.DataFrame(_flatten_rows(final_test["selected"], ("family", "arm"))).to_csv(
        OUTPUT / "final_test_selected.csv", index=False
    )
    pd.DataFrame(
        _flatten_rows(final_test["two_seed_masked"], ("family", "arm"))
    ).to_csv(OUTPUT / "final_test_two_seed_masked.csv", index=False)
    pd.DataFrame([budget_row]).to_csv(OUTPUT / "budget_shrinkage.csv", index=False)

    # A compact console report; the JSON is the source of truth.
    print(json.dumps({"protocol": summary["protocol"], "audit": audit}, indent=2))
    for row in ladder_selected:
        delta = row["delta_cal_vs_beta"]
        print(
            f"ladder selected {row['arm']:24s} raw={row['brier_alc_raw']:.6f} "
            f"cal={row['brier_alc_cal']:.6f} "
            f"d={delta['mean']:+.6f} task[{delta['task_ci_low']:+.6f},{delta['task_ci_high']:+.6f}] "
            f"bench[{delta['benchmark_ci_low']:+.6f},{delta['benchmark_ci_high']:+.6f}]"
        )
    for model, rows in calibration_summary["selected"].items():
        for row in rows:
            delta = row["delta_vs_full_bce"]
            print(
                f"calibration {model:16s} {row['variant']:22s} {row['brier_alc']:.6f} "
                f"d_bce={delta['mean']:+.6f} "
                f"task[{delta['task_ci_low']:+.6f},{delta['task_ci_high']:+.6f}] "
                f"bench[{delta['benchmark_ci_low']:+.6f},{delta['benchmark_ci_high']:+.6f}]"
            )
    for row in acquisition_summary:
        delta = row["delta_vs_C_R"]
        print(
            f"acquisition {row['seed']} {row['arm']:8s} {row['brier_alc']:.6f} "
            f"d={delta['mean']:+.6f} task[{delta['task_ci_low']:+.6f},{delta['task_ci_high']:+.6f}] "
            f"bench[{delta['benchmark_ci_low']:+.6f},{delta['benchmark_ci_high']:+.6f}]"
        )
    for row in final_test["selected"]:
        delta = row["delta_cal_vs_beta"]
        print(
            f"final test {row['arm']:12s} raw={row['brier_alc_raw']:.6f} "
            f"cal={row['brier_alc_cal']:.6f} d={delta['mean']:+.6f} "
            f"task[{delta['task_ci_low']:+.6f},{delta['task_ci_high']:+.6f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
