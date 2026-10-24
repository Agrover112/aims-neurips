#!/usr/bin/env python3
"""Summarize the validation-only embedding ablation.

The result JSONs are produced by ``modal_embedding_ablation.py``.  Each file
contains one row per task and predictor in ``task_alc``.  This script reports
the ALC table and paired task deltas between embedding arms.  Negative deltas
mean the first arm has the lower (better) Brier ALC.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PREDICTORS = ("beta", "airt", "cnp")


def load_results(root: Path) -> list[dict]:
    results = []
    for path in sorted(root.glob("embedding-*.json")):
        data = json.loads(path.read_text())
        if data.get("arm") and data.get("task_alc"):
            results.append(data)
    return results


def _task_values(result: dict, predictor: str, metric: str) -> dict[str, float]:
    return {
        str(row["task_id"]): float(row[metric])
        for row in result["task_alc"]
        if row["arm"] == predictor
    }


def paired_delta(
    first: dict,
    second: dict,
    predictor: str,
    *,
    metric: str = "weighted_cal",
) -> dict[str, float]:
    a = _task_values(first, predictor, metric)
    b = _task_values(second, predictor, metric)
    common = sorted(a.keys() & b.keys())
    if not common:
        raise ValueError(f"no shared tasks for {predictor}")
    delta = np.asarray([a[task] - b[task] for task in common], dtype=float)
    n = len(delta)
    se = float(delta.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "tasks": n,
        "mean": float(delta.mean()),
        "se": se,
        "ci_low": float(delta.mean() - 1.96 * se),
        "ci_high": float(delta.mean() + 1.96 * se),
        "first_better_tasks": int((delta < 0).sum()),
    }


def alc_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for row in result["alc"]:
            rows.append(
                {
                    "arm": result["arm"],
                    "seed": result["seed"],
                    "mode": result["mode"],
                    "predictor": row["arm"],
                    "brier_alc_cal": row["brier_alc_cal"],
                    "brier_alc_cal_se": row["brier_alc_cal_se"],
                    "brier_alc_raw": row["brier_alc_raw"],
                    "brier_alc_raw_se": row["brier_alc_raw_se"],
                    "tasks": row["tasks"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/tmp/opencode/embedding-ablation"),
        help="directory containing embedding-*.json",
    )
    parser.add_argument(
        "--baseline",
        default="bge_base",
        help="arm used for paired arm-level deltas",
    )
    args = parser.parse_args()

    results = load_results(args.root)
    if not results:
        raise SystemExit(f"no result JSONs found in {args.root}")
    by_arm = {result["arm"]: result for result in results}

    print("\nALC (validation, calibrated unless noted)")
    print(
        f"{'arm':<24} {'pred':<6} {'cal':>9} {'se':>9} "
        f"{'raw':>9} {'raw_se':>9} {'tasks':>6}"
    )
    for row in alc_rows(results):
        print(
            f"{row['arm']:<24} {row['predictor']:<6} "
            f"{row['brier_alc_cal']:9.6f} {row['brier_alc_cal_se']:9.6f} "
            f"{row['brier_alc_raw']:9.6f} {row['brier_alc_raw_se']:9.6f} "
            f"{row['tasks']:6d}"
        )

    if args.baseline not in by_arm:
        print(f"\nbaseline {args.baseline!r} not present; skipping paired deltas")
        return
    baseline = by_arm[args.baseline]

    print(f"\nPaired task deltas vs {args.baseline} (negative = arm is better)")
    print(
        f"{'arm':<24} {'pred':<6} {'delta':>10} {'se':>9} "
        f"{'95% CI':>22} {'arm better':>10}"
    )
    for result in results:
        if result["arm"] == args.baseline:
            continue
        for predictor in PREDICTORS:
            try:
                delta = paired_delta(result, baseline, predictor)
            except ValueError:
                continue
            ci = f"[{delta['ci_low']:+.6f},{delta['ci_high']:+.6f}]"
            print(
                f"{result['arm']:<24} {predictor:<6} "
                f"{delta['mean']:+10.6f} {delta['se']:9.6f} "
                f"{ci:>22} {delta['first_better_tasks']:>10d}/{delta['tasks']}"
            )

    print("\nWithin-arm CNP - AIRT (paired tasks, calibrated)")
    print(f"{'arm':<24} {'delta':>10} {'se':>9} {'95% CI':>22}")
    for result in results:
        airt = _task_values(result, "airt", "weighted_cal")
        cnp = _task_values(result, "cnp", "weighted_cal")
        common = sorted(airt.keys() & cnp.keys())
        values = np.asarray([cnp[task] - airt[task] for task in common])
        se = float(values.std(ddof=1) / math.sqrt(len(values)))
        ci = f"[{values.mean() - 1.96 * se:+.6f},{values.mean() + 1.96 * se:+.6f}]"
        print(
            f"{result['arm']:<24} {values.mean():+10.6f} {se:9.6f} {ci:>22}"
            f"  ({int((values < 0).sum())}/{len(values)} tasks)"
        )


if __name__ == "__main__":
    main()
