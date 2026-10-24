"""Summarize the reverse-order r_C ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNGS = (
    "anp_attn",
    "cnp",
    "direct_ab",
    "additive_ab",
    "residual",
    "scalar",
    "2pl",
)
T975_103 = 1.983


def _load(paths: list[Path]) -> dict[tuple[str, int], dict]:
    output = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        output[(data["rung"], int(data["seed"]))] = data
    return output


def _task_map(data: dict, arm: str) -> dict[str, float]:
    return {
        row["task_id"]: float(row["weighted"])
        for row in data["task_alc"]
        if row["arm"] == arm
    }


def _paired(deltas_by_seed: list[dict[str, float]]) -> dict[str, float]:
    tasks = sorted(set.intersection(*(set(values) for values in deltas_by_seed)))
    values = np.asarray(
        [np.mean([values[task] for values in deltas_by_seed]) for task in tasks],
        dtype=float,
    )
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {
        "mean": mean,
        "ci_low": mean - T975_103 * se,
        "ci_high": mean + T975_103 * se,
        "wins": int((values < 0.0).sum()),
        "tasks": len(values),
    }


def main(paths: list[Path], output: Path | None = None) -> str:
    data = _load(paths)
    seeds = sorted({seed for _, seed in data})
    lines = ["# Reverse r_C ablation", "", "Lower Brier ALC is better.", ""]
    lines.append("| rung | " + " | ".join(str(seed) for seed in seeds) + " | mean |")
    lines.append("|---|" + "---:|" * len(seeds) + "---:|")
    for rung in RUNGS:
        values = [data[(rung, seed)]["alc"] for seed in seeds if (rung, seed) in data]
        by_seed = [
            next(row["brier_alc"] for row in result if row["arm"] == rung)
            for result in values
        ]
        cells = [f"{value:.6f}" for value in by_seed]
        cells += ["—"] * (len(seeds) - len(by_seed))
        lines.append(
            f"| {rung} | " + " | ".join(cells) + f" | {np.mean(by_seed):.6f} |"
        )

    lines += ["", "## Paired deltas vs 2PL + MAP", "", "Negative = rung better.", ""]
    lines.append("| rung | delta | 95% CI | wins |")
    lines.append("|---|---:|---:|---:|")
    for rung in RUNGS[:-1]:
        deltas_by_seed = []
        for seed in seeds:
            if (rung, seed) not in data or ("2pl", seed) not in data:
                continue
            first = _task_map(data[(rung, seed)], rung)
            second = _task_map(data[("2pl", seed)], "2pl")
            tasks = sorted(set(first) & set(second))
            deltas_by_seed.append({task: first[task] - second[task] for task in tasks})
        if not deltas_by_seed:
            continue
        result = _paired(deltas_by_seed)
        lines.append(
            f"| {rung} | {result['mean']:+.6f} | "
            f"[{result['ci_low']:+.6f}, {result['ci_high']:+.6f}] | "
            f"{result['wins']}/{result['tasks']} |"
        )

    lines += ["", "## Beta by seed", ""]
    lines.append("| seed | Beta ALC |")
    lines.append("|---:|---:|")
    for seed in seeds:
        if (RUNGS[0], seed) not in data:
            continue
        beta = next(
            row["brier_alc"]
            for row in data[(RUNGS[0], seed)]["alc"]
            if row["arm"] == "beta"
        )
        lines.append(f"| {seed} | {beta:.6f} |")

    text = "\n".join(lines) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(main(args.paths, args.output))
