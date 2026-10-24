"""Summarize the corrected-protocol faithful ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

T975_103 = 1.983
T80_103 = 0.845


def _verdict(mean: float, ci_low: float, ci_high: float, mde: float) -> str:
    """Conservative paired verdict: small effects are underpowered, not null."""

    if abs(mean) < mde:
        return "can't tell"
    if ci_high < 0.0:
        return "better"
    if ci_low > 0.0:
        return "worse"
    return "can't tell"


def _load(paths: list[Path]) -> dict[tuple[str, str, int], dict]:
    output = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        key = (str(data["rung"]), str(data["objective"]), int(data["seed"]))
        output[key] = data
    return output


def _alc(data: dict, arm: str, calibration: str = "oof") -> float | None:
    for row in data.get("alc", []):
        if row.get("arm") == arm and row.get("calibration", "oof") == calibration:
            return float(row["brier_alc"])
    return None


def _task_map(data: dict, arm: str, calibration: str = "oof") -> dict[str, float]:
    return {
        str(row["task_id"]): float(row["weighted"])
        for row in data.get("task_alc", [])
        if row.get("arm") == arm and row.get("calibration", "oof") == calibration
    }


def _paired(deltas_by_seed: list[dict[str, float]]) -> dict[str, float]:
    tasks = sorted(set.intersection(*(set(values) for values in deltas_by_seed)))
    values = np.asarray(
        [np.mean([values[task] for values in deltas_by_seed]) for task in tasks],
        dtype=float,
    )
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    ci_low = mean - T975_103 * se
    ci_high = mean + T975_103 * se
    mde = (T975_103 + T80_103) * se
    return {
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mde": mde,
        "verdict": _verdict(mean, ci_low, ci_high, mde),
        "wins": int((values < 0.0).sum()),
        "tasks": len(values),
    }


def main(paths: list[Path], output: Path | None = None) -> str:
    data = _load(paths)
    if not data:
        raise ValueError("no result files")
    seeds = sorted({seed for _, _, seed in data})
    rungs = sorted({rung for rung, _, _ in data})
    objectives = sorted({objective for _, objective, _ in data})
    lines = ["# Faithful ladder", "", "Lower calibrated Brier ALC is better.", ""]

    lines.append("| rung | objective | " + " | ".join(str(seed) for seed in seeds) + " | mean |")
    lines.append("|---|---|" + "---:|" * len(seeds) + "---:|")
    for rung in rungs:
        for objective in objectives:
            values = []
            cells = []
            for seed in seeds:
                value = _alc(data.get((rung, objective, seed), {}), rung)
                if value is None:
                    cells.append("—")
                else:
                    cells.append(f"{value:.6f}")
                    values.append(value)
            mean = f"{np.mean(values):.6f}" if values else "—"
            lines.append(f"| {rung} | {objective} | " + " | ".join(cells) + f" | {mean} |")

    lines += [
        "",
        "## Paired deltas vs 2PL",
        "",
        "Negative = rung better. MDE80 is the paired 80%-power minimum detectable effect.",
        "A delta smaller than MDE80 is reported as `can't tell`, not as no effect.",
        "",
    ]
    lines.append("| rung | objective | delta | 95% CI | MDE80 | verdict | wins |")
    lines.append("|---|---|---:|---:|---:|---|---:|")
    for rung in rungs:
        if rung == "2pl":
            continue
        for objective in objectives:
            deltas_by_seed = []
            for seed in seeds:
                first_data = data.get((rung, objective, seed))
                second_data = data.get(("2pl", objective, seed))
                if first_data is None or second_data is None:
                    continue
                first = _task_map(first_data, rung)
                second = _task_map(second_data, "2pl")
                tasks = sorted(set(first) & set(second))
                deltas_by_seed.append({task: first[task] - second[task] for task in tasks})
            if not deltas_by_seed:
                continue
            result = _paired(deltas_by_seed)
            lines.append(
                f"| {rung} | {objective} | {result['mean']:+.6f} | "
                f"[{result['ci_low']:+.6f}, {result['ci_high']:+.6f}] | "
                f"{result['mde']:.6f} | {result['verdict']} | "
                f"{result['wins']}/{result['tasks']} |"
            )

    lines += ["", "## Component deltas", "", "Negative = first arm better.", ""]
    lines.append("| contrast | objective | delta | 95% CI | MDE80 | verdict | wins |")
    lines.append("|---|---|---:|---:|---:|---|---:|")
    contrasts = (
        ("2pl_anp", "2pl"),
        ("gp_2pl", "2pl"),
        ("gp_anp", "gp_2pl"),
        ("gp_anp", "2pl_anp"),
        ("gp_anp_prior", "gp_anp"),
        ("latent_anp", "latent_np"),
    )
    for first_rung, second_rung in contrasts:
        for objective in objectives:
            deltas_by_seed = []
            for seed in seeds:
                first_data = data.get((first_rung, objective, seed))
                second_data = data.get((second_rung, objective, seed))
                if first_data is None or second_data is None:
                    continue
                first = _task_map(first_data, first_rung)
                second = _task_map(second_data, second_rung)
                tasks = sorted(set(first) & set(second))
                deltas_by_seed.append({task: first[task] - second[task] for task in tasks})
            if not deltas_by_seed:
                continue
            result = _paired(deltas_by_seed)
            lines.append(
                f"| {first_rung} - {second_rung} | {objective} | {result['mean']:+.6f} | "
                f"[{result['ci_low']:+.6f}, {result['ci_high']:+.6f}] | "
                f"{result['mde']:.6f} | {result['verdict']} | "
                f"{result['wins']}/{result['tasks']} |"
            )

    lines += ["", "## Objective deltas vs BCE", "", "Negative = objective better.", ""]
    lines.append("| rung | objective | delta | 95% CI | MDE80 | verdict | wins |")
    lines.append("|---|---|---:|---:|---:|---|---:|")
    for rung in rungs:
        for objective in objectives:
            if objective == "bce":
                continue
            deltas_by_seed = []
            for seed in seeds:
                first_data = data.get((rung, objective, seed))
                second_data = data.get((rung, "bce", seed))
                if first_data is None or second_data is None:
                    continue
                first = _task_map(first_data, rung)
                second = _task_map(second_data, rung)
                tasks = sorted(set(first) & set(second))
                deltas_by_seed.append({task: first[task] - second[task] for task in tasks})
            if not deltas_by_seed:
                continue
            result = _paired(deltas_by_seed)
            lines.append(
                f"| {rung} | {objective} | {result['mean']:+.6f} | "
                f"[{result['ci_low']:+.6f}, {result['ci_high']:+.6f}] | "
                f"{result['mde']:.6f} | {result['verdict']} | "
                f"{result['wins']}/{result['tasks']} |"
            )

    lines += ["", "## Beta reference", ""]
    lines.append("| objective | " + " | ".join(str(seed) for seed in seeds) + " |")
    lines.append("|---|" + "---:|" * len(seeds))
    for objective in objectives:
        cells = []
        for seed in seeds:
            result = data.get(("2pl", objective, seed))
            value = _alc(result, "beta") if result else None
            cells.append("—" if value is None else f"{value:.6f}")
        lines.append(f"| {objective} | " + " | ".join(cells) + " |")

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
