"""Aggregate the calibration decomposition runs into report tables.

Each input JSON is one (seed, mode) calibration-ablation run with both models.
The script reports the selected configuration exactly and the mean, standard
deviation and range across all supplied configurations. Paired deltas use the
same configurations, so the comparison is not confused by between-run spread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
DECOMPOSITION = ("raw", "platt_bce", "anchor_bce", "full_bce", "context")
OBJECTIVE = ("full_bce", "full_brier", "platt_bce", "platt_brier", "anchor_bce", "anchor_brier")
WEIGHTING = ("full_bce", "full_bce_task", "full_brier", "full_brier_task")
BUDGET_FIT = ("full_bce", "full_bce_shared", "full_brier", "full_brier_shared")
ZERO_CONTROL = (
    "full_bce",
    "full_bce_zero_half",
    "full_brier",
    "full_brier_zero_half",
    "anchor_bce",
    "context",
)


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _fmt(value: float) -> str:
    return f"{value:.5f}"


def _row_index(rows: list[dict[str, Any]], split: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["model"], row["variant"]): row for row in rows if row["split"] == split}


def _by_budget_index(
    rows: list[dict[str, Any]], split: str
) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["model"], row["variant"], int(row["budget"])): row
        for row in rows
        if row["split"] == split
    }


def _table(
    title: str,
    variants: tuple[str, ...],
    configs: list[dict[str, Any]],
    model: str,
    split: str,
    selected: dict[str, Any] | None,
) -> str:
    lines = [f"### {title}", ""]
    lines.append("| Variant | Selected | Mean | SD | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    indexes = [_row_index(config["alc"], split) for config in configs]
    for variant in variants:
        values = [
            float(index[(model, variant)]["brier_alc"])
            for index in indexes
            if (model, variant) in index
        ]
        if selected is None:
            selected_value = float("nan")
        else:
            selected_row = _row_index(selected["alc"], split).get((model, variant))
            selected_value = float(selected_row["brier_alc"]) if selected_row else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    _fmt(selected_value),
                    _fmt(_mean(values)),
                    _fmt(_sd(values)),
                    _fmt(min(values)),
                    _fmt(max(values)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _delta_table(
    title: str,
    pairs: tuple[tuple[str, str], ...],
    configs: list[dict[str, Any]],
    model: str,
    split: str,
) -> str:
    lines = [f"### {title}", ""]
    lines.append("| Comparison | Mean delta | SD | Configs where first wins |")
    lines.append("|---|---:|---:|---:|")
    indexes = [_row_index(config["alc"], split) for config in configs]
    for first, second in pairs:
        deltas = []
        wins = 0
        for index in indexes:
            if (model, first) not in index or (model, second) not in index:
                continue
            delta = float(index[(model, first)]["brier_alc"]) - float(
                index[(model, second)]["brier_alc"]
            )
            deltas.append(delta)
            wins += int(delta < 0.0)
        if not deltas:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{first} - {second}",
                    f"{_mean(deltas):+.5f}",
                    _fmt(_sd(deltas)),
                    f"{wins}/{len(deltas)}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _budget_table(
    title: str,
    variants: tuple[str, ...],
    configs: list[dict[str, Any]],
    model: str,
    split: str,
    selected: dict[str, Any] | None,
) -> str:
    lines = [f"### {title}", ""]
    lines.append(
        "| Variant | "
        + " | ".join(f"B{budget}" for budget in BUDGETS)
        + " | ALC |"
    )
    lines.append("|---|" + "---:|" * (len(BUDGETS) + 1))
    indexes = [_by_budget_index(config["by_budget"], split) for config in configs]
    alc_indexes = [_row_index(config["alc"], split) for config in configs]
    selected_index = _by_budget_index(selected["by_budget"], split) if selected else None
    for variant in variants:
        selected_values = []
        for budget in BUDGETS:
            row = selected_index.get((model, variant, budget)) if selected_index else None
            selected_values.append(float(row["brier"]) if row else float("nan"))
        alc_row = _row_index(selected["alc"], split).get((model, variant)) if selected else None
        alc_value = float(alc_row["brier_alc"]) if alc_row else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [variant]
                + [_fmt(value) for value in selected_values]
                + [_fmt(alc_value)]
            )
            + " |"
        )
    lines.append("")
    lines.append("Mean across configurations:")
    lines.append("")
    lines.append(
        "| Variant | "
        + " | ".join(f"B{budget}" for budget in BUDGETS)
        + " | ALC |"
    )
    lines.append("|---|" + "---:|" * (len(BUDGETS) + 1))
    for variant in variants:
        values = []
        for budget in BUDGETS:
            budget_values = [
                float(index[(model, variant, budget)]["brier"])
                for index in indexes
                if (model, variant, budget) in index
            ]
            values.append(_mean(budget_values))
        alc_values = [
            float(index[(model, variant)]["brier_alc"])
            for index in alc_indexes
            if (model, variant) in index
        ]
        lines.append(
            "| "
            + " | ".join(
                [variant] + [_fmt(value) for value in values] + [_fmt(_mean(alc_values))]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/opencode/calibration-ablation/summary.md"))
    args = parser.parse_args()

    configs = [json.loads(path.read_text(encoding="utf-8")) for path in args.paths]
    configs.sort(key=lambda config: (int(config["seed"]), str(config["mode"])))
    selected_candidates = [
        config
        for config in configs
        if int(config["seed"]) == 20261018 and config["mode"] == "masked"
    ]
    selected = selected_candidates[0] if selected_candidates else None

    labels = [f"{config['seed']}-{config['mode']}" for config in configs]
    if selected is not None:
        selected_label = f"{selected['seed']}-{selected['mode']}"
    else:
        selected_label = "unavailable"
    lines = [
        "# Calibration ablation",
        "",
        f"Configurations: {', '.join(labels)}",
        "",
        f"Selected: {selected_label}",
        "",
    ]

    for model in ("cnp_airt", "online_airt_2pl"):
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            _table(
                "Test ALC by calibrator form",
                DECOMPOSITION,
                configs,
                model,
                "test",
                selected,
            )
        )
        lines.append(
            _table(
                "Test ALC by fit objective",
                OBJECTIVE,
                configs,
                model,
                "test",
                selected,
            )
        )
        lines.append(
            _table(
                "Test ALC by fitting weight",
                WEIGHTING,
                configs,
                model,
                "test",
                selected,
            )
        )
        lines.append(
            _table(
                "Test ALC by budget scope",
                BUDGET_FIT,
                configs,
                model,
                "test",
                selected,
            )
        )
        lines.append(
            _table(
                "Test ALC with the budget-zero control",
                ZERO_CONTROL,
                configs,
                model,
                "test",
                selected,
            )
        )
        lines.append(
            _delta_table(
                "Paired test ALC deltas",
                (
                    ("anchor_bce", "full_bce"),
                    ("context", "full_bce"),
                    ("full_bce", "raw"),
                    ("full_brier", "full_bce"),
                    ("full_bce_task", "full_bce"),
                    ("full_bce_shared", "full_bce"),
                    ("full_bce_zero_half", "full_bce"),
                    ("full_brier_zero_half", "full_brier"),
                ),
                configs,
                model,
                "test",
            )
        )
        lines.append(
            _budget_table(
                "Selected-configuration test Brier by budget",
                (
                    "raw",
                    "platt_bce",
                    "anchor_bce",
                    "full_bce",
                    "context",
                    "full_bce_zero_half",
                    "full_brier",
                ),
                configs,
                model,
                "test",
                selected,
            )
        )

    output = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
