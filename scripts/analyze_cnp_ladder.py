"""Aggregate the CNP encoder ladder into report tables.

The input JSONs are produced by ``modal_cnp_ladder.py`` for each seed/mode.
The selected configuration is seed 20261018, masked.  The script reports that
configuration exactly and the mean and spread over all supplied configurations.
Paired deltas are task-level ALC differences; positive means the first arm has
the higher (worse) Brier ALC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
ARM_ORDER = (
    "beta",
    "irt_1pl",
    "irt_2pl",
    "cnp_no_delta",
    "cnp_full",
    "cnp_no_delta_retrained",
)
ARM_LABELS = {
    "beta": "Beta-Binomial",
    "irt_1pl": "1PL + MAP",
    "irt_2pl": "2PL + MAP",
    "cnp_no_delta": "CNP item head + MAP ($\\Delta\\theta=0$)",
    "cnp_full": "CNP-AIRT ($\\Delta\\theta(C)$)",
    "cnp_no_delta_retrained": "CNP item head + MAP, retrained",
}
PAIRS = (
    ("cnp_full", "cnp_no_delta"),
    ("cnp_no_delta", "irt_2pl"),
    ("irt_2pl", "irt_1pl"),
    ("cnp_full", "cnp_no_delta_retrained"),
)


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _fmt(value: float) -> str:
    return f"{value:.5f}"


def _signed(value: float) -> str:
    return f"{value:+.5f}"


def _delta_brier(row: dict[str, Any]) -> float:
    """Full-minus-zero Brier; positive means the correction hurts."""

    if "brier_full" in row and "brier_zero" in row:
        return float(row["brier_full"]) - float(row["brier_zero"])
    return float(row["delta_brier"])


def _index(rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    return {row["arm"]: row for row in rows if row["split"] == split}


def _paired_index(
    rows: list[dict[str, Any]], split: str
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["first"], row["second"]): row
        for row in rows
        if row["split"] == split
    }


def _ladder_table(
    configs: list[dict[str, Any]], selected: dict[str, Any], split: str
) -> str:
    lines = [
        f"### {split.capitalize()} ALC ladder",
        "",
        "| Arm | Selected raw | Selected cal. | Mean raw | Mean cal. | Cal. SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected_index = _index(selected["alc"], split)
    indexes = [_index(config["alc"], split) for config in configs]
    for arm in ARM_ORDER:
        selected_row = selected_index.get(arm, {})
        raw_values = [
            float(index[arm]["brier_alc_raw"]) for index in indexes if arm in index
        ]
        cal_values = [
            float(index[arm]["brier_alc_cal"]) for index in indexes if arm in index
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    ARM_LABELS[arm],
                    _fmt(float(selected_row.get("brier_alc_raw", float("nan")))),
                    _fmt(float(selected_row.get("brier_alc_cal", float("nan")))),
                    _fmt(_mean(raw_values)),
                    _fmt(_mean(cal_values)),
                    _fmt(_sd(cal_values)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _delta_table(configs: list[dict[str, Any]], selected: dict[str, Any]) -> str:
    lines = [
        "### Test correction magnitude by budget",
        "",
        "| Budget | Contexts | Mean | SD | Mean abs. | 5--95% | Within-task SD | Between-task SD | Mean |logit| shift | Mean |p| shift | Full $-$ zero Brier |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected_delta = {
        int(row["budget"]): row for row in selected["delta"] if row["split"] == "test"
    }
    for budget in BUDGETS:
        row = selected_delta[budget]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    str(int(row["contexts"])),
                    _signed(float(row["delta_mean"])),
                    _fmt(float(row["delta_sd"])) if np.isfinite(row["delta_sd"]) else "--",
                    _fmt(float(row["delta_mean_abs"])),
                    f"{float(row['delta_q05']):+.3f} to {float(row['delta_q95']):+.3f}",
                    _fmt(float(row["delta_within_task_sd"]))
                    if np.isfinite(row["delta_within_task_sd"])
                    else "--",
                    _fmt(float(row["delta_task_mean_sd"]))
                    if np.isfinite(row["delta_task_mean_sd"])
                    else "--",
                    _fmt(float(row["mean_abs_logit_shift"])),
                    _fmt(float(row["mean_abs_probability_shift"])),
                    _signed(_delta_brier(row)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Positive Full $-$ zero Brier means the correction increases Brier error.")
    lines.append("")
    lines.append("Mean over configurations:")
    lines.append("")
    lines.append(
        "| Budget | Mean | SD | Mean abs. | Mean |logit| shift | Mean |p| shift | Full $-$ zero Brier |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for budget in BUDGETS:
        rows = [
            row
            for config in configs
            for row in config["delta"]
            if row["split"] == "test" and int(row["budget"]) == budget
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    _signed(_mean([float(row["delta_mean"]) for row in rows])),
                    _fmt(_mean([float(row["delta_sd"]) for row in rows])),
                    _fmt(_mean([float(row["delta_mean_abs"]) for row in rows])),
                    _fmt(_mean([float(row["mean_abs_logit_shift"]) for row in rows])),
                    _fmt(_mean([float(row["mean_abs_probability_shift"]) for row in rows])),
                    _signed(_mean([_delta_brier(row) for row in rows])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _paired_table(configs: list[dict[str, Any]], selected: dict[str, Any]) -> str:
    lines = [
        "### Paired test ALC differences",
        "",
        "Positive means the first arm has higher (worse) ALC.",
        "",
        "| Comparison | Selected raw | Selected cal. | Mean raw | Configs first wins (raw) | Mean cal. | Configs first wins (cal) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    selected_index = _paired_index(selected["paired"], "test")
    for first, second in PAIRS:
        selected_row = selected_index[(first, second)]
        raw_values = []
        cal_values = []
        raw_wins = 0
        cal_wins = 0
        for config in configs:
            index = _paired_index(config["paired"], "test")
            if (first, second) not in index:
                continue
            row = index[(first, second)]
            raw_values.append(float(row["weighted_raw"]["mean"]))
            cal_values.append(float(row["weighted_cal"]["mean"]))
            raw_wins += int(float(row["weighted_raw"]["mean"]) < 0.0)
            cal_wins += int(float(row["weighted_cal"]["mean"]) < 0.0)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{ARM_LABELS[first]} $-$ {ARM_LABELS[second]}",
                    _signed(float(selected_row["weighted_raw"]["mean"])),
                    _signed(float(selected_row["weighted_cal"]["mean"])),
                    _signed(_mean(raw_values)),
                    f"{raw_wins}/{len(raw_values)}",
                    _signed(_mean(cal_values)),
                    f"{cal_wins}/{len(cal_values)}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Selected-configuration 95% paired intervals:")
    lines.append("")
    lines.append("| Comparison | Metric | Mean | SE | 95% CI | Tasks |")
    lines.append("|---|---|---:|---:|---|---:|")
    for first, second in PAIRS:
        row = selected_index[(first, second)]
        for metric, label in (("weighted_raw", "raw"), ("weighted_cal", "cal.")):
            value = row[metric]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{ARM_LABELS[first]} $-$ {ARM_LABELS[second]}",
                        label,
                        _signed(float(value["mean"])),
                        _fmt(float(value["se"])),
                        f"[{float(value['ci_low']):+.5f}, {float(value['ci_high']):+.5f}]",
                        str(int(value["tasks"])),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/opencode/cnp-ladder/summary.md")
    )
    args = parser.parse_args()

    configs = [json.loads(path.read_text(encoding="utf-8")) for path in args.paths]
    configs.sort(key=lambda config: (int(config["seed"]), str(config["mode"])))
    selected_candidates = [
        config
        for config in configs
        if int(config["seed"]) == 20261018 and config["mode"] == "masked"
    ]
    selected = selected_candidates[0] if selected_candidates else configs[0]
    labels = [f"{config['seed']}-{config['mode']}" for config in configs]
    lines = [
        "# CNP encoder ladder",
        "",
        f"Configurations: {', '.join(labels)}",
        "",
        f"Selected: {selected['seed']}-{selected['mode']}",
        "",
    ]
    for split in ("validation", "test"):
        lines.append(_ladder_table(configs, selected, split))
    lines.append(_delta_table(configs, selected))
    lines.append(_paired_table(configs, selected))
    output = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
