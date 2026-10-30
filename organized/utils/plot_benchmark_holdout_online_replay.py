#!/usr/bin/env python3
"""Plot paired streaming-acquisition replays from their frozen summaries.

Example:
  .venv/bin/python organized/utils/plot_benchmark_holdout_online_replay.py \
    --runs-root /tmp/aims-online-replay-XXXXXX/benchmark-holdout-online-v1-s20261018-n15-m80-cap1000 \
    --output docs/online-replay-acquisition-validation.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "organized" / "utils"))
from plotstyle import MUTED, TEXTWIDTH, header, style  # noqa: E402


BUDGETS = (0, 1, 3, 7, 15, 31)
MODEL_ORDER = ("ladder", "filtered", "w128")
MODEL_LABELS = {
    "ladder": "W64 benchmark-holdout\nbudget-ladder",
    "filtered": "W64 benchmark-holdout\nfiltered",
    "w128": "W128 BGE-M3\nprior checkpoint",
}
POLICY_ORDER = ("random_prefix", "uncertainty", "confidence_extremes")
POLICY_LABELS = {
    "random_prefix": "Random prefix",
    "uncertainty": "Uncertainty |p - 0.5| ≤ 0.15",
    "confidence_extremes": "Confidence extremes |p - 0.5| ≥ 0.15",
}
POLICY_COLORS = {
    "random_prefix": MUTED["indigo"],
    "uncertainty": MUTED["teal"],
    "confidence_extremes": MUTED["rose"],
}


def _model_key(run_name: str) -> str:
    for key in MODEL_ORDER:
        if f"-{key}-" in run_name:
            return key
    raise ValueError(f"cannot identify checkpoint family from run name: {run_name}")


def _ordered_models(data: pd.DataFrame) -> list[str]:
    """Keep historical checkpoints first, then append registered baselines."""
    present = set(data["model"])
    return [*filter(present.__contains__, MODEL_ORDER), *sorted(present.difference(MODEL_ORDER))]


def load_summaries(runs_root: Path) -> pd.DataFrame:
    records = []
    for summary_path in sorted(runs_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("role") != "validation":
            continue
        run_name = summary_path.parent.name
        # New runners register their identity directly. Historical ANP runs
        # predate this field, so retain the explicit name parsing as a fallback.
        model = str(summary.get("model_id") or _model_key(run_name))
        model_label = str(summary.get("model_label") or MODEL_LABELS.get(model, model))
        policy = summary["policy"]
        if policy not in POLICY_ORDER:
            continue
        row = {
            "run_name": run_name,
            "model": model,
            "model_label": model_label,
            "policy": policy,
            "policy_label": POLICY_LABELS[policy],
            "brier_alc": float(summary["macro_task_brier_alc"]),
            "n_tasks": int(summary["n_tasks"]),
            "n_order_seeds": int(summary["n_order_seeds"]),
        }
        for budget in BUDGETS:
            row[f"brier_{budget}"] = float(summary["macro_task_brier_by_budget"][str(budget)])
        records.append(row)
    data = pd.DataFrame(records)
    if data.empty:
        raise RuntimeError(f"no validation summaries found under {runs_root}")
    duplicated = data.duplicated(["model", "policy"], keep=False)
    if duplicated.any():
        duplicated_names = data.loc[duplicated, "run_name"].tolist()
        raise RuntimeError(f"more than one run per checkpoint/policy: {duplicated_names}")
    return data


def _add_curves_page(pdf: PdfPages, data: pd.DataFrame) -> None:
    models = _ordered_models(data)
    n_columns = 2
    n_rows = int(np.ceil((len(models) + 1) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(TEXTWIDTH, 2.05 * n_rows + 0.15), constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()
    header(
        fig,
        "Streaming acquisition on the frozen benchmark-holdout validation panel",
        "Five paired subject×benchmark tasks, 10 fixed candidate orders, disjoint 50/50 acquisition/evaluation pools. Lower is better.",
    )
    y_values = data[[f"brier_{budget}" for budget in BUDGETS]].to_numpy().ravel()
    lower = max(0.0, float(np.floor((y_values.min() - 0.01) * 20) / 20))
    upper = float(np.ceil((y_values.max() + 0.01) * 20) / 20)
    x = np.arange(len(BUDGETS))
    for axis, model in zip(axes[:len(models)], models, strict=True):
        selected = data[data["model"] == model]
        for policy in POLICY_ORDER:
            row = selected[selected["policy"] == policy]
            if row.empty:
                continue
            values = row[[f"brier_{budget}" for budget in BUDGETS]].iloc[0].to_numpy(dtype=float)
            axis.plot(x, values, marker="o", color=POLICY_COLORS[policy],
                      label=POLICY_LABELS[policy])
        axis.set_title(str(selected["model_label"].iloc[0]).replace("\n", " "))
        axis.set_xticks(x, BUDGETS)
        axis.set_xlabel("Acquired labels, B")
        axis.set_ylabel("Macro-task Brier")
        axis.set_ylim(lower, upper)
        style(axis)

    bar_axis = axes[len(models)]
    width = 0.23
    centers = np.arange(len(models))
    for offset, policy in enumerate(POLICY_ORDER):
        values = []
        for model in models:
            subset = data[(data["model"] == model) & (data["policy"] == policy)]
            values.append(float(subset["brier_alc"].iloc[0]) if not subset.empty else np.nan)
        bar_axis.bar(centers + (offset - 1) * width, values, width=width,
                     color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
    bar_axis.set_xticks(
        centers,
        [str(data.loc[data["model"] == model, "model_label"].iloc[0]).replace("\n", " ") for model in models],
        fontsize=6.3,
    )
    bar_axis.set_ylabel("Final Brier ALC")
    bar_axis.set_ylim(bottom=max(0.0, float(data["brier_alc"].min() - 0.02)))
    style(bar_axis)
    for axis in axes[len(models) + 1:]:
        axis.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.025), fontsize=6.3)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.16, top=0.86, hspace=0.60, wspace=0.37)
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_page(pdf: PdfPages, data: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(TEXTWIDTH, 3.7))
    axis.axis("off")
    header(
        fig,
        "Exact validation Brier scores by budget",
        "Each row is averaged within task over the 10 frozen candidate orders, then averaged equally across five tasks.",
    )
    rows = []
    for model in _ordered_models(data):
        for policy in POLICY_ORDER:
            selected = data[(data["model"] == model) & (data["policy"] == policy)]
            if selected.empty:
                continue
            row = selected.iloc[0]
            rows.append([
                str(row["model_label"]).replace("\n", " "),
                POLICY_LABELS[policy],
                *[f"{row[f'brier_{budget}']:.4f}" for budget in BUDGETS],
                f"{row['brier_alc']:.4f}",
            ])
    columns = ["Checkpoint", "Acquisition", *[f"B={budget}" for budget in BUDGETS], "ALC"]
    table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="right",
                       colLoc="right", bbox=[0.0, 0.04, 1.0, 0.80])
    table.auto_set_font_size(False)
    table.set_fontsize(5.4)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("#BBBBBB")
        if row_index == 0:
            cell.set_facecolor("#EDEDED")
            cell.set_text_props(weight="bold")
        if column_index in (0, 1):
            cell.set_text_props(ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "online-replay-acquisition-validation.pdf")
    args = parser.parse_args()
    data = load_summaries(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    data.sort_values(["model", "policy"]).to_csv(csv_path, index=False)
    with PdfPages(args.output) as pdf:
        _add_curves_page(pdf, data)
        _add_table_page(pdf, data)
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
