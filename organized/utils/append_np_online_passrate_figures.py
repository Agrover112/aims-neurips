#!/usr/bin/env python3
"""Append current NP item-pass-rate and Figure-11 analogs to the model PDF.

The inputs are paired random-streaming replay outputs. The figures deliberately
describe only this small, frozen validation panel; they do not claim a global
item-difficulty measurement or use any future candidate during acquisition.

Example:
  .venv/bin/python organized/utils/append_np_online_passrate_figures.py \
    --runs-root /tmp/aims-online-replay-XXXXXX/benchmark-holdout-online-v1-s20261018-n15-m80-cap1000
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "organized" / "utils"))
from plotstyle import MUTED, TEXTWIDTH, header, style  # noqa: E402


MODELS = (
    ("w128", "val-w128-random-v1", "W128 BGE, non-holdout", MUTED["indigo"]),
    ("ladder", "val-ladder-random-v1", "W64 ANP, holdout ladder", MUTED["rose"]),
    ("filtered", "val-filtered-random-v1", "W64 ANP, holdout filtered", MUTED["teal"]),
)
BUDGETS = (3, 7, 31)
ROLES = ("train", "validation", "test")
ROLE_COLOURS = {"train": MUTED["indigo"], "validation": MUTED["rose"], "test": MUTED["teal"]}


def _finite_correlation(left: pd.Series, right: pd.Series) -> float:
    x, y = left.to_numpy(float), right.to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def _item_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    """Model p versus empirical panel pass rate, both at budget zero."""
    data = predictions[(predictions["requested_budget"] == 0)
                       & (predictions["order_seed"] == 0)].copy()
    return (data.groupby(["benchmark_id", "item_uid"], as_index=False)
            .agg(observed_pass_rate=("label", "mean"),
                 predicted_probability=("probability", "mean"),
                 n_task_responses=("label", "size")))


def _within_benchmark_correlation(data: pd.DataFrame) -> float:
    centred = data.assign(
        observed=data["observed_pass_rate"] - data.groupby("benchmark_id")["observed_pass_rate"].transform("mean"),
        predicted=data["predicted_probability"] - data.groupby("benchmark_id")["predicted_probability"].transform("mean"),
    )
    return _finite_correlation(centred["predicted"], centred["observed"])


def _task_estimates(predictions: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """CTT and model estimates versus the same evaluation-pool truth."""
    scored = (predictions[predictions["requested_budget"].isin(BUDGETS)]
              .groupby(["task_id", "benchmark_id", "order_seed", "requested_budget"], as_index=False)
              .agg(model=("probability", "mean"), truth=("label", "mean"))
              .rename(columns={"requested_budget": "budget"}))
    accepted = decisions[decisions["accepted"]].copy()
    if accepted["revealed_label"].isna().any():
        raise ValueError("random-streaming replay has accepted rows without labels")
    accepted = accepted.sort_values(["task_id", "order_seed", "stream_position"])
    ctt_rows = []
    for (task_id, order_seed), rows in accepted.groupby(["task_id", "order_seed"], sort=False):
        labels = rows["revealed_label"].to_numpy(float)
        for budget in BUDGETS:
            if len(labels) < budget:
                raise ValueError(f"{task_id} order {order_seed} has fewer than {budget} accepted labels")
            ctt_rows.append({"task_id": task_id, "order_seed": order_seed,
                             "budget": budget, "ctt": float(labels[:budget].mean())})
    ctt = pd.DataFrame(ctt_rows)
    result = scored.merge(ctt, on=["task_id", "order_seed", "budget"], how="inner", validate="one_to_one")
    if len(result) != len(scored):
        raise ValueError("Figure-11 CTT merge lost scored task/order/budget rows")
    return result


def _add_passrate_page(pdf: PdfPages, per_model: dict[str, dict]) -> list[dict]:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.25), constrained_layout=False)
    header(
        fig,
        "Is the Neural Process learning the panel item pass-rate distribution?",
        "Budget 0 only. Grey is empirical pass rate over the frozen validation evaluation cells; colour is the model probability, averaged over tasks containing the item.",
    )
    stats_rows = []
    for axis, (key, _, label, colour) in zip(axes, MODELS, strict=True):
        data = per_model[key]["items"]
        corr = _finite_correlation(data["predicted_probability"], data["observed_pass_rate"])
        within = _within_benchmark_correlation(data)
        axis.hist(data["observed_pass_rate"], bins=20, range=(0, 1), density=True,
                  color="#D5DBE1", label="observed")
        axis.hist(data["predicted_probability"], bins=20, range=(0, 1), density=True,
                  histtype="step", linewidth=1.6, color=colour, label="predicted")
        axis.set_title(label, loc="left", fontsize=7.7, fontweight="bold")
        axis.text(0.02, 0.96,
                  f"items {len(data)}\ncorr {corr:.2f}\nwithin-bench {within:.2f}\n"
                  f"sd: pred {data.predicted_probability.std():.2f}, obs {data.observed_pass_rate.std():.2f}",
                  transform=axis.transAxes, va="top", fontsize=6.3)
        axis.set(xlabel="pass rate / predicted probability", ylabel="density", xlim=(0, 1))
        style(axis)
        stats_rows.append({
            "model": label, "n_unique_items": len(data),
            "pass_rate_correlation": corr, "within_benchmark_correlation": within,
            "prediction_sd": float(data.predicted_probability.std()),
            "observed_pass_rate_sd": float(data.observed_pass_rate.std()),
        })
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.60, -0.44), ncol=2, fontsize=6.6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.72, bottom=0.32, wspace=0.34)
    pdf.savefig(fig)
    plt.close(fig)
    return stats_rows


def _density(counts: list[int], edges: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    width = np.diff(edges)
    return values / max(values.sum(), 1.0) / width


def _tvd(left: list[int], right: list[int]) -> float:
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return float(0.5 * np.abs(p / max(p.sum(), 1.0) - q / max(q.sum(), 1.0)).sum())


def _add_split_shift_page(pdf: PdfPages, split: dict) -> list[dict]:
    """Show the full filtered benchmark split, before small-panel sampling."""
    roles = split["roles"]
    edges = {key: np.asarray(value, dtype=float) for key, value in split["histogram_edges"].items()}
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.25), squeeze=False)
    header(
        fig,
        "How different are the benchmark-holdout train, validation, and test distributions?",
        "Full leakage-safe filtered split before panel sampling. Different colours are disjoint benchmark sets; histogram total-variation distance (TVD) is relative to train.",
    )
    table_axis = axes[0, 0]
    table_axis.axis("off")
    cell_text = []
    rows = []
    for role in ROLES:
        stats = roles[role]
        item = stats["item_pass_rate"]
        tvd = _tvd(roles["train"]["item_pass_rate"]["histogram"], item["histogram"])
        rows.append({"split": role, "responses": stats["responses"], "tasks": stats["tasks"],
                     "items": stats["items"], "mean_label": stats["mean_label"],
                     "item_pass_rate_std": item["std"], "item_pass_rate_tvd_vs_train": tvd,
                     "bge_centroid_cosine_to_train": split["bge_m3_long_centroid_cosine"]["train"][role]})
        cell_text.append([
            f"{len(stats['benchmarks'])}\n{', '.join(stats['benchmarks'])}",
            f"{stats['responses']:,}", f"{stats['tasks']:,}", f"{stats['items']:,}",
            f"{stats['mean_label']:.3f}", f"{item['std']:.3f}", f"{tvd:.3f}",
        ])
    table = table_axis.table(
        cellText=cell_text, rowLabels=["train", "validation", "test"],
        colLabels=["benchmarks", "cells", "tasks", "items", "mean y", "sd item p", "TVD item p"],
        cellLoc="center", rowLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.7)
    table.scale(1.00, 1.85)
    table_axis.set_title("Split sizes and item-pass-rate shift", loc="left", fontsize=7.5, fontweight="bold")
    for key, title, axis, hist_key in (
        ("item_pass_rate", "Item pass rate", axes[0, 1], "probability"),
        ("task_pass_rate", "Task (subject × benchmark) pass rate", axes[1, 0], "probability"),
        ("item_responders", "Item support", axes[1, 1], "log10_item_responders"),
    ):
        for role in ROLES:
            values = roles[role][key]["histogram" if key != "item_responders" else "histogram_log10"]
            axis.stairs(_density(values, edges[hist_key]), edges[hist_key], color=ROLE_COLOURS[role],
                        linewidth=1.4, label=role)
        axis.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
        axis.set_ylabel("density")
        axis.set_xlabel("pass rate" if hist_key == "probability" else r"$\log_{10}$(responders per item)")
        style(axis)
    axes[0, 1].legend(loc="upper right", fontsize=6.2)
    cos = split["bge_m3_long_centroid_cosine"]
    axes[1, 1].text(0.98, 0.96,
                     "BGE centroid cosine to train\n"
                     f"validation {cos['train']['validation']:.4f}\n"
                     f"test       {cos['train']['test']:.4f}",
                     transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=5.9,
                     bbox={"facecolor": "white", "edgecolor": "#C7CED6", "pad": 2.5})
    fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.13, hspace=0.57, wspace=0.30)
    pdf.savefig(fig)
    plt.close(fig)
    return rows


def _add_figure11_page(pdf: PdfPages, per_model: dict[str, dict]) -> list[dict]:
    benchmarks = sorted(per_model["filtered"]["fig11"]["benchmark_id"].unique())
    fig, axes = plt.subplots(len(benchmarks), len(BUDGETS), figsize=(TEXTWIDTH, 4.25), squeeze=False)
    header(
        fig,
        "Figure 11 analog: does a few-label estimate recover evaluation-pool accuracy?",
        "Random streaming acquisition only. Each observation is one task and candidate order. Curves show estimate minus that task's held-out evaluation-pool accuracy.",
    )
    bins = np.linspace(-0.75, 0.75, 49)
    rows = []
    for row_index, benchmark in enumerate(benchmarks):
        for column_index, budget in enumerate(BUDGETS):
            axis = axes[row_index, column_index]
            reference = per_model["filtered"]["fig11"]
            ctt_data = reference[(reference["benchmark_id"] == benchmark) & (reference["budget"] == budget)]
            ctt_error = ctt_data["ctt"] - ctt_data["truth"]
            axis.hist(ctt_error, bins=bins, density=True, histtype="stepfilled", color="#D5DBE1", label="CTT")
            annotation = [f"CTT  bias {ctt_error.mean():+.2f}  rmse {np.sqrt(np.mean(ctt_error ** 2)):.2f}"]
            for key, _, label, colour in MODELS:
                data = per_model[key]["fig11"]
                data = data[(data["benchmark_id"] == benchmark) & (data["budget"] == budget)]
                error = data["model"] - data["truth"]
                bias = float(error.mean())
                rmse = float(np.sqrt(np.mean(error ** 2)))
                axis.hist(error, bins=bins, density=True, histtype="step", linewidth=1.35, color=colour, label=label)
                annotation.append(f"{key:8} {bias:+.2f} / {rmse:.2f}")
                rows.append({"benchmark_id": benchmark, "budget": budget, "model": label,
                             "bias": bias, "rmse": rmse, "n_task_orders": len(data)})
            axis.axvline(0, color="black", linewidth=0.6)
            axis.text(0.98, 0.96, "\n".join(annotation), transform=axis.transAxes, ha="right", va="top",
                      fontsize=5.25, family="monospace")
            axis.set_title(f"{benchmark}, B={budget}", loc="left", fontsize=7.1, fontweight="bold")
            if row_index == len(benchmarks) - 1:
                axis.set_xlabel("estimate - truth")
            if column_index == 0:
                axis.set_ylabel("density")
            style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=6.4)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.17, hspace=0.43, wspace=0.28)
    pdf.savefig(fig)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "docs" / "model-results-visualization.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model-results-visualization.pdf")
    parser.add_argument("--summary-csv", type=Path,
                        default=ROOT / "docs" / "np-online-passrate-figure11-summary.csv")
    parser.add_argument("--distribution-json", type=Path, required=True,
                        help="aggregate summary from export_holdout_split_distribution")
    args = parser.parse_args()
    per_model: dict[str, dict] = {}
    for key, run_name, _, _ in MODELS:
        run_dir = args.runs_root / run_name
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        decisions = pd.read_parquet(run_dir / "acquisition_decisions.parquet")
        per_model[key] = {
            "items": _item_distribution(predictions),
            "fig11": _task_estimates(predictions, decisions),
        }
    initial_pages = len(PdfReader(args.base).pages)
    split = json.loads(args.distribution_json.read_text(encoding="utf-8"))
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        shift_rows = _add_split_shift_page(pdf, split)
        stats_rows = _add_passrate_page(pdf, per_model)
        fig11_rows = _add_figure11_page(pdf, per_model)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats_rows).to_csv(args.summary_csv, index=False)
    pd.DataFrame(fig11_rows).to_csv(args.summary_csv.with_name(args.summary_csv.stem + "-errors.csv"), index=False)
    pd.DataFrame(shift_rows).to_csv(ROOT / "docs" / "benchmark-holdout-split-distribution-summary.csv", index=False)
    buffer.seek(0)
    writer = PdfWriter()
    for page in PdfReader(args.base).pages:
        writer.add_page(page)
    for page in PdfReader(buffer).pages:
        writer.add_page(page)
    temporary = args.output.with_suffix(".tmp.pdf")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(args.output)
    final_pages = len(PdfReader(args.output).pages)
    if final_pages != initial_pages + 3:
        raise RuntimeError(f"expected {initial_pages + 3} pages after append, got {final_pages}")
    print({"base_pages": initial_pages, "final_pages": final_pages,
           "output": str(args.output), "summary": str(args.summary_csv)})


if __name__ == "__main__":
    main()
