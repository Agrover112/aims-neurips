#!/usr/bin/env python3
"""Append the ANP-128/Qwen item-pass-rate diagnostic to the model PDF."""

from __future__ import annotations

import argparse
import io
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


def _correlation(x: pd.Series, y: pd.Series) -> float:
    x, y = x.to_numpy(float), y.to_numpy(float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _within_benchmark(data: pd.DataFrame) -> float:
    centred = data.assign(
        p=data["predicted_probability"] - data.groupby("benchmark_id")["predicted_probability"].transform("mean"),
        y=data["observed_pass_rate"] - data.groupby("benchmark_id")["observed_pass_rate"].transform("mean"),
    )
    return _correlation(centred["p"], centred["y"])


def _page(pdf: PdfPages, data: pd.DataFrame) -> list[dict]:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.25), constrained_layout=False)
    header(
        fig,
        "Is ANP-128 with Qwen embeddings learning the item pass-rate distribution?",
        "Original non-benchmark-holdout split, budget 0. Grey is observed pass rate among at least five responses in that role; colour is the ANP prior prediction with an empty context. These are diagnostic predictions, not calibrated probabilities.",
    )
    rows = []
    for axis, split in zip(axes, ("train", "validation", "test"), strict=True):
        subset = data[(data["split"] == split) & (data["n_responses"] >= 5)].copy()
        corr = _correlation(subset["predicted_probability"], subset["observed_pass_rate"])
        within = _within_benchmark(subset)
        axis.hist(subset["observed_pass_rate"], bins=25, range=(0, 1), density=True,
                  color="#D5DBE1", label="observed")
        axis.hist(subset["predicted_probability"], bins=25, range=(0, 1), density=True,
                  histtype="step", linewidth=1.6, color=MUTED["indigo"], label="ANP prior")
        axis.set_title(split, loc="left", fontsize=7.8, fontweight="bold")
        axis.text(
            0.02, 0.96,
            f"items {len(subset):,}\ncorr {corr:.2f}\nwithin-bench {within:.2f}\n"
            f"sd: pred {subset.predicted_probability.std():.2f}, obs {subset.observed_pass_rate.std():.2f}",
            transform=axis.transAxes, va="top", fontsize=6.3,
        )
        axis.set(xlabel="pass rate / predicted probability", ylabel="density", xlim=(0, 1))
        style(axis)
        rows.append({
            "split": split, "n_items_response_ge_5": len(subset), "correlation": corr,
            "within_benchmark_correlation": within,
            "prediction_sd": float(subset.predicted_probability.std()),
            "observed_pass_rate_sd": float(subset.observed_pass_rate.std()),
            "prediction_mean": float(subset.predicted_probability.mean()),
            "observed_pass_rate_mean": float(subset.observed_pass_rate.mean()),
        })
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.60, -0.44), ncol=2, fontsize=6.6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.72, bottom=0.32, wspace=0.34)
    pdf.savefig(fig)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "docs" / "model-results-visualization.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model-results-visualization.pdf")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "docs" / "non-benchmark-anp128-qwen-item-passrate-summary.csv")
    args = parser.parse_args()
    data = pd.read_parquet(args.input)
    expected = {"split", "benchmark_id", "item_uid", "observed_pass_rate", "n_responses", "predicted_probability"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"input lacks columns: {sorted(missing)}")
    before = len(PdfReader(args.base).pages)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        rows = _page(pdf, data)
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
    after = len(PdfReader(args.output).pages)
    if after != before + 1:
        raise RuntimeError(f"expected {before + 1} pages, got {after}")
    pd.DataFrame(rows).to_csv(args.summary, index=False)
    print({"base_pages": before, "final_pages": after, "output": str(args.output), "summary": str(args.summary)})


if __name__ == "__main__":
    main()
