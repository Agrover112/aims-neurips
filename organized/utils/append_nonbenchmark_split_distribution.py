#!/usr/bin/env python3
"""Append the original random task/item split distribution to the model PDF."""

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


ROLES = ("train", "validation", "test")
COLOURS = {"train": MUTED["indigo"], "validation": MUTED["rose"], "test": MUTED["teal"]}


def _density(counts: list[int], edges: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    return values / max(values.sum(), 1.0) / np.diff(edges)


def _tvd(left: list[int], right: list[int]) -> float:
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return float(0.5 * np.abs(p / max(p.sum(), 1.0) - q / max(q.sum(), 1.0)).sum())


def _page(pdf: PdfPages, payload: dict) -> list[dict]:
    roles = payload["roles"]
    edges = {key: np.asarray(value, dtype=float) for key, value in payload["histogram_edges"].items()}
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.25), squeeze=False)
    header(
        fig,
        "How different are splits in the original non-benchmark-holdout NP training?",
        "Exact standard NP split: random subject × benchmark tasks, plus a global 20% held-item set removed from train/validation. Test below retains all task items for the later response/item evaluation axes.",
    )
    table_axis = axes[0, 0]
    table_axis.axis("off")
    rows = []
    cells = []
    for role in ROLES:
        stats = roles[role]
        tvd = _tvd(roles["train"]["item_pass_rate"]["histogram"], stats["item_pass_rate"]["histogram"])
        cells.append([
            f"{len(stats['benchmarks'])}", f"{stats['responses']:,}", f"{stats['tasks']:,}",
            f"{stats['items']:,}", f"{stats['mean_label']:.3f}",
            f"{stats['item_pass_rate']['std']:.3f}", f"{tvd:.3f}",
        ])
        rows.append({
            "split": role, "responses": stats["responses"], "tasks": stats["tasks"], "items": stats["items"],
            "mean_label": stats["mean_label"], "item_pass_rate_std": stats["item_pass_rate"]["std"],
            "item_pass_rate_tvd_vs_train": tvd,
            "qwen3_8b_centroid_cosine_to_train": payload["qwen3_8b_centroid_cosine"]["train"][role],
            "benchmark_overlap_with_train": (1.0 if role == "train" else payload["overlap_with_train"][role]["benchmark_fraction"]),
            "item_jaccard_with_train": (1.0 if role == "train" else payload["overlap_with_train"][role]["item_jaccard"]),
        })
    table = table_axis.table(
        cellText=cells, rowLabels=list(ROLES),
        colLabels=["# benchmarks", "cells", "tasks", "items", "mean y", "sd item p", "TVD item p"],
        cellLoc="center", rowLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    table.scale(1.0, 1.55)
    overlap = payload["overlap_with_train"]
    table_axis.set_title("Split sizes and pass-rate shift", loc="left", fontsize=7.5, fontweight="bold")
    table_axis.text(
        0.0, -0.22,
        "Overlap with train (benchmark fraction / item Jaccard)\n"
        f"validation: {overlap['validation']['benchmark_fraction']:.2f} / {overlap['validation']['item_jaccard']:.2f}\n"
        f"test:       {overlap['test']['benchmark_fraction']:.2f} / {overlap['test']['item_jaccard']:.2f}",
        transform=table_axis.transAxes, fontsize=6.1, va="top",
    )
    plots = (
        ("item_pass_rate", "Item pass rate", axes[0, 1], "probability", "histogram"),
        ("task_pass_rate", "Task (subject × benchmark) pass rate", axes[1, 0], "probability", "histogram"),
        ("item_responders", "Item support", axes[1, 1], "log10_item_responders", "histogram_log10"),
    )
    for data_key, title, axis, edge_key, count_key in plots:
        for role in ROLES:
            axis.stairs(_density(roles[role][data_key][count_key], edges[edge_key]), edges[edge_key],
                        color=COLOURS[role], linewidth=1.4, label=role)
        axis.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
        axis.set_ylabel("density")
        axis.set_xlabel("pass rate" if edge_key == "probability" else r"$\log_{10}$(responders per item)")
        style(axis)
    axes[0, 1].legend(loc="upper right", fontsize=6.2)
    cosine = payload["qwen3_8b_centroid_cosine"]["train"]
    axes[1, 1].text(
        0.98, 0.96,
        "Qwen-8B-MRL centroid cosine to train\n"
        f"validation {cosine['validation']:.4f}\n"
        f"test       {cosine['test']:.4f}",
        transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=5.9,
        bbox={"facecolor": "white", "edgecolor": "#C7CED6", "pad": 2.5},
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.13, hspace=0.57, wspace=0.30)
    pdf.savefig(fig)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "docs" / "model-results-visualization.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model-results-visualization.pdf")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "docs" / "non-benchmark-holdout-split-distribution-summary.csv")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    before = len(PdfReader(args.base).pages)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        rows = _page(pdf, payload)
    buffer.seek(0)
    writer = PdfWriter()
    for page in PdfReader(args.base).pages:
        writer.add_page(page)
    for page in PdfReader(buffer).pages:
        writer.add_page(page)
    temp = args.output.with_suffix(".tmp.pdf")
    with temp.open("wb") as handle:
        writer.write(handle)
    temp.replace(args.output)
    after = len(PdfReader(args.output).pages)
    if after != before + 1:
        raise RuntimeError(f"expected {before + 1} pages, got {after}")
    pd.DataFrame(rows).to_csv(args.summary, index=False)
    print({"base_pages": before, "final_pages": after, "output": str(args.output), "summary": str(args.summary)})


if __name__ == "__main__":
    main()
