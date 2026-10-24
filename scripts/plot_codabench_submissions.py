#!/usr/bin/env python3
"""Compare all completed Codabench submissions from their scoring logs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_codabench_scoring import parse  # noqa: E402

BUDGETS = (0, 1, 3, 7, 15, 31)
RUNS = (
    (1005155, "Shrunken mean", "#000000"),
    (1005187, "CNP-AIRT · random", "#377eb8"),
    (1005210, "Online-AIRT · active", "#4daf4a"),
    (1005225, "CNP-AIRT · active", "#984ea3"),
    (1005650, "Empirical mean", "#e41a1c"),
)


def main() -> int:
    output = ROOT / "results" / "codabench-comparison"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    x = np.arange(len(BUDGETS))

    for submission_id, label, color in RUNS:
        log = ROOT / "results" / f"codabench-{submission_id}" / "scoring_stdout.txt"
        alc, by_budget = parse(log)
        pairs = sorted(alc)
        brier = np.asarray(
            [[by_budget[b][pair][1] for b in BUDGETS] for pair in pairs], dtype=float
        )
        ece = np.asarray(
            [[by_budget[b][pair][2] for b in BUDGETS] for pair in pairs], dtype=float
        )
        alc_mean = float(np.mean([row[1] for row in alc.values()]))
        ece_mean = float(np.mean([row[2] for row in alc.values()]))
        summaries.append((submission_id, label, len(pairs), alc_mean, ece_mean, brier, ece))

        for axis, values in ((axes[0], brier), (axes[1], ece)):
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            axis.fill_between(
                x,
                np.maximum(0.0, mean - sd),
                np.minimum(1.0, mean + sd),
                color=color,
                alpha=0.08,
                linewidth=0,
            )
            axis.plot(x, mean, marker="o", linewidth=2.2, color=color, label=label)

    axes[0].set_title("Brier score by acquired-label budget")
    axes[0].set_ylabel("Mean Brier ↓")
    axes[1].set_title("Calibration error by acquired-label budget")
    axes[1].set_ylabel("Mean ECE ↓")
    for axis in axes:
        axis.set_xlabel("Labels acquired")
        axis.set_xticks(x, BUDGETS)
        axis.set_ylim(bottom=0)
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Codabench formative submissions\nlines = pair means; shading = ±1 SD across hidden subject–benchmark pairs",
        fontsize=14,
    )
    figure_path = output / "all_submissions_budget_curves.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)

    with (output / "submission_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["submission_id", "method", "hidden_pairs", "brier_alc", "ece"]
            + [f"brier_b{b}" for b in BUDGETS]
            + [f"ece_b{b}" for b in BUDGETS]
        )
        for submission_id, label, n, alc_mean, ece_mean, brier, ece in summaries:
            writer.writerow(
                [submission_id, label, n, alc_mean, ece_mean]
                + list(brier.mean(axis=0))
                + list(ece.mean(axis=0))
            )

    print(figure_path)
    for submission_id, label, n, alc_mean, ece_mean, brier, ece in summaries:
        print(
            submission_id,
            label,
            f"n={n}",
            f"ALC={alc_mean:.6f}",
            f"ECE={ece_mean:.6f}",
            "Brier=" + ",".join(f"{v:.4f}" for v in brier.mean(axis=0)),
            "ECEcurve=" + ",".join(f"{v:.4f}" for v in ece.mean(axis=0)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
