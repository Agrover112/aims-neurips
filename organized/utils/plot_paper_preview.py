#!/usr/bin/env python3
"""Build an honest paper-figure preview from stored smoke and baseline metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUDGETS = np.asarray([0, 1, 3, 7, 15, 31])


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/figures/paper_diagnostic_preview.png")
    )
    args = parser.parse_args()

    smoke = _load(args.smoke)
    baseline = _load(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)

    # A: analytical item characteristic curves isolate the extra 2PL parameter.
    ax = axes[0, 0]
    theta = np.linspace(-4, 4, 500)
    colors = ["#4C78A8", "#59A14F", "#E45756"]
    for discrimination, color in zip([0.5, 1.0, 2.0], colors, strict=True):
        probability = 1.0 / (1.0 + np.exp(-discrimination * theta))
        label = f"2PL: a={discrimination:g}, b=0"
        ax.plot(theta, probability, lw=2.5, color=color, label=label)
    ax.axvline(0, color="0.35", lw=1, ls="--")
    ax.set(
        title="A  What 2PL adds beyond 1PL",
        xlabel=r"subject ability $\theta$",
        ylabel="predicted success probability",
        ylim=(-0.02, 1.02),
    )
    ax.legend(frameon=True, fontsize=9, loc="lower right")
    ax.text(
        -3.85,
        0.94,
        "1PL is the a=1 curve; 2PL learns item-specific slope a",
        fontsize=9,
        va="top",
    )

    # B: real integration evidence, explicitly labelled as training-only.
    ax = axes[0, 1]
    losses = [smoke["initial_train_bce"], smoke["final_train_bce"]]
    bars = ax.bar(
        ["before fitting", "after 200 steps"],
        losses,
        color=["#BAB0AC", "#4C78A8"],
        width=0.62,
    )
    for bar, value in zip(bars, losses, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    ax.set(
        title=(
            "B  Amortized-2PL smoke test (real)\n"
            f"{smoke['benchmark_id']} · {smoke['n_tasks']} tasks · "
            f"{smoke['n_subjects']} subjects · train-only integration evidence"
        ),
        ylabel="binary cross-entropy (training)",
        ylim=(0, max(losses) * 1.18),
    )

    # C: real competition-shaped baseline with task-bootstrap uncertainty.
    ax = axes[1, 0]
    task_curves = np.asarray(
        [
            [float(task["by_budget"][str(int(budget))]["brier"]) for budget in BUDGETS]
            for task in baseline["tasks"]
        ]
    )
    aggregate = task_curves.mean(axis=0)
    rng = np.random.default_rng(20261018)
    samples = rng.integers(0, len(task_curves), size=(10_000, len(task_curves)))
    bootstrap_means = task_curves[samples].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.05, 0.95], axis=0)
    ax.fill_between(BUDGETS, lower, upper, color="#F28E2B", alpha=0.22, label="90% task bootstrap")
    ax.plot(BUDGETS, aggregate, marker="o", lw=2.6, color="#F28E2B", label="Beta-binomial")
    ax.axhline(0.25, color="0.35", ls="--", lw=1.2, label="uninformed p=0.5 reference")
    ax.set(
        title=(
            "C  Competition-shaped learning curve (real)\n"
            f"{baseline['n_tasks']} held-out tasks · Brier ALC={baseline['brier_alc']:.3f}"
        ),
        xlabel="acquired labels",
        ylabel="Brier score (lower is better)",
        xticks=BUDGETS,
    )
    ax.legend(frameon=True, fontsize=9)

    # D: real task heterogeneity motivates benchmark-aware and relational models.
    ax = axes[1, 1]
    ordered_tasks = sorted(baseline["tasks"], key=lambda row: float(row["brier_alc"]))
    scores = np.asarray([float(task["brier_alc"]) for task in ordered_tasks])
    labels = [str(task["benchmark_id"]) for task in ordered_tasks]
    unique_benchmarks = list(dict.fromkeys(labels))
    palette = plt.get_cmap("tab10")
    color_map = {name: palette(i % 10) for i, name in enumerate(unique_benchmarks)}
    ax.bar(np.arange(len(scores)), scores, color=[color_map[label] for label in labels], width=0.86)
    ax.axhline(0.25, color="0.25", ls="--", lw=1.2)
    ax.set(
        title="D  Baseline failures vary strongly by task (real)",
        xlabel="held-out subject × benchmark task (sorted)",
        ylabel="Brier ALC",
        xticks=[],
    )
    ax.text(
        0.02,
        0.96,
        f"range {scores.min():.3f}–{scores.max():.3f}\n"
        "This heterogeneity is the paper opportunity",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    fig.suptitle(
        "PAIEC diagnostic preview: current evidence and the Amortized-2PL opportunity",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(args.output)


if __name__ == "__main__":
    main()
