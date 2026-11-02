#!/usr/bin/env python3
"""Create a fabricated gallery showing proposed PAIEC paper diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    budgets = np.array([0, 1, 3, 7, 15, 31])

    # A: posterior/state trajectories. Values are deliberately illustrative.
    ax = axes[0, 0]
    theta_mean = np.array([0.00, 0.28, 0.13, 0.39, 0.48, 0.52])
    theta_sd = np.array([1.00, 0.78, 0.58, 0.39, 0.27, 0.19])
    shift_mean = np.array([0.00, -0.16, -0.09, -0.04, 0.01, 0.03])
    shift_sd = np.array([0.65, 0.51, 0.39, 0.28, 0.20, 0.14])
    ax.fill_between(
        budgets, theta_mean - 1.96 * theta_sd, theta_mean + 1.96 * theta_sd,
        color="#4C78A8", alpha=0.18,
    )
    ax.plot(budgets, theta_mean, color="#4C78A8", marker="o", lw=2.5, label=r"ability $\theta$")
    ax.fill_between(
        budgets, shift_mean - 1.96 * shift_sd, shift_mean + 1.96 * shift_sd,
        color="#E45756", alpha=0.16,
    )
    ax.plot(
        budgets, shift_mean, color="#E45756", marker="s", lw=2.3,
        label=r"calibration shift $\delta$",
    )
    ax.axhline(0, color="0.35", lw=1, ls="--")
    ax.set(
        title="A  Online parameter adaptation with 95% uncertainty",
        xlabel="acquired labels",
        ylabel="posterior estimate",
        xticks=budgets,
    )
    ax.legend(frameon=True)
    ax.text(
        0.02, 0.04,
        r"In standard AIRT, item $a_j,b_j$ stay fixed; $\theta$, calibration, or NP state adapts.",
        transform=ax.transAxes, fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    # B: acquisition curves and an interpretable label-efficiency statement.
    ax = axes[0, 1]
    acquisition = {
        "Random": np.array([0.220, 0.216, 0.208, 0.198, 0.187, 0.178]),
        "Predictive uncertainty": np.array([0.220, 0.211, 0.200, 0.187, 0.176, 0.169]),
        "Fisher information": np.array([0.220, 0.208, 0.195, 0.181, 0.171, 0.165]),
    }
    for (name, values), color, marker in zip(
        acquisition.items(), ["#9C9C9C", "#59A14F", "#F28E2B"], ["o", "s", "D"], strict=True
    ):
        ax.plot(budgets, values, marker=marker, color=color, lw=2.5, label=name)
    target = 0.185
    ax.axhline(target, color="0.25", ls="--", lw=1.2, label=f"target Brier={target:.3f}")
    ax.annotate(
        "Fisher reaches target\nwith ≈7 labels",
        xy=(7, 0.181), xytext=(10, 0.207),
        arrowprops={"arrowstyle": "->", "color": "0.25"}, fontsize=9,
    )
    ax.set(
        title="B  Acquisition efficiency",
        xlabel="acquired labels",
        ylabel="Brier score (lower is better)",
        xticks=budgets,
    )
    ax.legend(frameon=True, fontsize=9)

    # C: benchmark-by-model heatmap as delta from a shared baseline.
    ax = axes[1, 0]
    benchmarks = ["Reasoning", "Safety", "Coding", "Knowledge", "Multimodal", "Agents"]
    models = ["1PL", "2PL", "Amort. 1PL", "Amort. 2PL", "ANP-AIRT"]
    delta = np.array(
        [
            [-0.004, -0.011, -0.018, -0.026, -0.031],
            [+0.003, +0.007, -0.008, -0.006, -0.012],
            [-0.002, -0.015, -0.021, -0.034, -0.038],
            [-0.006, -0.009, -0.019, -0.023, -0.025],
            [+0.006, +0.002, -0.004, -0.013, -0.021],
            [+0.001, -0.005, -0.010, -0.019, -0.033],
        ]
    )
    image = ax.imshow(delta, cmap="RdBu_r", vmin=-0.04, vmax=0.04, aspect="auto")
    for row in range(delta.shape[0]):
        for column in range(delta.shape[1]):
            value = delta[row, column]
            ax.text(
                column, row, f"{value:+.3f}", ha="center", va="center",
                color="white" if abs(value) > 0.022 else "black", fontsize=8.5,
            )
    ax.set(
        title="C  Benchmark × model heatmap: Δ Brier ALC vs Beta-Binomial",
        xticks=np.arange(len(models)),
        xticklabels=models,
        yticks=np.arange(len(benchmarks)),
        yticklabels=benchmarks,
    )
    ax.tick_params(axis="x", rotation=28)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Δ Brier ALC (negative = improvement)")

    # D: paired ablation effects with benchmark-clustered confidence intervals.
    ax = axes[1, 1]
    components = [
        "Item embeddings",
        "Learn discrimination a",
        "Online ability update",
        "Anchor calibration",
        "Fisher acquisition",
        "Cross-attention",
        "Relation-aware attention",
    ]
    effects = np.array([-0.017, -0.009, -0.014, -0.011, -0.006, -0.003, -0.008])
    ci_low = np.array([-0.026, -0.017, -0.022, -0.018, -0.011, -0.010, -0.015])
    ci_high = np.array([-0.008, -0.001, -0.006, -0.004, -0.001, +0.004, -0.001])
    y = np.arange(len(components))
    xerr = np.vstack([effects - ci_low, ci_high - effects])
    colors = ["#4C78A8" if high < 0 else "#BAB0AC" for high in ci_high]
    for index in range(len(components)):
        ax.errorbar(
            effects[index], y[index],
            xerr=np.array([[xerr[0, index]], [xerr[1, index]]]),
            fmt="o", color=colors[index], ecolor=colors[index],
            markersize=7, capsize=4, lw=2,
        )
    ax.axvline(0, color="0.25", ls="--", lw=1.2)
    ax.set(
        title="D  Ablation forest plot: paired component effects",
        xlabel="Δ Brier ALC versus parent model (negative = better)",
        yticks=y,
        yticklabels=components,
    )
    ax.invert_yaxis()
    ax.text(
        0.98, 0.97, "Intervals crossing zero = inconclusive",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )

    fig.suptitle(
        "ILLUSTRATIVE ONLY — proposed PAIEC paper diagnostics (all values fabricated)",
        color="#B22222", fontsize=16, fontweight="bold",
    )
    output = Path("results/figures/illustrative_paper_figure_gallery.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()
