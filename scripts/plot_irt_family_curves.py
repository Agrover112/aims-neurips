#!/usr/bin/env python3
"""Plot the available baseline and clearly labelled illustrative IRT-family curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUDGETS = np.array([0, 1, 3, 7, 15, 31])
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def main() -> None:
    baseline_path = Path("/tmp/beta-binomial-full-metrics.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    beta = np.array([baseline["by_budget"][str(int(b))]["brier"] for b in BUDGETS])

    # Hypothetical values solely for visualizing the intended comparison.
    curves = {
        "Beta-binomial (REAL)": beta,
        "1PL (illustrative)": np.array([0.238, 0.229, 0.218, 0.205, 0.196, 0.190]),
        "2PL (illustrative)": np.array([0.231, 0.222, 0.210, 0.197, 0.188, 0.182]),
        "Amortized 1PL (illustrative)": np.array([0.218, 0.211, 0.201, 0.190, 0.181, 0.175]),
        "Amortized 2PL (illustrative)": np.array([0.207, 0.200, 0.190, 0.179, 0.171, 0.165]),
    }
    colors = ["#F28E2B", "#9C9C9C", "#59A14F", "#4C78A8", "#E45756"]
    markers = ["o", "s", "D", "^", "P"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.4, 5.7), constrained_layout=True)
    for index, ((name, values), color, marker) in enumerate(
        zip(curves.items(), colors, markers, strict=True)
    ):
        alc = float(values @ WEIGHTS)
        ax.plot(
            BUDGETS,
            values,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=3 if index == 0 else 2.2,
            linestyle="-" if index == 0 else "--",
            label=f"{name}  ·  ALC={alc:.3f}",
        )

    ax.set(
        title="How the model-family comparison would look\n"
        "Only beta-binomial is measured; IRT-family curves are hypothetical",
        xlabel="Acquired labels",
        ylabel="Brier score (lower is better)",
        xticks=BUDGETS,
    )
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.text(
        0.02,
        0.03,
        r"Brier ALC = $0.1B_0 + 0.2B_1 + 0.2B_3 + 0.2B_7 + 0.2B_{15} + 0.1B_{31}$",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )
    output = Path("results/figures/irt_family_brier_curves.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()
