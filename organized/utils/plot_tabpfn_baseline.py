"""Compare off-the-shelf TabPFN-v2 with the measured BeaverTails controls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aims_starter.model_names import display_name

FAITHFUL = ROOT / "results" / "faithful-np-beavertails-3seed-v1"
RUN = ROOT / "results" / "tabpfn-v2-beavertails-3seed-v1"
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
MODELS = ("online_airt_2pl", "beta_binomial", "tabpfn_v2")
LABELS = {model: display_name(model) for model in MODELS}
COLORS = {
    "online_airt_2pl": "#0072B2",
    "beta_binomial": "#555555",
    "tabpfn_v2": "#E69F00",
}


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(RUN / ".matplotlib"))
    faithful = json.loads((FAITHFUL / "metrics.json").read_text(encoding="utf-8"))["aggregate"]
    tabpfn = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))["aggregate"]
    aggregate = {**faithful, **tabpfn}
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), gridspec_kw={"width_ratios": [1.45, 1, 0.9]})

    for model in MODELS:
        curve = aggregate[model]["by_budget"]
        mean = np.array([curve[str(b)]["brier"] for b in BUDGETS])
        se = np.array([curve[str(b)]["brier_se"] for b in BUDGETS])
        axes[0].plot(
            BUDGETS,
            mean,
            marker="o",
            linewidth=2.3,
            label=LABELS[model],
            color=COLORS[model],
        )
        axes[0].fill_between(
            BUDGETS,
            mean - 1.96 * se,
            mean + 1.96 * se,
            color=COLORS[model],
            alpha=0.09,
            linewidth=0,
        )

    alc = np.array([aggregate[model]["brier_alc"] for model in MODELS])
    seed_sd = np.array([aggregate[model]["brier_alc_seed_std"] for model in MODELS])
    bars = axes[1].barh(
        [LABELS[model] for model in MODELS],
        alc,
        xerr=seed_sd,
        color=[COLORS[model] for model in MODELS],
        capsize=3,
    )
    axes[1].invert_yaxis()
    axes[1].bar_label(bars, labels=[f"{value:.4f}" for value in alc], padding=4, fontsize=9)

    availability = np.array(
        [tabpfn["tabpfn_v2"]["by_budget"][str(b)]["tabpfn_evaluations"] / 12 for b in BUDGETS]
    )
    budget_positions = np.arange(len(BUDGETS))
    axes[2].bar(budget_positions, availability * 100, color=COLORS["tabpfn_v2"], width=0.62)
    for x, value in zip(budget_positions, availability, strict=True):
        axes[2].text(x, value * 100 + 2.5, f"{value:.0%}", ha="center", fontsize=9)

    axes[0].set_title("A  Predictive performance", loc="left", fontweight="bold")
    axes[0].set_xlabel("Acquired labels")
    axes[0].set_ylabel("Brier score ↓")
    axes[0].set_xticks(BUDGETS)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].text(
        0.02,
        0.04,
        "Shading: 95% CI across 12 evaluations",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#444444",
    )
    axes[1].set_title("B  Budget-weighted summary", loc="left", fontweight="bold")
    axes[1].set_xlabel("Brier ALC ↓")
    axes[1].set_xlim(0, 0.235)
    axes[1].grid(axis="x", alpha=0.2)
    axes[2].set_title("C  TabPFN usable contexts", loc="left", fontweight="bold")
    axes[2].set_xlabel("Acquired labels")
    axes[2].set_ylabel("Evaluations with both classes")
    axes[2].set_xticks(budget_positions, labels=BUDGETS)
    axes[2].set_ylim(0, 110)
    axes[2].set_yticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
    axes[2].grid(axis="y", alpha=0.2)

    figure.suptitle("Off-the-shelf TabPFN-v2 on BeaverTails", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "MEASURED RESULTS · 3 seeds · 4 frozen folds/seed · Beta(2,2) fallback when fewer than two classes are observed",
        ha="center",
        fontsize=8.7,
        color="#8A2D2D",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    figure.savefig(RUN / "comparison.png", dpi=240, bbox_inches="tight")
    figure.savefig(RUN / "comparison.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
