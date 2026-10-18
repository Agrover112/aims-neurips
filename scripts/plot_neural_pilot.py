"""Plot measured BeaverTails held-out-subject neural pilot results."""

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

RUN = ROOT / "results" / "neural-pilot-beavertails-loso-stacked-mhsa-v2-seed20261018"
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
LABELS = {
    "beta_binomial": display_name("beta_binomial"),
    "online_airt_2pl": display_name("online_airt_2pl"),
    "np_airt": display_name("np_airt"),
    "anp_airt": display_name("anp_airt"),
    "stacked_mhsa_anp_airt": display_name("stacked_mhsa_anp_airt"),
}
COLORS = {
    "beta_binomial": "#555555",
    "online_airt_2pl": "#0072B2",
    "np_airt": "#D55E00",
    "anp_airt": "#009E73",
    "stacked_mhsa_anp_airt": "#CC79A7",
}


def main() -> None:
    values = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    aggregate = values["aggregate"]
    os.environ.setdefault("MPLCONFIGDIR", str(RUN / ".matplotlib"))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    for model, label in LABELS.items():
        curve = aggregate[model]["by_budget"]
        mean = np.array([curve[str(budget)]["brier"] for budget in BUDGETS])
        std = np.array([curve[str(budget)]["brier_std"] for budget in BUDGETS])
        axes[0].plot(BUDGETS, mean, marker="o", linewidth=2.2, label=label, color=COLORS[model])
        axes[0].fill_between(BUDGETS, mean - std, mean + std, color=COLORS[model], alpha=0.10)
        ece = np.array([curve[str(budget)]["ece"] for budget in BUDGETS])
        axes[1].plot(BUDGETS, ece, marker="o", linewidth=2.2, label=label, color=COLORS[model])

    axes[0].set_title("A  Predictive accuracy")
    axes[0].set_ylabel("Brier score ↓")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].text(
        0.02,
        0.04,
        "Bands: ±1 SD across 4 held-out subjects",
        transform=axes[0].transAxes,
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    axes[1].set_title("B  Calibration error")
    axes[1].set_ylabel("ECE ↓")

    names = list(LABELS)
    alc = [aggregate[name]["brier_alc"] for name in names]
    bars = axes[2].barh(
        [LABELS[name] for name in names],
        alc,
        color=[COLORS[name] for name in names],
        alpha=0.9,
    )
    axes[2].invert_yaxis()
    axes[2].set_title("C  Budget-weighted summary")
    axes[2].set_xlabel("Brier ALC ↓")
    axes[2].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[2].set_xlim(0, max(alc) * 1.22)

    for axis in axes[:2]:
        axis.set_xlabel("Acquired labels")
        axis.set_xticks(BUDGETS)
        axis.grid(alpha=0.22)
    axes[2].grid(axis="x", alpha=0.22)
    figure.suptitle(
        "Measured BeaverTails pilot — leave-one-subject-out evaluation",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "REAL PILOT RESULTS · 4 folds, one training seed · not an unseen-benchmark or final competition estimate",
        ha="center",
        fontsize=9,
        color="#9C2F2F",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(RUN / "curves.png", dpi=220, bbox_inches="tight")
    figure.savefig(RUN / "curves.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
