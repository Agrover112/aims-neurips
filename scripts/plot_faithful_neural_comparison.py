"""Plot the three-seed paper-faithful NP-family BeaverTails experiment."""

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

RUN = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "faithful-np-beavertails-3seed-v1")
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
MODELS = [
    "online_airt_2pl",
    "beta_binomial",
    "cnp_airt",
    "anp",
    "stacked_mhsa_anp",
    "tnpd_airt",
    "np",
]
LABELS = {model: display_name(model) for model in MODELS}
COLORS = {
    "online_airt_2pl": "#0072B2",
    "beta_binomial": "#555555",
    "cnp_airt": "#56B4E9",
    "np": "#D55E00",
    "anp": "#009E73",
    "stacked_mhsa_anp": "#CC79A7",
    "tnpd_airt": "#E69F00",
}


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(RUN / ".matplotlib"))
    metrics_path = RUN / "metrics-calibration.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    aggregate = metrics["aggregate"]
    cnp_path = ROOT / "results" / "cnp-airt-beavertails-3seed-v1" / "metrics.json"
    if cnp_path.exists():
        aggregate.update(json.loads(cnp_path.read_text(encoding="utf-8"))["aggregate"])
    available = [model for model in MODELS if model in aggregate]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.35, 1.35, 1]})

    for model in available:
        curve = aggregate[model]["by_budget"]
        mean = np.array([curve[str(b)]["brier"] for b in BUDGETS])
        se = np.array([curve[str(b)].get("brier_se", 0.0) for b in BUDGETS])
        axes[0].plot(
            BUDGETS,
            mean,
            marker="o",
            linewidth=2.4 if model == "online_airt_2pl" else 1.8,
            markersize=5,
            label=LABELS[model],
            color=COLORS[model],
        )
        axes[0].fill_between(
            BUDGETS,
            mean - 1.96 * se,
            mean + 1.96 * se,
            color=COLORS[model],
            alpha=0.075,
            linewidth=0,
        )
        ece_rows = [curve[str(b)] for b in BUDGETS]
        if all("ece" in row for row in ece_rows):
            ece = np.array([row["ece"] for row in ece_rows])
            ece_se = np.array([row.get("ece_se", 0.0) for row in ece_rows])
            axes[1].plot(
                BUDGETS, ece, marker="o", linewidth=1.8, markersize=5,
                label=LABELS[model], color=COLORS[model],
            )
            axes[1].fill_between(
                BUDGETS, ece - 1.96 * ece_se, ece + 1.96 * ece_se,
                color=COLORS[model], alpha=0.075, linewidth=0,
            )

    ordered = sorted(available, key=lambda name: aggregate[name]["brier_alc"])
    alc = np.array([aggregate[name]["brier_alc"] for name in ordered])
    seed_sd = np.array([aggregate[name].get("brier_alc_seed_std", 0.0) for name in ordered])
    bars = axes[2].barh(
        [LABELS[name] for name in ordered],
        alc,
        xerr=seed_sd,
        capsize=3,
        color=[COLORS[name] for name in ordered],
        alpha=0.9,
    )
    axes[2].invert_yaxis()
    axes[2].bar_label(bars, labels=[f"{x:.3f}" for x in alc], padding=4, fontsize=9)

    axes[0].set_title("A  Brier score across acquisition budgets", loc="left", fontweight="bold")
    axes[0].set_xlabel("Acquired labels")
    axes[0].set_ylabel("Brier score ↓")
    axes[0].set_xticks(BUDGETS)
    curve_values = [aggregate[m]["by_budget"][str(b)]["brier"] for m in available for b in BUDGETS]
    axes[0].set_ylim(min(curve_values) * 0.85, max(curve_values) * 1.03)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8.8, ncol=2, loc="upper right")
    axes[0].text(
        0.02,
        0.03,
        "Shading: 95% CI across 12 held-out evaluations",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#444444",
    )

    axes[1].set_title("B  Calibration across budgets", loc="left", fontweight="bold")
    axes[1].set_xlabel("Acquired labels")
    axes[1].set_ylabel("ECE ↓")
    axes[1].set_xticks(BUDGETS)
    axes[1].grid(alpha=0.2)

    axes[2].set_title("C  Budget-weighted summary", loc="left", fontweight="bold")
    axes[2].set_xlabel("Brier ALC ↓")
    axes[2].set_xlim(0, alc.max() * 1.28)
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].text(
        0.98,
        0.03,
        "Error bars: ±1 SD across seeds",
        transform=axes[2].transAxes,
        ha="right",
        fontsize=8,
        color="#444444",
    )

    figure.suptitle(
        "Faithful NP-family comparison on BeaverTails",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "MEASURED RESULTS · 3 seeds · 4 leave-one-subject-out folds/seed · "
        "100 meta-training tasks from 39 benchmarks · "
        "dual-likelihood NP training + IRT MAP anchor, v2-fixed",
        ha="center",
        fontsize=8.8,
        color="#8A2D2D",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    figure.savefig(RUN / "comparison.png", dpi=240, bbox_inches="tight")
    figure.savefig(RUN / "comparison.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
