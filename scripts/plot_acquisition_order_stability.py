"""Plot ten-order acquisition stability results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "faithful-np-beavertails-3seed-v3-residual"
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
NAMES = {
    "online_airt_2pl": "Online AIRT-2PL",
    "tnpd_airt": "TNP-D AIRT",
    "anp": "ANP-AIRT",
    "stacked_mhsa_anp": "Stacked MHSA ANP",
    "np": "NP-AIRT",
    "beta_binomial": "Beta-binomial",
    "cnp_airt": "CNP-AIRT",
}
COLORS = {
    "online_airt_2pl": "#0072B2",
    "tnpd_airt": "#E69F00",
    "anp": "#009E73",
    "stacked_mhsa_anp": "#CC79A7",
    "np": "#D55E00",
    "beta_binomial": "#555555",
    "cnp_airt": "#56B4E9",
}


def main() -> None:
    data = json.loads((RUN / "metrics-acquisition-orders.json").read_text())["aggregate"]
    cnp_path = (
        ROOT / "results" / "cnp-airt-beavertails-3seed-v2-residual" / "metrics-acquisition-orders.json"
    )
    if cnp_path.exists():
        data.update(json.loads(cnp_path.read_text())["aggregate"])
    models = sorted(data, key=lambda model: data[model]["brier_alc"])
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), gridspec_kw={"width_ratios": [1.35, 1]})
    for model in models:
        curve = data[model]["by_budget"]
        mean = np.array([curve[str(b)]["brier"] for b in BUDGETS])
        std = np.array([curve[str(b)]["brier_order_std"] for b in BUDGETS])
        axes[0].plot(BUDGETS, mean, marker="o", label=NAMES[model], color=COLORS[model])
        axes[0].fill_between(BUDGETS, mean - std, mean + std, color=COLORS[model], alpha=0.10)
    values = [data[model]["brier_alc"] for model in models]
    errors = [data[model]["brier_alc_order_std"] for model in models]
    bars = axes[1].barh(
        [NAMES[model] for model in models],
        values,
        xerr=errors,
        color=[COLORS[model] for model in models],
        capsize=4,
    )
    axes[1].invert_yaxis()
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
    axes[0].set(
        title="A  Performance across label budgets",
        xlabel="Acquired labels",
        ylabel="Brier score ↓",
    )
    axes[0].set_xticks(BUDGETS)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2)
    axes[1].set(title="B  Stability across reveal orders", xlabel="Brier ALC ↓")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Ten-order acquisition robustness on BeaverTails", fontweight="bold", fontsize=15
    )
    figure.text(
        0.5,
        0.015,
        "Mean ± 1 SD across 10 shared deterministic label-reveal orders · no retraining",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(RUN / "acquisition-order-stability.png", dpi=240, bbox_inches="tight")
    figure.savefig(RUN / "acquisition-order-stability.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
