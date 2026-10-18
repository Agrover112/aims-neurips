"""Log the measured BeaverTails neural pilot to Weights & Biases."""

from __future__ import annotations

import json
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "neural-pilot-beavertails-loso-stacked-mhsa-v2-seed20261018"


def main() -> None:
    metrics = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    with wandb.init(
        entity="agrgnn-kth-royal-institute-of-technology",
        project="aims-predictive-evaluation",
        name="beavertails-loso-stacked-mhsa-anp-pilot-seed20261018",
        job_type="neural-pilot",
        config=metrics["protocol"],
        tags=["beavertails", "loso", "np-airt", "anp-airt", "stacked-mhsa", "pilot"],
    ) as run:
        for step, budget in enumerate(metrics["protocol"]["online_context_budgets"]):
            row: dict[str, float | int] = {"budget": budget}
            for model, result in metrics["aggregate"].items():
                values = result["by_budget"][str(budget)]
                row[f"brier/{model}"] = values["brier"]
                row[f"brier_std/{model}"] = values["brier_std"]
                row[f"ece/{model}"] = values["ece"]
            run.log(row, step=step)
        for model, result in metrics["aggregate"].items():
            run.summary[f"brier_alc/{model}"] = result["brier_alc"]
        run.summary["evidence_status"] = metrics["status"]
        artifact = wandb.Artifact(
            "beavertails-stacked-mhsa-neural-pilot-seed20261018",
            type="evaluation",
            metadata={"benchmark": "beavertails", "folds": metrics["n_folds"]},
        )
        for filename in ("metrics.json", "curves.png", "curves.pdf"):
            artifact.add_file(str(RUN_DIR / filename))
        run.log_artifact(artifact)
        run.log({"comparison_figure": wandb.Image(str(RUN_DIR / "curves.png"))})


if __name__ == "__main__":
    main()
