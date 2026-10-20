"""Log the three-seed faithful NP-family comparison to Weights & Biases."""

from __future__ import annotations

import json
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "faithful-np-beavertails-3seed-v3-residual"
BUDGETS = (0, 1, 3, 7, 15, 31)


def main() -> None:
    metrics = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    with wandb.init(
        entity="agrgnn-kth-royal-institute-of-technology",
        project="aims-predictive-evaluation",
        name="beavertails-faithful-np-family-3seed-v3-residual",
        job_type="faithful-neural-comparison",
        config={
            "seeds": metrics["seeds"],
            "folds_per_seed": metrics["folds_per_seed"],
            "meta_tasks": 100,
            "meta_benchmarks": 39,
            "training_steps_per_model_fold": 800,
            "n_z_samples_train": 4,
            "n_z_samples_eval": 256,
            "budgets": list(BUDGETS),
        },
        tags=["beavertails", "three-seed", "np", "anp", "tfnp-d", "faithful"],
    ) as run:
        for step, budget in enumerate(BUDGETS):
            row: dict[str, float | int] = {"budget": budget}
            for model, result in metrics["aggregate"].items():
                values = result["by_budget"][str(budget)]
                row[f"brier/{model}"] = values["brier"]
                row[f"brier_se/{model}"] = values["brier_se"]
            run.log(row, step=step)
        for model, result in metrics["aggregate"].items():
            run.summary[f"brier_alc/{model}"] = result["brier_alc"]
            run.summary[f"brier_alc_seed_std/{model}"] = result["brier_alc_seed_std"]
        run.summary["evidence_status"] = metrics["status"]
        artifact = wandb.Artifact(
            "beavertails-faithful-np-family-3seed-v1",
            type="evaluation",
            metadata={"seeds": 3, "folds_per_seed": metrics["folds_per_seed"]},
        )
        for path in sorted(RUN_DIR.rglob("metrics.json")):
            artifact.add_file(str(path), name=str(path.relative_to(RUN_DIR)))
        for filename in ("comparison.png", "comparison.pdf"):
            artifact.add_file(str(RUN_DIR / filename))
        run.log_artifact(artifact)
        run.log({"comparison_figure": wandb.Image(str(RUN_DIR / "comparison.png"))})
        print(run.url)


if __name__ == "__main__":
    main()
