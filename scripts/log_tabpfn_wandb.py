"""Log the off-the-shelf TabPFN-v2 baseline to Weights & Biases."""

from __future__ import annotations

import json
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "tabpfn-v2-beavertails-3seed-v1"
BUDGETS = (0, 1, 3, 7, 15, 31)


def main() -> None:
    metrics = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    result = metrics["aggregate"]["tabpfn_v2"]
    with wandb.init(
        entity="agrgnn-kth-royal-institute-of-technology",
        project="aims-predictive-evaluation",
        name="beavertails-off-the-shelf-tabpfn-v2-3seed-v1",
        job_type="external-baseline",
        config={
            "package": "tabpfn==9.2.0",
            "checkpoint": "TabPFN-v2 default classifier",
            "seeds": metrics["seeds"],
            "folds_per_seed": metrics["folds_per_seed"],
            "features": "all-MiniLM-L6-v2 normalized item embeddings (384d)",
            "budgets": list(BUDGETS),
            "fallback": "Beta(2,2) if context lacks two classes",
        },
        tags=["beavertails", "tabpfn", "off-the-shelf", "three-seed"],
    ) as run:
        for step, budget in enumerate(BUDGETS):
            row = result["by_budget"][str(budget)]
            run.log(
                {
                    "budget": budget,
                    "brier/tabpfn_v2": row["brier"],
                    "brier_se/tabpfn_v2": row["brier_se"],
                    "coverage/tabpfn_v2": row["tabpfn_evaluations"] / row["n"],
                },
                step=step,
            )
        run.summary["brier_alc/tabpfn_v2"] = result["brier_alc"]
        run.summary["brier_alc_seed_std/tabpfn_v2"] = result["brier_alc_seed_std"]
        artifact = wandb.Artifact("beavertails-tabpfn-v2-3seed-v1", type="evaluation")
        for path in sorted(RUN_DIR.glob("*.json")):
            artifact.add_file(str(path))
        for filename in ("comparison.png", "comparison.pdf"):
            artifact.add_file(str(RUN_DIR / filename))
        run.log_artifact(artifact)
        run.log({"comparison_figure": wandb.Image(str(RUN_DIR / "comparison.png"))})
        print(run.url)


if __name__ == "__main__":
    main()
