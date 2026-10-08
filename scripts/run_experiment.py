#!/usr/bin/env python3
"""Score a JSON experiment record and optionally log it to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aims_starter.io import write_json
from aims_starter.metrics import brier_alc, brier_score, calibration_ece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions", type=Path, help="JSON with truth and per-budget probabilities"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/latest/metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    record = json.loads(args.predictions.read_text(encoding="utf-8"))
    truth = record["truth"]
    budgets = config["evaluation"]["budgets"]
    weights = config["evaluation"]["alc_weights"]
    scores = []
    diagnostics = {}
    for budget in budgets:
        probability = record["probabilities"][str(budget)]
        score = brier_score(truth, probability)
        scores.append(score)
        diagnostics[str(budget)] = {
            "brier": score,
            "ece": calibration_ece(truth, probability),
        }
    metrics = {"brier_alc": brier_alc(scores, weights), "by_budget": diagnostics}
    write_json(args.output, metrics)

    mode = config.get("tracking", {}).get("wandb_mode", "disabled")
    if mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("install the 'experiment' extra to enable WandB") from exc
        with wandb.init(
            project=os.getenv("WANDB_PROJECT", "aims-predictive-evaluation"),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=config["experiment"]["name"],
            config=config,
            mode=mode,
        ) as run:
            run.log({"brier_alc": metrics["brier_alc"]})
            for budget, values in diagnostics.items():
                run.log({"budget": int(budget), **values})
            run.log_artifact(str(args.output), name="metrics", type="evaluation")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
