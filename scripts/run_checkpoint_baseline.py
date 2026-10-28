#!/usr/bin/env python3
"""Replay the checked-in historical demo weights; never train or select a model.

This is an installation/inference baseline on five already-exported validation
tasks, NOT the newer filtered CNP/ANP or an official competition score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo" / "forecaster"))


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def revision() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run(asset_root: Path, threads: int = 4, offline: bool = False) -> dict:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    import transformers
    from engine import ForecastEngine, Reveal

    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, brier_alc, brier_score

    torch.set_num_threads(threads)
    started = time.monotonic()
    engine = ForecastEngine(asset_root)
    names = ("beta", "cnp", "airt")
    task_results = []
    for task_id, task in engine.tasks.items():
        if len(task["acquisition"]) < max(DEFAULT_BUDGETS):
            raise ValueError(f"Insufficient acquisition items: {task_id}")
        acquired_ids = {item["item_id"] for item in task["acquisition"]}
        evaluation_ids = {item["item_id"] for item in task["evaluation"]}
        if acquired_ids & evaluation_ids:
            raise ValueError(f"Acquisition/evaluation item overlap: {task_id}")
        revealed = []
        curves = {name: [] for name in names}
        probabilities = {name: {} for name in names}
        for budget in DEFAULT_BUDGETS:
            while len(revealed) < budget:
                index = len(revealed)
                # Only the newly acquired prefix labels enter the predictor.
                outcome = int(task["acquisition"][index]["label"])
                if outcome not in (0, 1):
                    raise ValueError("Expected binary acquired outcomes")
                revealed.append(Reveal(index=index, outcome=outcome, cnp=0.0, airt=0.0))
            beta, cnp, airt = engine.predict_targets(task, revealed)
            predictions = {"beta": np.full(len(cnp), beta), "cnp": cnp, "airt": airt}
            # Evaluation outcomes are accessed for scoring, after predictions.
            outcomes = [item["label"] for item in task["evaluation"]]
            for name, values in predictions.items():
                curves[name].append(brier_score(outcomes, values))
                probabilities[name][str(budget)] = values.tolist()
        row = {
            "task_id": task_id,
            "evaluation_items": len(task["evaluation"]),
            "brier_by_budget": {
                name: dict(zip(map(str, DEFAULT_BUDGETS), curves[name])) for name in names
            },
            "brier_alc": {name: brier_alc(curves[name]) for name in names},
            "predictions": probabilities,
        }
        task_results.append(row)
        print(
            json.dumps(
                {"stage": "task_completed", "task_id": task_id, "brier_alc": row["brier_alc"]}
            ),
            flush=True,
        )
    result = {
        "status": "historical_checkpoint_replay_completed",
        "scope": "Five bundled historical validation demo tasks; setup/inference check only",
        "source_revision": revision(),
        "acquisition": "Frozen export order; one shared nested prefix for all predictors",
        "training_performed": False,
        "official_test_outcomes_read": False,
        "new_filtered_checkpoints_evaluated": False,
        "tasks": len(task_results),
        "evaluation_items_per_task": [row["evaluation_items"] for row in task_results],
        "budgets": DEFAULT_BUDGETS,
        "alc_weights": DEFAULT_WEIGHTS,
        "macro_task_brier_alc": {
            name: float(np.mean([row["brier_alc"][name] for row in task_results])) for name in names
        },
        "macro_task_brier_by_budget": {
            name: {
                str(budget): float(
                    np.mean([row["brier_by_budget"][name][str(budget)] for row in task_results])
                )
                for budget in DEFAULT_BUDGETS
            }
            for name in names
        },
        "assets_sha256": {
            name: sha256(asset_root / name)
            for name in (
                "cnp_airt.pt",
                "online_airt_2pl.pt",
                "metadata_cnp.json",
                "metadata_airt.json",
                "tasks.json",
            )
        },
        "encoder": {
            "name": engine.cnp_meta["embedding_model"],
            "revision": getattr(engine.encoder.config, "_commit_hash", None),
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "task_results": task_results,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "demo/forecaster/assets")
    parser.add_argument("--output", type=Path, default=ROOT / "results/checkpoint-baseline.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--offline", action="store_true", help="Use an already cached MiniLM encoder"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a fresh --output path to preserve the previous run")
    result = run(args.assets.resolve(), args.threads, args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps({key: value for key, value in result.items() if key != "task_results"}, indent=2)
    )
    print(f"Detailed replay: {args.output.resolve()}")


if __name__ == "__main__":
    main()
