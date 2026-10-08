"""Modal entry point for a reproducible experiment run.

Usage: modal run modal_app.py --predictions path/to/predictions.json
"""

from pathlib import Path

import modal

app = modal.App("aims-experiments")
image = modal.Image.debian_slim(python_version="3.11").pip_install("wandb>=0.17")


@app.function(image=image, timeout=60 * 60)
def run_remote(predictions_text: str, config_text: str) -> str:
    import json
    import tomllib

    config = tomllib.loads(config_text)
    record = json.loads(predictions_text)
    weights = config["evaluation"]["alc_weights"]
    truth = record["truth"]
    scores = []
    for budget in config["evaluation"]["budgets"]:
        probabilities = record["probabilities"][str(budget)]
        scores.append(sum((float(p) - float(y)) ** 2 for y, p in zip(truth, probabilities)) / len(truth))
    return json.dumps({"brier_alc": sum(score * weight for score, weight in zip(scores, weights))})


@app.local_entrypoint()
def main(predictions: str, config: str = "configs/default.toml") -> None:
    output = run_remote.remote(Path(predictions).read_text(), Path(config).read_text())
    print(output)
