"""Paired full-protocol evaluation of empirical-Bayes context-rate predictors.

The prior mean is estimated from training tasks. Prior concentration and the
amount of shrinkage of that mean toward 0.5 are selected on validation only.
The frozen choice is then evaluated once on the benchmark-held-out test split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-empirical-bayes"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "empirical-bayes-v1"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
ORDER_SEEDS = tuple(range(10))

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scipy>=1.14,<2"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    volumes={str(REMOTE_ROOT): volume},
    timeout=30 * 60,
)
def run() -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from scipy.special import betaln, expit, logit

    train = pd.read_parquet(CORPUS_ROOT / "train_tasks.parquet")
    evaluation = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")

    # Empirical-Bayes marginal-likelihood fit on training tasks only.
    counts = train.groupby("task_id")["label"].agg(["sum", "count"])
    successes = counts["sum"].to_numpy(dtype=float)
    totals = counts["count"].to_numpy(dtype=float)

    def objective(log_ab):
        alpha, beta = np.exp(log_ab)
        value = np.sum(
            betaln(successes + alpha, totals - successes + beta) - betaln(alpha, beta)
        )
        return -float(value)

    fitted = minimize(
        objective,
        np.log(np.asarray([2.0, 2.0])),
        method="L-BFGS-B",
        bounds=[(-6.0, 8.0), (-6.0, 8.0)],
    )
    if not fitted.success:
        raise RuntimeError(f"beta-binomial MLE failed: {fitted.message}")
    mle_alpha, mle_beta = np.exp(fitted.x)
    mle_mean = float(mle_alpha / (mle_alpha + mle_beta))
    mle_concentration = float(mle_alpha + mle_beta)

    # Validation chooses only two robust hyperparameters: how far the learned
    # mean may move from 0.5, and the prior concentration. The test split never
    # participates in this choice.
    mean_blends = (0.0, 0.25, 0.5, 0.75, 1.0)
    concentrations = sorted(
        set((0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, round(mle_concentration, 6)))
    )
    candidates = []
    for blend in mean_blends:
        mean = 0.5 + blend * (mle_mean - 0.5)
        for concentration in concentrations:
            candidates.append(
                {
                    "name": f"eb_blend{blend:g}_k{concentration:g}",
                    "mean_blend": float(blend),
                    "mean": float(mean),
                    "concentration": float(concentration),
                    "alpha": float(mean * concentration),
                    "beta": float((1.0 - mean) * concentration),
                }
            )

    def episode_summaries(split: str):
        summaries = []
        split_rows = evaluation[evaluation["split"] == split]
        for task_id, task_rows in split_rows.groupby("task_id", sort=True):
            target = task_rows[task_rows["pool"] == "evaluation"]
            target_rate = float(target["label"].mean())
            lookup = task_rows.set_index("item_index")
            for order_seed in ORDER_SEEDS:
                selected_order = orders[
                    (orders["task_id"] == task_id) & (orders["order_seed"] == order_seed)
                ].sort_values("acquisition_rank")
                context = lookup.loc[selected_order["item_index"].to_numpy()]
                context_labels = context["label"].to_numpy(dtype=float)
                summaries.append(
                    {
                        "task_id": str(task_id),
                        "benchmark_id": str(task_rows["benchmark_id"].iloc[0]),
                        "order_seed": int(order_seed),
                        "target_rate": target_rate,
                        "prefix_successes": np.concatenate(
                            [np.asarray([0.0]), np.cumsum(context_labels)]
                        ),
                    }
                )
        return summaries

    episodes = {
        "validation": episode_summaries("validation"),
        "test": episode_summaries("test"),
    }

    def score_prior(prior: dict[str, Any], split: str):
        rows = []
        alpha = float(prior["alpha"])
        concentration = float(prior["concentration"])
        for episode in episodes[split]:
            budget_scores = []
            for budget, weight in zip(BUDGETS, WEIGHTS, strict=True):
                successes_at_budget = float(episode["prefix_successes"][budget])
                probability = (alpha + successes_at_budget) / (concentration + budget)
                rate = float(episode["target_rate"])
                # For a constant forecast on binary outcomes, these are exact.
                brier = probability * probability - 2.0 * probability * rate + rate
                ece = abs(probability - rate)
                budget_scores.append((budget, weight, brier, ece, probability))
            rows.append(
                {
                    "task_id": episode["task_id"],
                    "benchmark_id": episode["benchmark_id"],
                    "order_seed": episode["order_seed"],
                    "brier_alc": float(sum(w * b for _, w, b, _, _ in budget_scores)),
                    "ece_alc": float(sum(w * e for _, w, _, e, _ in budget_scores)),
                    "by_budget": {
                        str(b): {"brier": float(bs), "ece": float(e), "probability": float(p)}
                        for b, _, bs, e, p in budget_scores
                    },
                }
            )
        task_means = {}
        for row in rows:
            task_means.setdefault(row["task_id"], []).append(row["brier_alc"])
        macro_alc = float(np.mean([np.mean(values) for values in task_means.values()]))
        per_budget = {}
        for budget in BUDGETS:
            by_task = {}
            for row in rows:
                by_task.setdefault(row["task_id"], []).append(row["by_budget"][str(budget)])
            per_budget[str(budget)] = {
                "brier": float(
                    np.mean([np.mean([x["brier"] for x in values]) for values in by_task.values()])
                ),
                "ece": float(
                    np.mean([np.mean([x["ece"] for x in values]) for values in by_task.values()])
                ),
            }
        return {"brier_alc": macro_alc, "by_budget": per_budget, "rows": rows}

    baseline = {
        "name": "beta_2_2",
        "mean_blend": 0.0,
        "mean": 0.5,
        "concentration": 4.0,
        "alpha": 2.0,
        "beta": 2.0,
    }
    validation_results = []
    for candidate in candidates:
        scored = score_prior(candidate, "validation")
        validation_results.append({**candidate, "brier_alc": scored["brier_alc"]})
    selected = min(validation_results, key=lambda row: (row["brier_alc"], row["name"]))
    selected_prior = {key: selected[key] for key in candidates[0] if key in selected}

    output = {
        "status": "complete",
        "selection_rule": "minimum macro validation ALC over fixed prior grid",
        "training_fit": {
            "tasks": int(len(counts)),
            "alpha": float(mle_alpha),
            "beta": float(mle_beta),
            "mean": mle_mean,
            "concentration": mle_concentration,
        },
        "baseline_prior": baseline,
        "selected_prior": selected_prior,
        "validation": {
            "baseline": score_prior(baseline, "validation"),
            "selected": score_prior(selected_prior, "validation"),
        },
        "test": {
            "baseline": score_prior(baseline, "test"),
            "selected": score_prior(selected_prior, "test"),
        },
        "validation_grid": sorted(validation_results, key=lambda row: row["brier_alc"]),
    }
    output["test_improvement"] = float(
        output["test"]["baseline"]["brier_alc"]
        - output["test"]["selected"]["brier_alc"]
    )
    _write_json(RESULT_ROOT / "results.json", output)
    volume.commit()
    print(
        json.dumps(
            {
                "training_fit": output["training_fit"],
                "selected_prior": selected_prior,
                "validation_baseline": output["validation"]["baseline"]["brier_alc"],
                "validation_selected": output["validation"]["selected"]["brier_alc"],
                "test_baseline": output["test"]["baseline"]["brier_alc"],
                "test_selected": output["test"]["selected"]["brier_alc"],
                "test_improvement": output["test_improvement"],
            },
            indent=2,
        ),
        flush=True,
    )
    return output


@app.local_entrypoint()
def main():
    result = run.remote()
    print(
        "RESULT "
        + json.dumps(
            {
                "selected_prior": result["selected_prior"],
                "validation_baseline": result["validation"]["baseline"]["brier_alc"],
                "validation_selected": result["validation"]["selected"]["brier_alc"],
                "test_baseline": result["test"]["baseline"]["brier_alc"],
                "test_selected": result["test"]["selected"]["brier_alc"],
                "test_improvement": result["test_improvement"],
            },
            sort_keys=True,
        )
    )
