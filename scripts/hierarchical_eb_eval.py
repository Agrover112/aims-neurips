#!/usr/bin/env python3
"""Evaluate a three-level empirical-Bayes predictor on the benchmark-held-out split.

The hierarchy is:

    global Beta prior -> subject effect -> benchmark effect -> group Beta prior

Subject and benchmark effects are fit on training tasks only.  An effect for an
unseen subject or benchmark is zero.  Given context labels from the target
subject--benchmark group, the group probability is the posterior mean under a
Beta prior whose mean is the hierarchy prior and whose concentration is chosen
on the validation split.

The script is deliberately local: it reads the small protocol parquet files and
does not write to the Modal volume.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
KAPPA_BUDGET = {
    0: 4.0,
    1: 3.890451449942805,
    3: 3.630780547701014,
    7: 3.715352290971724,
    15: 3.8018939632056123,
    31: 3.981071705534973,
}
LGAMMA = np.vectorize(math.lgamma, otypes=[float])


def logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def expit(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_beta_binomial(successes: np.ndarray, totals: np.ndarray) -> tuple[float, float]:
    """Fit a Beta-Binomial prior by grid search plus coordinate refinement."""

    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)

    def nll(alpha: float, beta: float) -> float:
        return -float(
            np.sum(
                LGAMMA(successes + alpha)
                + LGAMMA(totals - successes + beta)
                - LGAMMA(totals + alpha + beta)
                - LGAMMA(alpha)
                - LGAMMA(beta)
                + LGAMMA(alpha + beta)
            )
        )

    best = (math.inf, 2.0, 2.0)
    for log_alpha in np.linspace(-3.0, 3.0, 61):
        alpha = math.exp(float(log_alpha))
        for log_beta in np.linspace(-3.0, 3.0, 61):
            beta = math.exp(float(log_beta))
            value = nll(alpha, beta)
            if value < best[0]:
                best = (value, alpha, beta)

    alpha, beta = best[1], best[2]
    for step in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        improved = True
        while improved:
            improved = False
            for da, db in (
                (step, 0.0),
                (-step, 0.0),
                (0.0, step),
                (0.0, -step),
                (step, step),
                (-step, -step),
                (step, -step),
                (-step, step),
            ):
                candidate = (max(1e-6, alpha + da), max(1e-6, beta + db))
                value = nll(*candidate)
                if value < best[0] - 1e-12:
                    best = (value, candidate[0], candidate[1])
                    alpha, beta = candidate
                    improved = True
    return float(alpha), float(beta)


def fit_variance_components(task: pd.DataFrame) -> dict[str, Any]:
    """Method-of-moments variance components on the logit scale.

    A subject has almost exactly one training task (6,701 subjects for 6,702
    tasks), so subject variance is not separately identified from task noise.
    The estimate below is therefore a conservative upper bound; it is reported
    and used only when positive.
    """

    def between_var(frame: pd.DataFrame) -> tuple[float, float]:
        rate = (frame["s"] + 0.5) / (frame["n"] + 1.0)
        logit_rate = np.log(rate / (1.0 - rate))
        p = np.clip(frame["s"] / frame["n"], 1e-4, 1.0 - 1e-4)
        sampling = 1.0 / (frame["n"] * p * (1.0 - p))
        observed = float(np.var(logit_rate, ddof=1)) if len(frame) > 1 else 0.0
        return max(0.0, observed - float(np.mean(sampling))), float(np.mean(sampling))

    subjects = (
        task.groupby("subject_uid", as_index=False)
        .agg(s=("s", "sum"), n=("n", "sum"))
    )
    benchmarks = (
        task.groupby("benchmark_id", as_index=False)
        .agg(s=("s", "sum"), n=("n", "sum"))
    )
    subject_var, subject_sampling = between_var(subjects)
    benchmark_var, benchmark_sampling = between_var(benchmarks)
    return {
        "subject_variance": subject_var,
        "subject_sd": math.sqrt(subject_var),
        "subject_mean_sampling_variance": subject_sampling,
        "benchmark_variance": benchmark_var,
        "benchmark_sd": math.sqrt(benchmark_var),
        "benchmark_mean_sampling_variance": benchmark_sampling,
        "subject_tasks_per_subject": subjects["n"].describe().to_dict(),
        "subject_count": int(len(subjects)),
        "benchmark_count": int(len(benchmarks)),
    }


def build_episodes(
    eval_items: pd.DataFrame,
    orders: pd.DataFrame,
    split: str,
    subject_map: dict[int, str],
) -> dict[str, Any]:
    rows = eval_items[eval_items["split"] == split]
    target = rows[rows["pool"] == "evaluation"].groupby("task_id")["label"].mean()
    acquisition = rows[rows["pool"] == "acquisition"][["task_id", "item_index", "label"]]
    task_info = (
        rows[["task_id", "subject_index", "benchmark_id"]]
        .drop_duplicates("task_id")
        .set_index("task_id")
    )
    task_ids = sorted(target.index)
    order_seeds = sorted(orders["order_seed"].unique())
    order = orders[orders["task_id"].isin(task_ids)].copy()
    order = order.merge(acquisition, on=["task_id", "item_index"], how="left", validate="many_to_one")
    if order["label"].isna().any():
        missing = int(order["label"].isna().sum())
        raise RuntimeError(f"{split}: {missing} acquisition labels are missing")
    order = order.sort_values(["task_id", "order_seed", "acquisition_rank"]).copy()
    order["position"] = order.groupby(["task_id", "order_seed"]).cumcount()
    wide = order.pivot(index=["task_id", "order_seed"], columns="position", values="label")
    wide = wide.reindex(
        pd.MultiIndex.from_product([task_ids, order_seeds], names=["task_id", "order_seed"])
    )
    labels = wide.to_numpy(dtype=int).reshape(len(task_ids), len(order_seeds), 31)
    if labels.shape != (len(task_ids), len(order_seeds), 31):
        raise RuntimeError(f"{split}: unexpected episode shape {labels.shape}")
    prefix = np.concatenate(
        [np.zeros((len(task_ids), len(order_seeds), 1), dtype=int), np.cumsum(labels, axis=2)],
        axis=2,
    )
    return {
        "task_ids": task_ids,
        "subject_uid": [subject_map[int(task_info.loc[t, "subject_index"])] for t in task_ids],
        "benchmark_id": [str(task_info.loc[t, "benchmark_id"]) for t in task_ids],
        "target_rate": target.loc[task_ids].to_numpy(dtype=float),
        "successes": prefix,
        "order_seeds": order_seeds,
    }


def evaluate(
    prior_mean: np.ndarray,
    kappas: np.ndarray,
    episodes: dict[str, Any],
) -> dict[str, Any]:
    successes = episodes["successes"]
    target = episodes["target_rate"]
    n_tasks = len(target)
    n_orders = successes.shape[1]
    brier = np.empty((n_tasks, n_orders, len(BUDGETS)), dtype=float)
    ece = np.empty_like(brier)
    for j, budget in enumerate(BUDGETS):
        kappa = float(kappas[j])
        probability = (prior_mean[:, None] * kappa + successes[:, :, budget]) / (kappa + budget)
        brier[:, :, j] = (
            probability * probability - 2.0 * probability * target[:, None] + target[:, None]
        )
        ece[:, :, j] = np.abs(probability - target[:, None])
    brier_task = brier.mean(axis=1)
    ece_task = ece.mean(axis=1)
    task_alc = brier_task @ WEIGHTS
    task_ece_alc = ece_task @ WEIGHTS
    return {
        "brier_alc": float(task_alc.mean()),
        "ece_alc": float(task_ece_alc.mean()),
        "brier_by_budget": brier_task.mean(axis=0).tolist(),
        "ece_by_budget": ece_task.mean(axis=0).tolist(),
        "task_alc": task_alc,
        "task_ece_alc": task_ece_alc,
        "task_brier": brier_task,
        "task_ece": ece_task,
    }


def select_kappa(
    prior_mean: np.ndarray,
    episodes: dict[str, Any],
    grid: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for kappa in grid:
        result = evaluate(prior_mean, np.full(len(BUDGETS), kappa), episodes)
        rows.append({"kappa": kappa, "brier_alc": result["brier_alc"]})
    best = min(rows, key=lambda row: (row["brier_alc"], row["kappa"]))
    return float(best["kappa"]), rows


def select_kappa_by_budget(
    prior_mean: np.ndarray,
    episodes: dict[str, Any],
    grid: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> tuple[np.ndarray, dict[str, list[dict[str, float]]]]:
    selected = []
    details: dict[str, list[dict[str, float]]] = {}
    for j, budget in enumerate(BUDGETS):
        rows = []
        for kappa in grid:
            result = evaluate(prior_mean, np.full(len(BUDGETS), kappa), episodes)
            rows.append({"kappa": kappa, "brier": float(result["task_brier"][:, j].mean())})
        best = min(rows, key=lambda row: (row["brier"], row["kappa"]))
        selected.append(float(best["kappa"]))
        details[str(budget)] = rows
    return np.asarray(selected), details


def hierarchy_prior(
    episodes: dict[str, Any],
    train_subjects: set[str],
    train_benchmarks: set[str],
    subject_effect: dict[str, float],
    benchmark_effect: dict[str, float],
    global_mean: float,
) -> tuple[np.ndarray, dict[str, int]]:
    prior = []
    seen_subjects = 0
    seen_benchmarks = 0
    for subject, benchmark in zip(episodes["subject_uid"], episodes["benchmark_id"], strict=True):
        u = subject_effect.get(subject, 0.0) if subject in train_subjects else 0.0
        v = benchmark_effect.get(benchmark, 0.0) if benchmark in train_benchmarks else 0.0
        if subject in train_subjects:
            seen_subjects += 1
        if benchmark in train_benchmarks:
            seen_benchmarks += 1
        prior.append(float(expit(logit(global_mean) + u + v)))
    return np.asarray(prior), {
        "seen_subjects": seen_subjects,
        "seen_benchmarks": seen_benchmarks,
        "tasks": len(prior),
    }


def paired_summary(
    difference: np.ndarray,
    task_benchmarks: list[str],
    seed: int = 20261020,
    draws: int = 20000,
) -> dict[str, Any]:
    difference = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(seed)
    task_idx = rng.integers(0, len(difference), size=(draws, len(difference)))
    task_means = difference[task_idx].mean(axis=1)

    clusters = sorted(set(task_benchmarks))
    cluster_indices = {cluster: np.where(np.asarray(task_benchmarks) == cluster)[0] for cluster in clusters}
    cluster_means = []
    for _ in range(draws):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([cluster_indices[c] for c in chosen])
        cluster_means.append(float(difference[idx].mean()))
    cluster_means = np.asarray(cluster_means)
    return {
        "mean": float(difference.mean()),
        "se_task": float(difference.std(ddof=1) / math.sqrt(len(difference))),
        "ci95_task": [float(np.quantile(task_means, 0.025)), float(np.quantile(task_means, 0.975))],
        "ci95_benchmark_cluster": [
            float(np.quantile(cluster_means, 0.025)),
            float(np.quantile(cluster_means, 0.975)),
        ],
        "wins": int((difference < 0).sum()),
        "losses": int((difference > 0).sum()),
        "tasks": int(len(difference)),
        "benchmarks": len(clusters),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/tmp/opencode/allbench-features-v1"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hierarchical-eb-v1"),
    )
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.data_root / "train_tasks.parquet")
    eval_items = pd.read_parquet(args.data_root / "eval_items.parquet")
    orders = pd.read_parquet(args.data_root / "eval_orders.parquet")
    subjects = pd.read_parquet(args.data_root / "subjects.parquet")
    subject_map = dict(enumerate(subjects["subject_uid"]))

    train["subject_uid"] = train["subject_index"].map(subject_map)
    if train["subject_uid"].isna().any():
        raise RuntimeError("training subject index could not be mapped")
    train["benchmark_id"] = train["task_id"].str.split("::").str[1]
    task = (
        train.groupby(["task_id", "subject_uid", "benchmark_id"], as_index=False)
        .agg(s=("label", "sum"), n=("label", "size"))
    )

    alpha0, beta0 = fit_beta_binomial(task["s"].to_numpy(), task["n"].to_numpy())
    global_mean = float(alpha0 / (alpha0 + beta0))
    global_concentration = float(alpha0 + beta0)
    variance = fit_variance_components(task)

    subjects_stats = (
        task.groupby("subject_uid", as_index=False).agg(s=("s", "sum"), n=("n", "sum"))
    )
    benchmarks_stats = (
        task.groupby("benchmark_id", as_index=False).agg(s=("s", "sum"), n=("n", "sum"))
    )
    subjects_stats["post_mean"] = (alpha0 + subjects_stats["s"]) / (
        alpha0 + beta0 + subjects_stats["n"]
    )
    benchmarks_stats["post_mean"] = (alpha0 + benchmarks_stats["s"]) / (
        alpha0 + beta0 + benchmarks_stats["n"]
    )
    subject_effect = {
        str(row.subject_uid): float(logit(row.post_mean) - logit(global_mean))
        for row in subjects_stats.itertuples()
    }
    benchmark_effect = {
        str(row.benchmark_id): float(logit(row.post_mean) - logit(global_mean))
        for row in benchmarks_stats.itertuples()
    }
    subject_denominator = (
        variance["subject_variance"] + variance["subject_mean_sampling_variance"]
    )
    benchmark_denominator = (
        variance["benchmark_variance"] + variance["benchmark_mean_sampling_variance"]
    )
    subject_scale = (
        variance["subject_variance"] / subject_denominator
        if subject_denominator > 0.0
        else 0.0
    )
    benchmark_scale = (
        variance["benchmark_variance"] / benchmark_denominator
        if benchmark_denominator > 0.0
        else 0.0
    )
    subject_effect_scaled = {key: subject_scale * value for key, value in subject_effect.items()}
    benchmark_effect_scaled = {
        key: benchmark_scale * value for key, value in benchmark_effect.items()
    }

    neutral_alpha = neutral_beta = 2.0
    neutral_mean = 0.5
    neutral_subjects = (
        task.groupby("subject_uid", as_index=False).agg(s=("s", "sum"), n=("n", "sum"))
    )
    neutral_benchmarks = (
        task.groupby("benchmark_id", as_index=False).agg(s=("s", "sum"), n=("n", "sum"))
    )
    neutral_subjects["post_mean"] = (neutral_alpha + neutral_subjects["s"]) / (
        neutral_alpha + neutral_beta + neutral_subjects["n"]
    )
    neutral_benchmarks["post_mean"] = (neutral_alpha + neutral_benchmarks["s"]) / (
        neutral_alpha + neutral_beta + neutral_benchmarks["n"]
    )
    neutral_subject_effect = {
        str(row.subject_uid): float(logit(row.post_mean) - logit(neutral_mean))
        for row in neutral_subjects.itertuples()
    }
    neutral_benchmark_effect = {
        str(row.benchmark_id): float(logit(row.post_mean) - logit(neutral_mean))
        for row in neutral_benchmarks.itertuples()
    }
    neutral_subject_effect_scaled = {
        key: subject_scale * value for key, value in neutral_subject_effect.items()
    }
    neutral_benchmark_effect_scaled = {
        key: benchmark_scale * value for key, value in neutral_benchmark_effect.items()
    }

    validation = build_episodes(eval_items, orders, "validation", subject_map)
    test = build_episodes(eval_items, orders, "test", subject_map)
    train_subjects = set(task["subject_uid"].astype(str))
    train_benchmarks = set(task["benchmark_id"].astype(str))

    hier_mle_prior, hier_mle_seen = hierarchy_prior(
        test,
        train_subjects,
        train_benchmarks,
        subject_effect_scaled,
        benchmark_effect_scaled,
        global_mean,
    )
    hier_mle_val_prior, hier_mle_val_seen = hierarchy_prior(
        validation,
        train_subjects,
        train_benchmarks,
        subject_effect_scaled,
        benchmark_effect_scaled,
        global_mean,
    )
    hier_neutral_prior, hier_neutral_seen = hierarchy_prior(
        test,
        train_subjects,
        train_benchmarks,
        neutral_subject_effect_scaled,
        neutral_benchmark_effect_scaled,
        neutral_mean,
    )
    hier_neutral_val_prior, hier_neutral_val_seen = hierarchy_prior(
        validation,
        train_subjects,
        train_benchmarks,
        neutral_subject_effect_scaled,
        neutral_benchmark_effect_scaled,
        neutral_mean,
    )

    # The subject variance is at the boundary, so a fitted hierarchy uses no
    # subject effect.  Keep a naive variant only as a sensitivity check.
    hier_naive_subject_prior = []
    for subject, benchmark in zip(test["subject_uid"], test["benchmark_id"], strict=True):
        u = subject_effect.get(subject, 0.0) if subject in train_subjects else 0.0
        v = benchmark_effect.get(benchmark, 0.0) if benchmark in train_benchmarks else 0.0
        hier_naive_subject_prior.append(float(expit(logit(global_mean) + u + v)))
    hier_naive_subject_prior = np.asarray(hier_naive_subject_prior)

    kappa_hier_mle, kappa_hier_mle_grid = select_kappa(hier_mle_val_prior, validation)
    kappa_hier_neutral, kappa_hier_neutral_grid = select_kappa(
        hier_neutral_val_prior, validation
    )
    kappa_hier_mle_budget, kappa_hier_mle_budget_grid = select_kappa_by_budget(
        hier_mle_val_prior, validation
    )
    kappa_hier_neutral_budget, kappa_hier_neutral_budget_grid = select_kappa_by_budget(
        hier_neutral_val_prior, validation
    )

    models: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "flat_beta22": (np.full(len(test["task_ids"]), 0.5), np.full(len(BUDGETS), 4.0)),
        "flat_budget": (
            np.full(len(test["task_ids"]), 0.5),
            np.asarray([KAPPA_BUDGET[b] for b in BUDGETS]),
        ),
        "global_mle": (
            np.full(len(test["task_ids"]), global_mean),
            np.full(len(BUDGETS), global_concentration),
        ),
        "hier_mle": (hier_mle_prior, np.full(len(BUDGETS), kappa_hier_mle)),
        "hier_mle_budget": (hier_mle_prior, kappa_hier_mle_budget),
        "hier_neutral": (
            hier_neutral_prior,
            np.full(len(BUDGETS), kappa_hier_neutral),
        ),
        "hier_neutral_budget": (hier_neutral_prior, kappa_hier_neutral_budget),
        "hier_naive_subject": (
            hier_naive_subject_prior,
            np.full(len(BUDGETS), kappa_hier_mle),
        ),
    }

    validation_models: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "flat_beta22": (
            np.full(len(validation["task_ids"]), 0.5),
            np.full(len(BUDGETS), 4.0),
        ),
        "flat_budget": (
            np.full(len(validation["task_ids"]), 0.5),
            np.asarray([KAPPA_BUDGET[b] for b in BUDGETS]),
        ),
        "global_mle": (
            np.full(len(validation["task_ids"]), global_mean),
            np.full(len(BUDGETS), global_concentration),
        ),
        "hier_mle": (hier_mle_val_prior, np.full(len(BUDGETS), kappa_hier_mle)),
        "hier_mle_budget": (hier_mle_val_prior, kappa_hier_mle_budget),
        "hier_neutral": (
            hier_neutral_val_prior,
            np.full(len(BUDGETS), kappa_hier_neutral),
        ),
        "hier_neutral_budget": (hier_neutral_val_prior, kappa_hier_neutral_budget),
        "hier_naive_subject": (
            hier_mle_val_prior,
            np.full(len(BUDGETS), kappa_hier_mle),
        ),
    }

    results: dict[str, Any] = {}
    task_rows = []
    for split_name, episodes, model_map in (
        ("validation", validation, validation_models),
        ("test", test, models),
    ):
        split_results = {}
        for name, (prior_mean, kappas) in model_map.items():
            scored = evaluate(prior_mean, kappas, episodes)
            split_results[name] = {
                "brier_alc": scored["brier_alc"],
                "ece_alc": scored["ece_alc"],
                "brier_by_budget": scored["brier_by_budget"],
                "ece_by_budget": scored["ece_by_budget"],
                "kappas": np.asarray(kappas).tolist(),
                "prior_mean_mean": float(np.mean(prior_mean)),
                "prior_mean_min": float(np.min(prior_mean)),
                "prior_mean_max": float(np.max(prior_mean)),
                "task_alc": scored["task_alc"].tolist(),
                "task_ece_alc": scored["task_ece_alc"].tolist(),
            }
            for i, task_id in enumerate(episodes["task_ids"]):
                task_rows.append(
                    {
                        "split": split_name,
                        "model": name,
                        "task_id": task_id,
                        "benchmark_id": episodes["benchmark_id"][i],
                        "subject_uid": episodes["subject_uid"][i],
                        "subject_seen": episodes["subject_uid"][i] in train_subjects,
                        "benchmark_seen": episodes["benchmark_id"][i] in train_benchmarks,
                        "brier_alc": float(scored["task_alc"][i]),
                        "ece_alc": float(scored["task_ece_alc"][i]),
                        **{
                            f"brier_b{budget}": float(scored["task_brier"][i, j])
                            for j, budget in enumerate(BUDGETS)
                        },
                    }
                )
        results[split_name] = split_results

    test_results = results["test"]
    comparisons = {}
    for name in test_results:
        if name == "flat_beta22":
            continue
        comparisons[f"{name}_minus_flat_beta22"] = paired_summary(
            np.asarray(test_results[name]["task_alc"])
            - np.asarray(test_results["flat_beta22"]["task_alc"]),
            test["benchmark_id"],
            draws=args.draws,
        )
    for name in ("hier_mle", "hier_mle_budget", "hier_neutral", "hier_neutral_budget"):
        comparisons[f"{name}_minus_flat_budget"] = paired_summary(
            np.asarray(test_results[name]["task_alc"])
            - np.asarray(test_results["flat_budget"]["task_alc"]),
            test["benchmark_id"],
            draws=args.draws,
        )

    task_frame = pd.DataFrame(task_rows)
    task_frame.to_csv(args.out / "task_metrics.csv", index=False)

    output = {
        "status": "complete",
        "data": {
            "train_tasks": int(task["task_id"].nunique()),
            "train_subjects": int(task["subject_uid"].nunique()),
            "train_benchmarks": int(task["benchmark_id"].nunique()),
            "validation_tasks": int(len(validation["task_ids"])),
            "test_tasks": int(len(test["task_ids"])),
            "budgets": list(BUDGETS),
            "weights": WEIGHTS.tolist(),
        },
        "global_beta_binomial": {
            "alpha": alpha0,
            "beta": beta0,
            "mean": global_mean,
            "concentration": global_concentration,
        },
        "variance_components": variance,
        "hierarchy": {
            "mle_effects": {
                "validation": hier_mle_val_seen,
                "test": hier_mle_seen,
            },
            "neutral_effects": {
                "validation": hier_neutral_val_seen,
                "test": hier_neutral_seen,
            },
            "effect_scales": {
                "subject": subject_scale,
                "benchmark": benchmark_scale,
            },
            "kappa_grid": {
                "hier_mle": kappa_hier_mle_grid,
                "hier_neutral": kappa_hier_neutral_grid,
            },
            "kappa_selected": {
                "hier_mle": kappa_hier_mle,
                "hier_neutral": kappa_hier_neutral,
                "hier_mle_budget": kappa_hier_mle_budget.tolist(),
                "hier_neutral_budget": kappa_hier_neutral_budget.tolist(),
            },
        },
        "results": results,
        "paired_test_comparisons": comparisons,
    }
    (args.out / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for axis, split_name in zip(axes, ("validation", "test"), strict=True):
        for name in ("flat_beta22", "flat_budget", "global_mle", "hier_mle", "hier_naive_subject"):
            values = results[split_name][name]["brier_by_budget"]
            axis.plot(BUDGETS, values, "o-", label=name, lw=2.0)
        axis.set(
            title=split_name.title(),
            xlabel="labels acquired",
            ylabel="Brier (macro over tasks)",
            xticks=BUDGETS,
        )
    axes[0].legend(fontsize=8)
    fig.suptitle("Hierarchical empirical-Bayes evaluation: 22 test tasks, 10 orders")
    fig.savefig(args.out / "brier_curves.png", dpi=190)
    plt.close(fig)

    print(
        json.dumps(
            {
                "global": output["global_beta_binomial"],
                "variance_components": variance,
                "hierarchy_seen": output["hierarchy"]["mle_effects"],
                "kappa_selected": output["hierarchy"]["kappa_selected"],
                "validation_alc": {
                    name: value["brier_alc"] for name, value in results["validation"].items()
                },
                "test_alc": {
                    name: value["brier_alc"] for name, value in results["test"].items()
                },
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
