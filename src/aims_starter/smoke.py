"""Small end-to-end checks for model-ready benchmark episodes."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for the model smoke test") from exc
    return np


def _sigmoid(values):
    np = _numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _softplus(values):
    np = _numpy()
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _hashed_item_features(items, *, dimensions: int = 32):
    """Create deterministic numeric smoke features without fitting a vocabulary."""

    np = _numpy()
    matrix = np.zeros((len(items), dimensions + 3), dtype=np.float64)
    for row_number, text in enumerate(items["model_text"].fillna("").astype(str)):
        matrix[row_number, 0] = 1.0
        matrix[row_number, 1] = math.log1p(len(text)) / 10.0
        matrix[row_number, 2] = float(not text.strip())
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            column = 3 + value % dimensions
            sign = 1.0 if value & 1 else -1.0
            matrix[row_number, column] += sign
    norms = np.linalg.norm(matrix[:, 3:], axis=1, keepdims=True)
    matrix[:, 3:] /= np.maximum(norms, 1.0)
    return matrix


def _binary_cross_entropy(labels, probabilities) -> float:
    np = _numpy()
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities))
    )


def run_benchmark_smoke(
    observations,
    episode_items,
    items,
    subjects,
    *,
    budgets: Sequence[int] = (0, 1, 3, 7, 15, 31),
    optimization_steps: int = 200,
    seed: int = 20261017,
) -> dict[str, Any]:
    """Validate one benchmark and fit a tiny amortized 2PL integration model."""

    np = _numpy()
    benchmark_ids = set(episode_items["benchmark_id"].astype(str))
    if len(benchmark_ids) != 1:
        raise ValueError("smoke input must contain exactly one benchmark")
    benchmark_id = next(iter(benchmark_ids))
    task_ids = set(episode_items["task_id"].astype(str))
    task_observations = observations[observations["task_id"].astype(str).isin(task_ids)].copy()
    task_assignments = episode_items[episode_items["task_id"].astype(str).isin(task_ids)].copy()
    if task_observations.empty or task_assignments.empty:
        raise ValueError("smoke benchmark has no observations or episode assignments")

    duplicate_assignments = int(task_assignments.duplicated(["task_id", "item_uid"]).sum())
    if duplicate_assignments:
        raise AssertionError(f"duplicate task-item assignments: {duplicate_assignments}")
    pool_counts = task_assignments.groupby(["task_id", "pool"]).size().unstack(fill_value=0)
    if (pool_counts.min(axis=1) == 0).any():
        raise AssertionError("every task must have non-empty acquisition and evaluation pools")
    if ((pool_counts["acquisition"] - pool_counts["evaluation"]).abs() > 1).any():
        raise AssertionError("acquisition/evaluation pools are not balanced")
    if (pool_counts.sum(axis=1) < 80).any():
        raise AssertionError("an episode task has fewer than 80 distinct items")

    acquisition = task_assignments[task_assignments["pool"] == "acquisition"].copy()
    acquisition_ranks = acquisition.groupby("task_id")["acquisition_rank"].apply(
        lambda ranks: sorted(int(rank) for rank in ranks)
    )
    for ranks in acquisition_ranks:
        if ranks != list(range(1, len(ranks) + 1)):
            raise AssertionError("acquisition ranks are not consecutive")

    observation_lookup = task_observations.set_index("observation_id", drop=False)
    label_ids = acquisition["label_observation_id"].astype(str).tolist()
    missing_labels = sorted(set(label_ids) - set(observation_lookup.index.astype(str)))
    if missing_labels:
        raise AssertionError(f"missing acquisition observations: {len(missing_labels)}")
    selected_labels = observation_lookup.loc[label_ids]
    selected_keys = set(zip(selected_labels["task_id"], selected_labels["item_uid"], strict=True))
    assignment_keys = set(zip(acquisition["task_id"], acquisition["item_uid"], strict=True))
    if selected_keys != assignment_keys:
        raise AssertionError("an acquisition label does not belong to its assigned task-item")

    item_ids = sorted(set(task_assignments["item_uid"].astype(str)))
    benchmark_items = items[items["item_uid"].astype(str).isin(item_ids)].copy()
    if benchmark_items["item_uid"].nunique() != len(item_ids):
        raise AssertionError("item feature join is incomplete")
    subject_ids = sorted(set(task_assignments["subject_uid"].astype(str)))
    benchmark_subjects = subjects[subjects["subject_uid"].astype(str).isin(subject_ids)]
    if benchmark_subjects["subject_uid"].nunique() != len(subject_ids):
        raise AssertionError("subject feature join is incomplete")

    benchmark_items = benchmark_items.sort_values("item_uid").reset_index(drop=True)
    item_to_index = {item_uid: index for index, item_uid in enumerate(benchmark_items["item_uid"])}
    subject_to_index = {subject_uid: index for index, subject_uid in enumerate(subject_ids)}
    x = _hashed_item_features(benchmark_items)

    training_rows = selected_labels.reset_index(drop=True)
    train_item = np.asarray(
        [item_to_index[str(value)] for value in training_rows["item_uid"]], dtype=np.int64
    )
    train_subject = np.asarray(
        [subject_to_index[str(value)] for value in training_rows["subject_uid"]], dtype=np.int64
    )
    y = training_rows["label"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    theta = np.zeros(len(subject_ids), dtype=np.float64)
    weight_b = rng.normal(0.0, 0.01, x.shape[1])
    weight_a = rng.normal(0.0, 0.01, x.shape[1])

    def probabilities():
        x_i = x[train_item]
        b = x_i @ weight_b
        raw_a = x_i @ weight_a
        a = _softplus(raw_a) + 1e-3
        logits = a * (theta[train_subject] - b)
        return _sigmoid(logits), b, raw_a, a, x_i

    initial_probability, *_ = probabilities()
    initial_loss = _binary_cross_entropy(y, initial_probability)
    learning_rate = 0.08
    for _ in range(optimization_steps):
        probability, b, raw_a, a, x_i = probabilities()
        logit_gradient = (probability - y) / len(y)
        theta_gradient = np.zeros_like(theta)
        np.add.at(theta_gradient, train_subject, logit_gradient * a)
        b_gradient = -logit_gradient * a
        a_gradient = logit_gradient * (theta[train_subject] - b)
        raw_a_gradient = a_gradient * _sigmoid(raw_a)
        weight_b_gradient = x_i.T @ b_gradient
        weight_a_gradient = x_i.T @ raw_a_gradient

        theta -= learning_rate * theta_gradient
        weight_b -= learning_rate * weight_b_gradient
        weight_a -= learning_rate * weight_a_gradient
        theta -= theta.mean()

    final_probability, *_ = probabilities()
    final_loss = _binary_cross_entropy(y, final_probability)
    if not final_loss < initial_loss:
        raise AssertionError("2PL smoke optimization did not reduce training loss")
    if not np.isfinite(final_probability).all() or not (
        (final_probability >= 0.0) & (final_probability <= 1.0)
    ).all():
        raise AssertionError("2PL smoke model emitted an invalid probability")

    nested_budget_checks = 0
    for task_id, task_acquisition in acquisition.groupby("task_id", sort=True):
        ordered = task_acquisition.sort_values("acquisition_rank")["item_uid"].astype(str).tolist()
        previous: set[str] = set()
        for budget in budgets:
            current = set(ordered[:budget])
            if not previous.issubset(current):
                raise AssertionError(f"contexts are not nested for {task_id}")
            previous = current
            nested_budget_checks += 1

    evaluation_items = task_assignments.loc[
        task_assignments["pool"] == "evaluation", ["task_id", "item_uid"]
    ]
    evaluation_keys = set(zip(evaluation_items["task_id"], evaluation_items["item_uid"], strict=True))
    evaluation_rows = task_observations[
        [
            (task_id, item_uid) in evaluation_keys
            for task_id, item_uid in zip(
                task_observations["task_id"], task_observations["item_uid"], strict=True
            )
        ]
    ]
    if evaluation_rows.empty:
        raise AssertionError("evaluation pool contains no response rows")

    return {
        "benchmark_id": benchmark_id,
        "n_tasks": len(task_ids),
        "n_subjects": len(subject_ids),
        "n_distinct_items": len(item_ids),
        "n_observations": len(task_observations),
        "n_acquisition_items": len(acquisition),
        "n_evaluation_rows": len(evaluation_rows),
        "item_feature_shape": list(x.shape),
        "budgets": list(budgets),
        "nested_budget_checks": nested_budget_checks,
        "initial_train_bce": initial_loss,
        "final_train_bce": final_loss,
        "checks": {
            "binary_labels": bool(set(task_observations["label"].unique()).issubset({0, 1})),
            "balanced_item_pools": True,
            "complete_item_join": True,
            "complete_subject_join": True,
            "consecutive_acquisition_ranks": True,
            "nested_contexts": True,
            "no_acquisition_evaluation_overlap": True,
            "finite_probabilities": True,
            "optimization_reduced_loss": True,
        },
    }
