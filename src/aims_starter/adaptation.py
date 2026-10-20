"""Ability priors estimated from training tasks.

Two channels, both computed without held-out labels:

- a benchmark-level mean ability, the random effect every item in an episode
  shares;
- a subject deviation regressed on the subject metadata embedding, with the
  ridge penalty chosen by leave-one-out error.

The deviation is what separates a lenient subject from a strict one at budget
zero, where the context is empty and a zero-mean MAP anchor has no level to
start from.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def map_ability(
    a_C: np.ndarray,
    b_C: np.ndarray,
    y_C: np.ndarray,
    *,
    prior_mean: float = 0.0,
    prior_variance: float = 4.0,
    steps: int = 12,
) -> float:
    """Newton MAP for one 2PL ability, matching the torch anchor in the models."""

    a = np.asarray(a_C, dtype=float)
    b = np.asarray(b_C, dtype=float)
    y = np.asarray(y_C, dtype=float)
    theta = float(prior_mean)
    if y.size == 0:
        return theta
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        gradient = float(np.sum(a * (y - p))) - (theta - prior_mean) / prior_variance
        hessian = -float(np.sum(a * a * p * (1.0 - p))) - 1.0 / prior_variance
        theta = float(np.clip(theta - gradient / hessian, -4.0, 4.0))
    return theta


@dataclass(frozen=True)
class AbilityPrior:
    """Benchmark mean plus a ridge subject deviation."""

    benchmark_mean: dict[str, float]
    global_mean: float
    feature_mean: np.ndarray
    weights: np.ndarray
    intercept: float
    alpha: float
    loo_mse: float | None
    n_fit: int
    n_benchmarks: int

    def predict(
        self,
        benchmark_id: str,
        subject_embedding: np.ndarray,
        *,
        subject: bool = True,
    ) -> float:
        """Return the prior mean for one benchmark-subject pair."""

        base = self.benchmark_mean.get(str(benchmark_id), self.global_mean)
        if not subject or self.weights.size == 0:
            return float(np.clip(base, -4.0, 4.0))
        centered = np.asarray(subject_embedding, dtype=float) - self.feature_mean
        value = base + self.intercept + centered @ self.weights
        return float(np.clip(value, -4.0, 4.0))


def fit_ability_prior(
    task_abilities: np.ndarray,
    task_benchmarks: list[str],
    task_subjects: list[str],
    subject_embeddings: np.ndarray,
    subject_to_index: dict[str, int],
    *,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    min_group_size: int = 2,
) -> AbilityPrior:
    """Fit the benchmark mean and the subject-deviation ridge.

    Subject targets are deviations from the leave-one-task-out benchmark mean,
    so a benchmark with a single training task carries no subject information
    and is excluded from the ridge fit.
    """

    abilities = np.asarray(task_abilities, dtype=float)
    if len(abilities) != len(task_benchmarks) or len(abilities) != len(task_subjects):
        raise ValueError("abilities, benchmarks, and subjects must have the same length")

    by_benchmark: dict[str, list[int]] = defaultdict(list)
    for index, benchmark in enumerate(task_benchmarks):
        by_benchmark[str(benchmark)].append(index)
    benchmark_mean = {
        benchmark: float(abilities[indices].mean()) for benchmark, indices in by_benchmark.items()
    }
    global_mean = float(abilities.mean())

    rows: list[int] = []
    targets: list[float] = []
    for indices in by_benchmark.values():
        if len(indices) < min_group_size:
            continue
        for index in indices:
            others = [candidate for candidate in indices if candidate != index]
            rows.append(index)
            targets.append(float(abilities[index] - abilities[others].mean()))

    empty = AbilityPrior(
        benchmark_mean=benchmark_mean,
        global_mean=global_mean,
        feature_mean=np.zeros(subject_embeddings.shape[1]),
        weights=np.zeros(0),
        intercept=0.0,
        alpha=float("nan"),
        loo_mse=None,
        n_fit=0,
        n_benchmarks=len(by_benchmark),
    )
    if len(rows) < 8:
        return empty

    features = np.stack(
        [subject_embeddings[subject_to_index[task_subjects[index]]] for index in rows]
    ).astype(float)
    target = np.asarray(targets, dtype=float)
    feature_mean = features.mean(axis=0)
    centered = features - feature_mean
    target_mean = float(target.mean())
    centered_target = target - target_mean

    best: tuple[float, float, np.ndarray] | None = None
    identity = np.eye(centered.shape[1])
    for alpha in alphas:
        inverse = np.linalg.inv(centered.T @ centered + alpha * identity)
        weights = inverse @ centered.T @ centered_target
        fitted = centered @ weights
        leverage = np.einsum("ij,jk,ik->i", centered, inverse, centered)
        residual = (centered_target - fitted) / np.maximum(1.0 - leverage, 1e-6)
        mse = float(np.mean(residual**2))
        if best is None or mse < best[0]:
            best = (mse, float(alpha), weights)

    assert best is not None
    mse, alpha, weights = best
    return AbilityPrior(
        benchmark_mean=benchmark_mean,
        global_mean=global_mean,
        feature_mean=feature_mean,
        weights=weights,
        intercept=target_mean,
        alpha=alpha,
        loo_mse=mse,
        n_fit=len(rows),
        n_benchmarks=len(by_benchmark),
    )
