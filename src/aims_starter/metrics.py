"""Competition-aligned metrics with explicit aggregation choices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

DEFAULT_BUDGETS = (0, 1, 3, 7, 15, 31)
DEFAULT_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


def _validated_pairs(y_true: Iterable[float], y_prob: Iterable[float]) -> list[tuple[float, float]]:
    truth = [float(value) for value in y_true]
    probabilities = [float(value) for value in y_prob]
    if not truth or len(truth) != len(probabilities):
        raise ValueError("truth and probability arrays must have the same non-zero length")
    pairs = list(zip(truth, probabilities, strict=True))
    if any(y not in (0.0, 1.0) for y, _ in pairs):
        raise ValueError("outcomes must be binary")
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for _, p in pairs):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return pairs


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    """Return mean squared probability error."""

    pairs = _validated_pairs(y_true, y_prob)
    return sum((probability - outcome) ** 2 for outcome, probability in pairs) / len(pairs)


def calibration_ece(
    y_true: Iterable[float], y_prob: Iterable[float], bins: int = 10
) -> float:
    """Return count-weighted ECE using fixed equal-width bins.

    A prediction of exactly 1.0 belongs to the final bin.
    """

    if bins <= 0:
        raise ValueError("bins must be positive")
    pairs = _validated_pairs(y_true, y_prob)
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for outcome, probability in pairs:
        index = min(int(probability * bins), bins - 1)
        grouped[index].append((outcome, probability))
    total = len(pairs)
    return sum(
        len(group) / total
        * abs(
            sum(probability for _, probability in group) / len(group)
            - sum(outcome for outcome, _ in group) / len(group)
        )
        for group in grouped
        if group
    )


def brier_alc(
    scores: Sequence[float], weights: Sequence[float] = DEFAULT_WEIGHTS
) -> float:
    """Combine per-budget Brier scores using the official normalized log-area weights."""

    if len(scores) != len(weights) or not scores:
        raise ValueError("scores and weights must have the same non-zero length")
    if any(not math.isfinite(float(score)) or float(score) < 0.0 for score in scores):
        raise ValueError("scores must be finite and non-negative")
    if any(float(weight) < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative")
    weight_sum = sum(float(weight) for weight in weights)
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-12):
        raise ValueError("weights must sum to one")
    return sum(float(score) * float(weight) for score, weight in zip(scores, weights, strict=True))

