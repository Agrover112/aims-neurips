"""Transparent beta-binomial shrinkage for sparse revealed outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictorConfig:
    """Controls shrinkage toward the neutral and subject-level estimates."""

    prior_mean: float = 0.5
    prior_strength: float = 4.0
    subject_strength: float = 8.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.prior_mean <= 1.0:
            raise ValueError("prior_mean must lie in [0, 1]")
        if self.prior_strength <= 0.0 or self.subject_strength < 0.0:
            raise ValueError("shrinkage strengths must be non-negative, with a positive prior")


def _subject_key(subject: dict[str, str]) -> tuple[str, ...]:
    """Identify a subject by all official subject fields, not its display name alone."""

    fields = (
        "normalized_name",
        "provider",
        "release_date",
        "access_date",
        "harness",
        "reasoning_effort",
        "harness_version",
        "subject_features_extra",
    )
    return tuple(str(subject.get(field, "")) for field in fields)


def _iter_evidence(labeled: Iterable[Any] | None):
    for entry in labeled or ():
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        pair, outcome = entry
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        subject, item = pair
        if not isinstance(subject, dict) or not isinstance(item, dict):
            continue
        if outcome in (0, 1, False, True):
            yield subject, item, int(outcome)


def predict_probability(
    input_pair: list[dict[str, str]],
    labeled: list | None = None,
    config: PredictorConfig | None = None,
) -> float:
    """Estimate correctness using nested beta-binomial shrinkage.

    Evidence for the same subject and anonymous benchmark gets the greatest
    weight. When that evidence is absent, the estimate falls back to the
    subject's outcomes across benchmarks, then to the configured prior.

    This baseline deliberately does not infer item difficulty from text. It is
    a controlled, auditable reference for measuring gains from richer models.
    """

    if not isinstance(input_pair, list) or len(input_pair) != 2:
        raise ValueError("input must be [subject, item]")
    subject, item = input_pair
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise TypeError("subject and item must be dictionaries")

    cfg = config or PredictorConfig()
    target_subject = _subject_key(subject)
    target_benchmark = str(item.get("benchmark_id", ""))
    subject_successes = subject_count = 0
    group_successes = group_count = 0

    for seen_subject, seen_item, outcome in _iter_evidence(labeled):
        if _subject_key(seen_subject) != target_subject:
            continue
        subject_successes += outcome
        subject_count += 1
        if str(seen_item.get("benchmark_id", "")) == target_benchmark:
            group_successes += outcome
            group_count += 1

    subject_mean = (cfg.prior_strength * cfg.prior_mean + subject_successes) / (
        cfg.prior_strength + subject_count
    )
    estimate = (cfg.subject_strength * subject_mean + group_successes) / (
        cfg.subject_strength + group_count
    )
    return float(min(1.0, max(0.0, estimate)))
