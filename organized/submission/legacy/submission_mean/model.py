"""Empirical context-mean baseline for the competition interface."""

from __future__ import annotations


_SUBJECT_FIELDS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "reasoning_effort",
    "harness_version",
    "subject_features_extra",
)


def _subject_key(subject: dict) -> tuple[str, ...]:
    return tuple(str(subject.get(field, "")) for field in _SUBJECT_FIELDS)


def predict(input: list, labeled: list | None = None) -> float:
    """Predict the empirical mean of relevant acquired labels.

    With no relevant labels, return the neutral Bernoulli mean 0.5. The model
    deliberately ignores item content and performs no smoothing.
    """

    if not isinstance(input, list) or len(input) != 2:
        raise ValueError("input must be [subject, item]")
    subject, item = input
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise TypeError("subject and item must be dictionaries")

    target_subject = _subject_key(subject)
    target_benchmark = str(item.get("benchmark_id", ""))
    outcomes: list[int] = []
    for entry in labeled or ():
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        pair, outcome = entry
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        seen_subject, seen_item = pair
        if not isinstance(seen_subject, dict) or not isinstance(seen_item, dict):
            continue
        if _subject_key(seen_subject) != target_subject:
            continue
        if str(seen_item.get("benchmark_id", "")) != target_benchmark:
            continue
        if outcome in (0, 1, False, True):
            outcomes.append(int(outcome))

    return float(sum(outcomes) / len(outcomes)) if outcomes else 0.5
