"""Validation-selected, budget-specific Beta shrinkage baseline."""

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

# Selected on the benchmark-held-out validation split only. The prior mean is
# fixed at 0.5; only its concentration varies with the acquired-label budget.
_CONCENTRATION_BY_BUDGET = {
    0: 4.0,
    1: 3.890451449942805,
    3: 3.630780547701014,
    7: 3.715352290971724,
    15: 3.8018939632056123,
    31: 3.981071705534973,
}


def _subject_key(subject: dict) -> tuple[str, ...]:
    return tuple(str(subject.get(field, "")) for field in _SUBJECT_FIELDS)


def predict(input: list, labeled: list | None = None) -> float:
    """Return the posterior mean under a budget-specific symmetric Beta prior."""

    if not isinstance(input, list) or len(input) != 2:
        raise ValueError("input must be [subject, item]")
    subject, item = input
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise TypeError("subject and item must be dictionaries")

    target_subject = _subject_key(subject)
    target_benchmark = str(item.get("benchmark_id", ""))
    successes = 0
    count = 0
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
            successes += int(outcome)
            count += 1

    concentration = _CONCENTRATION_BY_BUDGET.get(count, 4.0)
    alpha = 0.5 * concentration
    return float((alpha + successes) / (concentration + count))
