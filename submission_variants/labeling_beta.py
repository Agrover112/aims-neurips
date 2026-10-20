"""Priority-2 acquisition: Beta-mean acquisition with the neural predictor.

Notation: ``P_C + A_U(beta)``. The predictor is CNP-AIRT, but the acquisition
decision ignores the neural prediction. It uses only the revealed outcomes for
the same subject and benchmark, smoothed by a Beta(2, 2) prior. The policy buys
a candidate when the smoothed group mean is within 0.15 of 0.5 and reserves
enough candidates to fill the 31-label budget.
"""

from __future__ import annotations

from typing import Any

PRIOR_STRENGTH = 4.0
DISTANCE_THRESHOLD = 0.15


def _subject_key(subject: dict[str, Any]) -> tuple[str, ...]:
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
    return tuple(str(subject.get(field, "") or "") for field in fields)


def _group_labels(input: list[dict], labeled: list | None) -> list[int]:
    subject, item = input
    key = _subject_key(subject)
    benchmark = str(item.get("benchmark_id", ""))
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
        if outcome not in (0, 1, False, True):
            continue
        if _subject_key(seen_subject) != key:
            continue
        if str(seen_item.get("benchmark_id", "")) != benchmark:
            continue
        outcomes.append(int(outcome))
    return outcomes


def _beta_mean(outcomes: list[int]) -> float:
    successes = float(sum(outcomes))
    return (PRIOR_STRENGTH / 2.0 + successes) / (PRIOR_STRENGTH + len(outcomes))


def acquisition_function(
    input: list[dict],
    labeled: list | None = None,
    context: dict | None = None,
) -> bool | float:
    """Return a boolean acquisition decision, or a legacy pool-ranking score."""

    if context is None:
        content = str(input[1].get("item_content", ""))
        return float(-abs(len(content) - 1200))
    remaining = int(context["labels_remaining"])
    candidates = int(context["items_remaining"])
    if remaining <= 0:
        return False
    if candidates <= remaining:
        return True
    probability = _beta_mean(_group_labels(input, labeled))
    return bool(abs(probability - 0.5) <= DISTANCE_THRESHOLD)
