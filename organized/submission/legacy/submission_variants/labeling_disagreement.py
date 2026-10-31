"""Priority-2 acquisition: neural/Beta disagreement with the CNP predictor.

Notation: ``P_C + A_D``. The predictor is CNP-AIRT. The acquisition score is
``|p_C - p_beta|``, where ``p_beta`` is the smoothed revealed rate for the same
subject and benchmark. The policy buys a candidate when the two sources
disagree by at least 0.15 and reserves enough candidates to fill the 31-label
budget. The neural prediction is used only for this acquisition decision.
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
    prediction: float | None = None,
    labeled: list | None = None,
    context: dict | None = None,
) -> bool | float:
    """Return a boolean acquisition decision, or a legacy pool-ranking score."""

    if context is None:
        content = str(input[1].get("item_content", ""))
        return float(-abs(len(content) - 1200))
    if prediction is None:
        raise ValueError("disagreement acquisition requires the current prediction")
    remaining = int(context["labels_remaining"])
    candidates = int(context["items_remaining"])
    if remaining <= 0:
        return False
    if candidates <= remaining:
        return True
    probability = _beta_mean(_group_labels(input, labeled))
    return bool(abs(float(prediction) - probability) >= DISTANCE_THRESHOLD)
