"""Reproducible random acquisition for the streaming interface."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _uniform(input_pair: list[dict[str, Any]], context: dict[str, Any] | None) -> float:
    """Stable pseudo-random number in [0, 1) for one streaming decision."""

    payload = json.dumps(
        {
            "input": input_pair,
            "labels_acquired": (context or {}).get("labels_acquired", 0),
            "items_remaining": (context or {}).get("items_remaining", 0),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(1 << 64)


def acquisition_function(
    input: list[dict[str, Any]],
    prediction: float | None = None,
    labeled: list | None = None,
    context: dict[str, Any] | None = None,
) -> bool | float:
    """Uniform online sampling, while reserving enough later candidates."""

    del prediction, labeled
    if context is None:
        # Legacy pool-ranking evaluator: a stable random priority.
        return _uniform(input, None)

    remaining = int(context["labels_remaining"])
    candidates = int(context["items_remaining"])
    if remaining <= 0:
        return False
    if candidates <= remaining:
        return True
    return _uniform(input, context) < remaining / candidates
