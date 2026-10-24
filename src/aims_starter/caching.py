"""Fingerprints for experiment cache invalidation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def content_signature(
    paths: Iterable[Path],
    config: Mapping[str, Any],
) -> str:
    """Hash source files plus the run configuration.

    Cache keys must include both.  A checkpoints-only or output-only cache check
    will silently reuse stale artifacts after a training-code change.
    """

    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"cannot fingerprint missing source: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()[:16]
