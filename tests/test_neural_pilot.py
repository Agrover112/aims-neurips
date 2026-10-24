"""Tests pinning the neural-pilot metric helpers against ``metrics``.

``neural_pilot`` carries its own ``_brier`` and ``_ece`` rather than importing
from ``metrics``. The two ECE implementations bin differently:

* ``neural_pilot._ece`` walks explicit ``linspace`` edges with a right-open
  interval ``[edges[i], edges[i + 1])`` and folds ``p == 1.0`` into the last bin.
* ``metrics.calibration_ece`` indexes directly with
  ``min(int(p * bins), bins - 1)``.

Different parts of the codebase report ECE from different implementations, so
these tests make the relationship explicit and monitored.
"""

from __future__ import annotations

import numpy as np
import pytest

from aims_starter.metrics import brier_score, calibration_ece
from aims_starter.neural_pilot import _brier, _ece


def test_brier_agrees_with_metrics_brier_score() -> None:
    rng = np.random.default_rng(3)
    labels = (rng.random(128) < 0.5).astype(float)
    probabilities = rng.uniform(0.0, 1.0, size=128)

    assert _brier(labels, probabilities) == pytest.approx(
        brier_score(labels.tolist(), probabilities.tolist())
    )


def test_brier_known_value() -> None:
    assert _brier(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert _brier(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_ece_known_value_matches_hand_computation() -> None:
    """Two predictions in one bin: |mean(y) - mean(p)| weighted by bin share."""

    labels = np.array([1.0, 0.0])
    probabilities = np.array([0.52, 0.54])
    # Both land in bin [0.5, 0.6); mean(y) = 0.5, mean(p) = 0.53.
    assert _ece(labels, probabilities) == pytest.approx(0.03)


def test_ece_all_in_one_bin_equals_absolute_gap() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    probabilities = np.full(4, 0.25)
    assert _ece(labels, probabilities) == pytest.approx(0.25)


def test_ece_agrees_with_metrics_on_interior_values() -> None:
    """Away from bin edges, the two binning rules must coincide."""

    rng = np.random.default_rng(17)
    labels = (rng.random(256) < 0.5).astype(float)
    # Keep values clear of the 0.1-spaced edges used at bins=10.
    probabilities = rng.uniform(0.0, 1.0, size=256).round(2) + 0.005

    assert _ece(labels, probabilities) == pytest.approx(
        calibration_ece(labels.tolist(), probabilities.tolist())
    )


def test_ece_bin_edge_behavior_is_pinned() -> None:
    """Probabilities sitting exactly on bin edges, including p == 1.0.

    Both implementations place a value exactly on an edge into the upper bin,
    and both fold ``p == 1.0`` into the final bin, so they agree here today.
    This test fails loudly if either binning rule is changed in isolation.
    """

    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    probabilities = np.array([0.0, 0.1, 0.3, 0.9, 1.0])

    pilot = _ece(labels, probabilities)
    metrics = calibration_ece(labels.tolist(), probabilities.tolist())
    assert pilot == pytest.approx(metrics)


def test_ece_probability_one_is_not_dropped() -> None:
    """A bare right-open final bin would silently ignore p == 1.0."""

    labels = np.array([0.0, 0.0])
    probabilities = np.array([1.0, 1.0])
    # Every prediction is maximally confident and wrong, so ECE is 1.0.
    assert _ece(labels, probabilities) == pytest.approx(1.0)


def test_ece_respects_non_default_bin_count() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])

    # bins=2: each half-bin holds one correct and one wrong call, contributing
    # 0.5 * |0.5 - 0.3| and 0.5 * |0.5 - 0.7|.
    assert _ece(labels, probabilities, bins=2) == pytest.approx(0.2)
    # bins=10: every point lands alone, so each contributes 0.25 * |y - p|.
    assert _ece(labels, probabilities, bins=10) == pytest.approx(0.6)
