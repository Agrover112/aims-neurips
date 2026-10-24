"""Tests for the pure numeric helpers in the benchmark smoke module.

Only the dependency-light helpers are covered here; ``run_benchmark_smoke``
itself is orchestration over real episode frames and is out of scope for a
unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aims_starter import calibration
from aims_starter.smoke import (
    _binary_cross_entropy,
    _hashed_item_features,
    _sigmoid,
    _softplus,
)


def _items(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"model_text": texts})


def test_sigmoid_midpoint_and_bounds() -> None:
    assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    values = _sigmoid(np.array([-500.0, -1.0, 1.0, 500.0]))
    assert np.all(values > 0.0)
    assert np.all(values < 1.0)
    assert np.all(np.isfinite(values))


def test_sigmoid_is_symmetric_about_zero() -> None:
    values = np.array([-3.0, -0.5, 0.25, 2.0])
    assert _sigmoid(values) == pytest.approx(1.0 - _sigmoid(-values))


def test_softplus_is_positive_and_asymptotically_identity() -> None:
    values = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
    result = _softplus(values)

    assert np.all(result > 0.0)
    assert np.all(np.isfinite(result))
    # For large positive x, softplus(x) -> x.
    assert result[-1] == pytest.approx(50.0, abs=1e-6)
    assert _softplus(np.array([0.0]))[0] == pytest.approx(np.log(2.0))


def test_hashed_item_features_shape_and_bias_column() -> None:
    matrix = _hashed_item_features(_items(["alpha beta", "gamma"]), dimensions=16)

    assert matrix.shape == (2, 19)
    assert np.all(matrix[:, 0] == 1.0)


def test_hashed_item_features_are_deterministic() -> None:
    texts = ["refuse the harmful request", "comply with the benign one"]
    first = _hashed_item_features(_items(texts))
    second = _hashed_item_features(_items(texts))

    assert np.array_equal(first, second)


def test_hashed_item_features_distinguish_different_items() -> None:
    matrix = _hashed_item_features(_items(["alpha beta gamma", "delta epsilon zeta"]))
    assert not np.allclose(matrix[0, 3:], matrix[1, 3:])


def test_hashed_item_features_flag_blank_text() -> None:
    matrix = _hashed_item_features(_items(["   ", "real text"]))
    assert matrix[0, 2] == 1.0
    assert matrix[1, 2] == 0.0


def test_hashed_item_features_handle_missing_text() -> None:
    matrix = _hashed_item_features(_items([None, "text"]))
    assert np.all(np.isfinite(matrix))
    assert matrix[0, 2] == 1.0


def test_hashed_item_features_norm_floor_leaves_sparse_rows_unscaled() -> None:
    """Rows whose hashed block has norm below 1.0 are divided by 1.0, not by the norm."""

    matrix = _hashed_item_features(_items(["token"]))
    norm = float(np.linalg.norm(matrix[0, 3:]))
    assert norm == pytest.approx(1.0)


def test_binary_cross_entropy_known_value() -> None:
    labels = np.array([1.0, 0.0])
    probabilities = np.array([0.5, 0.5])
    assert _binary_cross_entropy(labels, probabilities) == pytest.approx(np.log(2.0))


def test_binary_cross_entropy_clips_extreme_probabilities() -> None:
    """Clipping at 1e-7 keeps a confidently wrong forecast finite."""

    result = _binary_cross_entropy(np.array([1.0]), np.array([0.0]))
    assert np.isfinite(result)
    assert result > 0.0


def test_binary_cross_entropy_agrees_with_calibration_module() -> None:
    """smoke and calibration carry independent copies of the same formula."""

    rng = np.random.default_rng(11)
    labels = (rng.random(64) < 0.5).astype(float)
    probabilities = rng.uniform(0.05, 0.95, size=64)

    assert _binary_cross_entropy(labels, probabilities) == pytest.approx(
        calibration._binary_cross_entropy(probabilities, labels)
    )
