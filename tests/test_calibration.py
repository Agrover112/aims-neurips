"""Tests for post-hoc episodic calibration."""

from __future__ import annotations

import numpy as np

from aims_starter.calibration import AnchorCalibrator, _sigmoid, context_rate


def _samples(seed: int = 3, n: int = 20_000):
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(scale=1.2, size=n)
    labels = (rng.random(n) < _sigmoid(true_logits)).astype(float)
    raw_logits = 2.0 * true_logits + 0.8
    context_rates = _sigmoid(rng.normal(scale=1.0, size=n))
    budgets = rng.choice(np.array([0, 1, 3, 7, 15, 31]), size=n)
    return raw_logits, labels, context_rates, budgets


def test_context_rate_is_laplace_smoothed() -> None:
    assert abs(context_rate(np.array([])) - 0.5) < 1e-12
    assert abs(context_rate(np.array([1.0, 1.0])) - 3.0 / 4.0) < 1e-12
    assert abs(context_rate(np.array([0.0, 1.0, 0.0, 0.0])) - 2.0 / 6.0) < 1e-12


def test_logistic_recalibration_recovers_an_inverse_scale() -> None:
    raw_logits, labels, _, budgets = _samples()
    calibrator = AnchorCalibrator.fit(raw_logits, labels, np.full_like(labels, 0.5), budgets)
    entry = calibrator.budgets_map[7]
    assert entry.fitted
    # Raw logits are doubled and shifted, so the fit should halve the scale.
    assert 0.35 < entry.scale < 0.65
    assert -0.8 < entry.bias < 0.2


def test_calibration_reduces_brier_on_held_out_samples() -> None:
    raw_logits, labels, rates, budgets = _samples()
    half = len(labels) // 2
    calibrator = AnchorCalibrator.fit(
        raw_logits[:half], labels[:half], rates[:half], budgets[:half]
    )
    probability = calibrator.transform(raw_logits[half:], 7, np.full(3, labels[half:].mean()))
    raw_probability = _sigmoid(raw_logits[half:])
    calibrated_brier = float(np.mean((probability - labels[half:]) ** 2))
    raw_brier = float(np.mean((raw_probability - labels[half:]) ** 2))
    assert calibrated_brier < raw_brier


def test_small_budgets_fall_back_to_the_raw_sigmoid() -> None:
    rng = np.random.default_rng(11)
    logits = rng.normal(size=50)
    labels = (rng.random(50) < _sigmoid(logits)).astype(float)
    calibrator = AnchorCalibrator.fit(logits, labels, np.full(50, 0.5), np.full(50, 0))
    assert not calibrator.budgets_map[0].fitted
    transformed = calibrator.transform(logits, 0, np.array([1.0]))
    assert np.allclose(transformed, _sigmoid(logits))


def test_anchor_weight_pulls_toward_the_context_rate() -> None:
    logits, labels, rates, budgets = _samples()
    calibrator = AnchorCalibrator.fit(logits, labels, rates, budgets)
    entry = calibrator.budgets_map[15]
    assert entry.fitted
    entry.anchor = 1.0
    logits = np.array([3.0, -3.0])
    transformed = calibrator.transform(logits, 15, np.array([0.0, 1.0]))
    assert np.allclose(transformed, np.full(2, context_rate(np.array([0.0, 1.0]))))
