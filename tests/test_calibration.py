"""Tests for post-hoc episodic calibration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aims_starter import calibration
from aims_starter.calibration import (
    AnchorCalibrator,
    BudgetCalibrator,
    _sigmoid,
    context_rate,
)


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


def test_logistic_fit_runs_once_per_budget(monkeypatch) -> None:
    raw_logits, labels, rates, budgets = _samples(n=2_000)
    calls = []
    original = calibration._fit_logistic

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration, "_fit_logistic", counted)
    AnchorCalibrator.fit(
        raw_logits,
        labels,
        rates,
        budgets,
        budget_grid=(0,),
        joint_anchor=True,
    )
    assert len(calls) == 1


def test_joint_anchor_fit_improves_the_fixed_anchor_loss() -> None:
    raw_logits, labels, rates, budgets = _samples(n=2_000)
    mask = budgets == 7
    z = raw_logits[mask]
    y = labels[mask]
    rate = rates[mask]
    features = np.column_stack([z, np.ones_like(z)])
    initial = calibration._fit_logistic(features, y)
    anchor = 0.8
    fitted = calibration._fit_blend_logistic(
        features, y, rate, anchor, initial
    )

    def loss(weights):
        probability = (1.0 - anchor) * _sigmoid(features @ weights) + anchor * rate
        return calibration._binary_cross_entropy(probability, y)

    assert loss(fitted) <= loss(initial) + 1e-12


def test_nonnegative_scale_is_enforced() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=2_000)
    labels = (z < 0.0).astype(float)
    calibrator = AnchorCalibrator.fit(
        z,
        labels,
        np.full_like(labels, 0.5),
        np.zeros_like(labels),
        budget_grid=(0,),
    )
    entry = calibrator.budgets_map[0]
    assert entry.fitted
    assert entry.scale >= 0.0


def test_negative_scale_warns_when_constraint_is_disabled() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=2_000)
    labels = (z < 0.0).astype(float)
    with pytest.warns(RuntimeWarning, match="negative calibration scale"):
        calibrator = AnchorCalibrator.fit(
            z,
            labels,
            np.full_like(labels, 0.5),
            np.zeros_like(labels),
            budget_grid=(0,),
            nonnegative_scale=False,
            joint_anchor=False,
        )
    assert calibrator.budgets_map[0].scale < 0.0


def test_negative_scale_cannot_occur_under_default_settings() -> None:
    """The default clamp makes the rank-inversion warning unreachable.

    ``docs/uniform-audit.md`` records a Brier-fit calibrator whose negative
    low-budget scales inverted the validation item ranking. With
    ``nonnegative_scale=True`` the Newton step is clamped, so the same
    adversarial data cannot produce a negative scale and no warning fires.
    """

    rng = np.random.default_rng(5)
    labels = (rng.random(4_000) < 0.5).astype(float)
    # Labels are anti-correlated with the logits, so an unconstrained fit
    # would want a negative slope.
    logits = 3.0 * (0.5 - labels) + rng.normal(scale=0.1, size=labels.size)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        calibrator = AnchorCalibrator.fit(
            logits,
            labels,
            np.full_like(labels, 0.5),
            np.zeros_like(labels),
            budget_grid=(0,),
        )
    assert calibrator.budgets_map[0].scale >= 0.0


def test_joint_anchor_false_reuses_one_unblended_fit(monkeypatch) -> None:
    """Pins the opt-out path: only the blend weight varies, never scale/bias.

    With ``joint_anchor=False`` every anchor in the search reuses the single
    unblended logistic fit, so the search selects a blend coefficient against
    fixed scale/bias. This is the documented opt-out behaviour, not a defect;
    the test exists so the semantics of the flag stay explicit.
    """

    seen: list[np.ndarray] = []
    original = calibration._fit_logistic

    def spy(features, labels, **kwargs):
        weights = original(features, labels, **kwargs)
        seen.append(weights)
        return weights

    monkeypatch.setattr(calibration, "_fit_logistic", spy)
    logits, labels, rates, budgets = _samples()
    calibrator = AnchorCalibrator.fit(
        logits, labels, rates, budgets, budget_grid=(7,), joint_anchor=False
    )

    assert len(seen) == 1
    entry = calibrator.budgets_map[7]
    assert entry.scale == pytest.approx(float(seen[0][0]))
    assert entry.bias == pytest.approx(float(seen[0][1]))


def test_fit_accepts_a_custom_budget_grid() -> None:
    logits, labels, rates, budgets = _samples()
    calibrator = AnchorCalibrator.fit(
        logits, labels, rates, budgets, budget_grid=(0, 7)
    )

    assert calibrator.budgets == (0, 7)
    assert set(calibrator.budgets_map) == {0, 7}


def test_transform_falls_back_to_raw_sigmoid_for_an_unfitted_budget() -> None:
    logits, labels, rates, budgets = _samples()
    calibrator = AnchorCalibrator.fit(
        logits, labels, rates, budgets, budget_grid=(0,)
    )

    probe = np.array([-2.0, 0.0, 2.0])
    result = calibrator.transform(probe, 31, np.array([1.0, 0.0]))
    assert result == pytest.approx(_sigmoid(probe))


def test_summary_reports_every_budget_and_the_fit_flags() -> None:
    logits, labels, rates, budgets = _samples()
    calibrator = AnchorCalibrator.fit(
        logits, labels, rates, budgets, budget_grid=(0, 7), joint_anchor=False
    )
    summary = calibrator.summary()

    assert [row["budget"] for row in summary] == [0, 7]
    for row in summary:
        assert row["joint_anchor"] is False
        assert row["nonnegative_scale"] is True
        assert set(row) == {
            "budget",
            "scale",
            "bias",
            "anchor",
            "samples",
            "fitted",
            "joint_anchor",
            "nonnegative_scale",
        }


def test_budget_calibrator_without_base_rate_skips_the_blend() -> None:
    entry = BudgetCalibrator(scale=2.0, bias=0.5, anchor=0.75, samples=500, fitted=True)
    probe = np.array([-1.0, 0.0, 1.0])

    assert entry.transform(probe, base_rate=None) == pytest.approx(
        _sigmoid(2.0 * probe + 0.5)
    )


def test_budget_calibrator_with_zero_anchor_skips_the_blend() -> None:
    entry = BudgetCalibrator(scale=1.0, bias=0.0, anchor=0.0, samples=500, fitted=True)
    probe = np.array([-1.0, 0.0, 1.0])

    assert entry.transform(probe, base_rate=0.9) == pytest.approx(_sigmoid(probe))


def test_budget_calibrator_with_full_anchor_returns_the_base_rate() -> None:
    entry = BudgetCalibrator(scale=3.0, bias=-2.0, anchor=1.0, samples=500, fitted=True)
    probe = np.array([-5.0, 0.0, 5.0])

    assert entry.transform(probe, base_rate=0.3) == pytest.approx(np.full(3, 0.3))


def test_binary_cross_entropy_known_value() -> None:
    labels = np.array([1.0, 0.0])
    probabilities = np.array([0.5, 0.5])
    assert calibration._binary_cross_entropy(probabilities, labels) == pytest.approx(
        float(np.log(2.0))
    )


def test_binary_cross_entropy_clips_confident_mistakes() -> None:
    result = calibration._binary_cross_entropy(np.array([0.0]), np.array([1.0]))
    assert np.isfinite(result)
    assert result > 0.0


def test_fit_logistic_recovers_a_known_boundary() -> None:
    """A clean separable fit should recover a positive slope."""

    rng = np.random.default_rng(13)
    z = rng.normal(size=4_000)
    labels = (rng.random(4_000) < _sigmoid(1.5 * z - 0.4)).astype(float)
    features = np.column_stack([z, np.ones_like(z)])

    weights = calibration._fit_logistic(features, labels)
    assert weights[0] == pytest.approx(1.5, abs=0.15)
    assert weights[1] == pytest.approx(-0.4, abs=0.15)
