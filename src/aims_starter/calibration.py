"""Post-hoc calibration of episodic probability forecasts.

The forecast for a task arrives as a logit, one per evaluation item, at each
budget. Two miscalibrations show up in the BeaverTails pilot:

1. a global scale and bias error, which is the same across tasks, and
2. a level error on tasks whose base rate is far from the pool average.

Both are corrected per budget. The scale and bias term is a logistic
recalibration of the logit. The level term anchors the calibrated probability
toward the context base rate, which is the only task-specific quantity a
deployment policy can trust at low budgets. All parameters are fitted on
meta-calibration samples collected from training benchmarks, never on the
evaluation tasks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

DEFAULT_ANCHORS = tuple(np.linspace(0.0, 1.0, 21))
NEWTON_STEPS = 25
PROBABILITY_EPS = 1e-8


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def context_rate(labels: np.ndarray) -> float:
    """Laplace-smoothed base rate of the context labels."""

    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    nonnegative_scale: bool = False,
) -> np.ndarray:
    """Newton fit of a two-parameter logistic regression."""

    weights = np.zeros(features.shape[1], dtype=float)
    for _ in range(NEWTON_STEPS):
        probability = _sigmoid(features @ weights)
        gradient = features.T @ (labels - probability)
        curvature = (features * (probability * (1.0 - probability))[:, None]).T @ features
        curvature += 1e-8 * np.eye(features.shape[1])
        step = np.linalg.solve(curvature, gradient)
        weights = weights + step
        if nonnegative_scale:
            weights[0] = max(float(weights[0]), 0.0)
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return weights


def _fit_blend_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    rates: np.ndarray,
    anchor: float,
    initial: np.ndarray,
    *,
    nonnegative_scale: bool = True,
) -> np.ndarray:
    """Fit scale/bias for a fixed anchor blend.

    The probability is ``(1 - anchor) * sigmoid(features @ weights) + anchor *
    rates``.  The expected Hessian keeps the Newton step well behaved even when
    the anchor term dominates.
    """

    if anchor <= 0.0:
        return _fit_logistic(
            features, labels, nonnegative_scale=nonnegative_scale
        )
    if anchor >= 1.0:
        return np.asarray(initial, dtype=float).copy()

    weights = np.asarray(initial, dtype=float).copy()
    if nonnegative_scale:
        weights[0] = max(float(weights[0]), 0.0)
    for _ in range(NEWTON_STEPS):
        logits = features @ weights
        sigmoid = np.clip(_sigmoid(logits), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
        probability = np.clip(
            (1.0 - anchor) * sigmoid + anchor * rates,
            PROBABILITY_EPS,
            1.0 - PROBABILITY_EPS,
        )
        blend_slope = (1.0 - anchor) * sigmoid * (1.0 - sigmoid)
        gradient_factor = (probability - labels) * blend_slope / (
            probability * (1.0 - probability)
        )
        gradient = features.T @ gradient_factor
        information = blend_slope * blend_slope / (
            probability * (1.0 - probability)
        )
        curvature = (features * information[:, None]).T @ features
        curvature += 1e-8 * np.eye(features.shape[1])
        step = np.linalg.solve(curvature, gradient)
        weights = weights - step
        if nonnegative_scale:
            weights[0] = max(float(weights[0]), 0.0)
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return weights


def _binary_cross_entropy(probability: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return -float(np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


@dataclass
class BudgetCalibrator:
    """Scale, bias, and context-anchor weight for one budget."""

    scale: float
    bias: float
    anchor: float
    samples: int
    fitted: bool

    def transform(self, logits: np.ndarray, *, base_rate: float | None = None) -> np.ndarray:
        probability = _sigmoid(self.scale * np.asarray(logits, dtype=float) + self.bias)
        if self.anchor <= 0.0 or base_rate is None:
            return probability
        return (1.0 - self.anchor) * probability + self.anchor * float(base_rate)


@dataclass
class AnchorCalibrator:
    """Per-budget calibration fitted on meta-calibration samples."""

    budgets: tuple[int, ...]
    budgets_map: dict[int, BudgetCalibrator]
    joint_anchor: bool = True
    nonnegative_scale: bool = True

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        context_rates: np.ndarray,
        budgets: np.ndarray,
        *,
        budget_grid: tuple[int, ...] = (0, 1, 3, 7, 15, 31),
        minimum_samples: int = 200,
        anchors: tuple[float, ...] = DEFAULT_ANCHORS,
        joint_anchor: bool = True,
        nonnegative_scale: bool = True,
    ) -> AnchorCalibrator:
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)
        context_rates = np.asarray(context_rates, dtype=float)
        budgets = np.asarray(budgets)
        result: dict[int, BudgetCalibrator] = {}
        for budget in budget_grid:
            mask = budgets == budget
            count = int(mask.sum())
            if count < minimum_samples or len(np.unique(labels[mask])) < 2:
                result[budget] = BudgetCalibrator(1.0, 0.0, 0.0, count, False)
                continue
            z = logits[mask]
            y = labels[mask]
            rate = context_rates[mask]
            features = np.column_stack([z, np.ones_like(z)])
            base_weights = _fit_logistic(
                features, y, nonnegative_scale=nonnegative_scale
            )
            best = None
            for anchor in anchors:
                if joint_anchor and 0.0 < float(anchor) < 1.0:
                    weights = _fit_blend_logistic(
                        features,
                        y,
                        rate,
                        float(anchor),
                        base_weights,
                        nonnegative_scale=nonnegative_scale,
                    )
                else:
                    weights = base_weights
                probability = _sigmoid(weights[0] * z + weights[1])
                blended = (1.0 - anchor) * probability + anchor * rate
                loss = _binary_cross_entropy(blended, y)
                if best is None or loss < best[0]:
                    best = (loss, float(weights[0]), float(weights[1]), float(anchor))
            if best is None:
                result[budget] = BudgetCalibrator(1.0, 0.0, 0.0, count, False)
            else:
                if best[1] < 0.0 and best[3] < 1.0:
                    warnings.warn(
                        f"negative calibration scale at budget {budget}: "
                        f"{best[1]:.4f}; this inverts the item ranking",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                result[budget] = BudgetCalibrator(best[1], best[2], best[3], count, True)
        return cls(
            budgets=tuple(budget_grid),
            budgets_map=result,
            joint_anchor=joint_anchor,
            nonnegative_scale=nonnegative_scale,
        )

    def transform(self, logits: np.ndarray, budget: int, context_labels: np.ndarray) -> np.ndarray:
        calibrator = self.budgets_map.get(int(budget))
        if calibrator is None or not calibrator.fitted:
            return _sigmoid(np.asarray(logits, dtype=float))
        return calibrator.transform(logits, base_rate=context_rate(context_labels))

    def summary(self) -> list[dict[str, float | int | bool]]:
        return [
            {
                "budget": budget,
                "scale": calibrator.scale,
                "bias": calibrator.bias,
                "anchor": calibrator.anchor,
                "samples": calibrator.samples,
                "fitted": calibrator.fitted,
                "joint_anchor": self.joint_anchor,
                "nonnegative_scale": self.nonnegative_scale,
            }
            for budget, calibrator in sorted(self.budgets_map.items())
        ]
