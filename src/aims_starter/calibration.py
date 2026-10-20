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

from dataclasses import dataclass

import numpy as np

DEFAULT_ANCHORS = tuple(np.linspace(0.0, 1.0, 21))
NEWTON_STEPS = 25


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def context_rate(labels: np.ndarray) -> float:
    """Laplace-smoothed base rate of the context labels."""

    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Newton fit of a two-parameter logistic regression."""

    weights = np.zeros(features.shape[1], dtype=float)
    for _ in range(NEWTON_STEPS):
        probability = _sigmoid(features @ weights)
        gradient = features.T @ (labels - probability)
        curvature = (features * (probability * (1.0 - probability))[:, None]).T @ features
        curvature += 1e-8 * np.eye(features.shape[1])
        step = np.linalg.solve(curvature, gradient)
        weights = weights + step
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
            best = None
            for anchor in anchors:
                weights = _fit_logistic(features, y)
                probability = _sigmoid(weights[0] * z + weights[1])
                blended = (1.0 - anchor) * probability + anchor * rate
                loss = _binary_cross_entropy(blended, y)
                if best is None or loss < best[0]:
                    best = (loss, float(weights[0]), float(weights[1]), float(anchor))
            if best is None:
                result[budget] = BudgetCalibrator(1.0, 0.0, 0.0, count, False)
            else:
                result[budget] = BudgetCalibrator(best[1], best[2], best[3], count, True)
        return cls(budgets=tuple(budget_grid), budgets_map=result)

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
            }
            for budget, calibrator in sorted(self.budgets_map.items())
        ]
