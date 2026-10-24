import math
import unittest

from aims_starter.metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_WEIGHTS,
    brier_alc,
    brier_score,
    calibration_ece,
)


class MetricsTests(unittest.TestCase):
    def test_brier_score_known_value(self):
        self.assertAlmostEqual(brier_score([0, 1], [0.25, 0.75]), 0.0625)

    def test_constant_half_has_quarter_brier(self):
        self.assertAlmostEqual(brier_score([0, 1, 1, 0], [0.5] * 4), 0.25)

    def test_ece_handles_probability_one(self):
        self.assertAlmostEqual(calibration_ece([1, 0], [1.0, 0.0]), 0.0)

    def test_official_alc_weights(self):
        self.assertAlmostEqual(brier_alc([0, 1, 1, 1, 1, 0]), 0.8)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_score([1], [1.1])


class DefaultGridTests(unittest.TestCase):
    """The module-level budget grid is the competition contract."""

    def test_budgets_and_weights_line_up(self):
        self.assertEqual(len(DEFAULT_BUDGETS), len(DEFAULT_WEIGHTS))

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(math.fsum(DEFAULT_WEIGHTS), 1.0, places=12)

    def test_budgets_are_strictly_increasing_from_zero(self):
        self.assertEqual(DEFAULT_BUDGETS[0], 0)
        self.assertEqual(list(DEFAULT_BUDGETS), sorted(set(DEFAULT_BUDGETS)))

    def test_default_weights_are_the_brier_alc_default(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertAlmostEqual(brier_alc(scores), brier_alc(scores, DEFAULT_WEIGHTS))


class CalibrationEceTests(unittest.TestCase):
    def test_single_sample_is_the_absolute_gap(self):
        self.assertAlmostEqual(calibration_ece([1], [0.3]), 0.7)
        self.assertAlmostEqual(calibration_ece([0], [0.3]), 0.3)

    def test_all_same_label(self):
        # Both land in the [0.2, 0.3) and [0.4, 0.5) bins, each weighted 0.5.
        self.assertAlmostEqual(calibration_ece([1, 1], [0.25, 0.45]), 0.65)

    def test_realistic_miscalibration(self):
        # Values are kept off the bin edges so each pair shares one bin.
        # Bin 8 holds two confident calls, one of which is wrong:
        # mean(p) = 0.85, mean(y) = 0.5, weighted by 2/4.
        # Bin 1 holds two diffident calls, one of which is right:
        # mean(p) = 0.15, mean(y) = 0.5, weighted by 2/4.
        outcomes = [1, 0, 0, 1]
        probabilities = [0.82, 0.88, 0.12, 0.18]
        self.assertAlmostEqual(calibration_ece(outcomes, probabilities), 0.35)

    def test_values_on_a_bin_edge_fall_into_the_upper_bin(self):
        # int(0.8 * 10) == 8 and int(0.9 * 10) == 9, so these do not share a bin.
        outcomes = [1, 0, 0, 1]
        probabilities = [0.8, 0.9, 0.1, 0.2]
        self.assertAlmostEqual(calibration_ece(outcomes, probabilities), 0.5)

    def test_perfect_calibration_within_a_bin_is_zero(self):
        self.assertAlmostEqual(calibration_ece([1, 0], [0.5, 0.5]), 0.0)

    def test_non_default_bin_count(self):
        outcomes = [1, 0, 1, 0]
        probabilities = [0.2, 0.4, 0.6, 0.8]
        # bins=2 groups {0.2, 0.4} and {0.6, 0.8}; each contributes 0.5 * 0.2.
        self.assertAlmostEqual(calibration_ece(outcomes, probabilities, bins=2), 0.2)
        # bins=10 isolates every point, contributing 0.25 * |y - p| each.
        self.assertAlmostEqual(calibration_ece(outcomes, probabilities, bins=10), 0.6)

    def test_probability_one_lands_in_the_final_bin(self):
        # Without the min() clamp this would index past the last bin.
        self.assertAlmostEqual(calibration_ece([0, 0], [1.0, 1.0]), 1.0)

    def test_single_bin_reduces_to_the_global_gap(self):
        outcomes = [1, 1, 0, 0]
        probabilities = [0.9, 0.7, 0.3, 0.1]
        self.assertAlmostEqual(calibration_ece(outcomes, probabilities, bins=1), 0.0)

    def test_nonpositive_bins_are_rejected(self):
        for bins in (0, -1):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                calibration_ece([1, 0], [0.6, 0.4], bins=bins)


class ValidationTests(unittest.TestCase):
    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_score([], [])

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            brier_score([1, 0], [0.5])

    def test_nonbinary_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_score([0.5], [0.5])

    def test_nan_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_score([1], [float("nan")])

    def test_infinite_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_score([1], [float("inf")])

    def test_negative_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_score([1], [-0.1])

    def test_boundary_probabilities_are_accepted(self):
        self.assertAlmostEqual(brier_score([1, 0], [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(brier_score([0, 1], [1.0, 0.0]), 1.0)


class BrierAlcTests(unittest.TestCase):
    def test_weights_that_do_not_sum_to_one_are_rejected(self):
        with self.assertRaises(ValueError):
            brier_alc([0.2, 0.2], [0.5, 0.4])

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            brier_alc([0.2, 0.2], [1.0])

    def test_empty_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            brier_alc([], [])

    def test_negative_score_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_alc([-0.1, 0.2], [0.5, 0.5])

    def test_negative_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            brier_alc([0.1, 0.2], [1.5, -0.5])

    def test_constant_scores_pass_through_unchanged(self):
        self.assertAlmostEqual(brier_alc([0.25] * 6), 0.25)


if __name__ == "__main__":
    unittest.main()
