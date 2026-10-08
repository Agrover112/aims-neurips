import unittest

from aims_starter.metrics import brier_alc, brier_score, calibration_ece


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
