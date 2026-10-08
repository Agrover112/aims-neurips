import unittest

from aims_predictor import PredictorConfig, predict_probability

SUBJECT = {
    "normalized_name": "test-model",
    "provider": "test-provider",
    "release_date": "2026-01-01",
    "access_date": "2026-01-02",
    "harness": "test",
    "reasoning_effort": "low",
    "harness_version": "1",
    "subject_features_extra": "",
}


def item(benchmark: str) -> dict[str, str]:
    return {"item_content": "x", "item_features": "", "interactors": "", "benchmark_id": benchmark}


class PredictorTests(unittest.TestCase):
    def test_no_labels_returns_neutral_prior(self):
        self.assertEqual(predict_probability([SUBJECT, item("a")], []), 0.5)

    def test_matching_group_evidence_changes_prediction(self):
        labeled = [[[SUBJECT, item("a")], 1], [[SUBJECT, item("a")], 1]]
        self.assertGreater(predict_probability([SUBJECT, item("a")], labeled), 0.5)

    def test_other_subject_is_ignored(self):
        other = {**SUBJECT, "normalized_name": "other"}
        labeled = [[[other, item("a")], 1]]
        self.assertEqual(predict_probability([SUBJECT, item("a")], labeled), 0.5)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            PredictorConfig(prior_mean=2.0)
