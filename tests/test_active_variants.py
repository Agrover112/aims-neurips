import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "submission_variants"

SUBJECT = {
    "normalized_name": "subject",
    "provider": "provider",
    "release_date": "2026-01-01",
    "access_date": "2026-01-02",
    "harness": "harness",
    "reasoning_effort": "high",
    "harness_version": "v1",
    "subject_features_extra": "extra",
}
ITEM = {"benchmark_id": "benchmark", "item_content": "x"}
CONTEXT = {
    "labels_remaining": 31,
    "labels_acquired": 0,
    "max_labels": 31,
    "items_remaining": 100,
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, VARIANTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BetaAcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load("labeling_beta")

    def test_neutral_group_purchases(self) -> None:
        labeled = [[[SUBJECT, ITEM], 1], [[SUBJECT, ITEM], 0]]
        self.assertIs(
            self.module.acquisition_function([SUBJECT, ITEM], labeled, CONTEXT),
            True,
        )

    def test_confident_group_skips(self) -> None:
        labeled = [[[SUBJECT, ITEM], 1] for _ in range(6)]
        self.assertIs(
            self.module.acquisition_function([SUBJECT, ITEM], labeled, CONTEXT),
            False,
        )

    def test_other_benchmark_is_ignored(self) -> None:
        other = {**ITEM, "benchmark_id": "other"}
        labeled = [[[SUBJECT, other], 1] for _ in range(6)]
        self.assertIs(
            self.module.acquisition_function([SUBJECT, ITEM], labeled, CONTEXT),
            True,
        )

    def test_reserve_purchases(self) -> None:
        context = {**CONTEXT, "items_remaining": 1}
        self.assertIs(
            self.module.acquisition_function([SUBJECT, ITEM], [], context),
            True,
        )


class DisagreementAcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load("labeling_disagreement")

    def test_large_disagreement_purchases(self) -> None:
        self.assertIs(
            self.module.acquisition_function([SUBJECT, ITEM], 0.9, [], CONTEXT),
            True,
        )

    def test_small_disagreement_skips(self) -> None:
        self.assertIs(
            self.module.acquisition_function([SUBJECT, ITEM], 0.55, [], CONTEXT),
            False,
        )

    def test_missing_prediction_fails(self) -> None:
        with self.assertRaises(ValueError):
            self.module.acquisition_function([SUBJECT, ITEM], None, [], CONTEXT)


if __name__ == "__main__":
    unittest.main()
