import unittest

from labeling import acquisition_function


class LabelingTests(unittest.TestCase):
    def test_forces_purchase_when_candidates_equal_remaining_budget(self):
        context = {"labels_remaining": 2, "items_remaining": 2}
        self.assertIs(acquisition_function([{}, {"item_content": "x"}], 0.9, [], context), True)


    def test_does_not_purchase_after_budget_exhausted(self):
        context = {"labels_remaining": 0, "items_remaining": 4}
        self.assertIs(acquisition_function([{}, {"item_content": "x"}], 0.5, [], context), False)
