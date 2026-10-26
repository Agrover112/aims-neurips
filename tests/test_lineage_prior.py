import unittest

import numpy as np

from aims_starter.lineage_prior import LineagePrior


class LineagePriorTests(unittest.TestCase):
    def setUp(self):
        # provider A: family x is high, family y is low; provider B: family z
        self.families = ["x"] * 10 + ["y"] * 10 + ["z"] * 10
        self.providers = ["A"] * 20 + ["B"] * 10
        base = np.array([[2.0, 0.0]] * 10 + [[-2.0, 0.0]] * 10 + [[0.0, 3.0]] * 10)
        self.thetas = base

    def test_unseen_subject_gets_global_mean(self):
        prior = LineagePrior(strength=0.0).fit(self.families, self.providers, self.thetas)
        np.testing.assert_allclose(prior.predict("", ""), self.thetas.mean(axis=0))
        np.testing.assert_allclose(prior.predict("nope", "nobody"), self.thetas.mean(axis=0))

    def test_family_level_tracks_its_own_mean(self):
        prior = LineagePrior(strength=0.0).fit(self.families, self.providers, self.thetas)
        np.testing.assert_allclose(prior.predict("x", "A"), [2.0, 0.0])
        np.testing.assert_allclose(prior.predict("y", "A"), [-2.0, 0.0])

    def test_unknown_family_falls_back_to_provider(self):
        prior = LineagePrior(strength=0.0).fit(self.families, self.providers, self.thetas)
        np.testing.assert_allclose(prior.predict("new", "A"), [0.0, 0.0])
        self.assertEqual(prior.level("new", "A"), "provider")

    def test_shrinkage_pulls_small_groups_toward_parent(self):
        thetas = np.array([[5.0]] * 1 + [[0.0]] * 9)
        prior = LineagePrior(strength=4.0).fit(["a"] + ["b"] * 9, ["P"] * 10, thetas)
        # family "a" has n=1: (1*5 + 4*provider)/5, provider is itself shrunk
        self.assertLess(prior.predict("a", "P")[0], 5.0)
        self.assertGreater(prior.predict("a", "P")[0], 0.0)

    def test_more_evidence_means_less_shrinkage(self):
        small = LineagePrior(strength=4.0).fit(["a"] * 2 + ["b"] * 40, ["P"] * 42,
                                               np.array([[4.0]] * 2 + [[0.0]] * 40))
        large = LineagePrior(strength=4.0).fit(["a"] * 20 + ["b"] * 22, ["P"] * 42,
                                               np.array([[4.0]] * 20 + [[0.0]] * 22))
        self.assertGreater(large.predict("a", "P")[0], small.predict("a", "P")[0])

    def test_case_and_whitespace_are_ignored(self):
        prior = LineagePrior(strength=0.0).fit(self.families, self.providers, self.thetas)
        np.testing.assert_allclose(prior.predict(" X ", "a"), prior.predict("x", "A"))

    def test_name_without_provider_pools_across_providers(self):
        prior = LineagePrior(strength=0.0).fit(["m", "m"], ["A", "B"],
                                               np.array([[1.0], [3.0]]))
        np.testing.assert_allclose(prior.predict("m", ""), [2.0])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            LineagePrior(strength=-1.0)
        with self.assertRaises(ValueError):
            LineagePrior().fit(["a"], ["P", "Q"], np.zeros((1, 2)))
        with self.assertRaises(RuntimeError):
            LineagePrior().predict("a", "P")

    def test_prediction_is_a_copy(self):
        prior = LineagePrior(strength=0.0).fit(self.families, self.providers, self.thetas)
        prior.predict("x", "A")[:] = 99.0
        np.testing.assert_allclose(prior.predict("x", "A"), [2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
