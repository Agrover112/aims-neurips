import unittest
import numpy as np
from organized.training import statistical_baselines as core

class StatisticalBaselineTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(120, 4))
        tasks = []
        for j in range(6):
            items = np.arange(80) if j < 4 else np.arange(40, 120)
            p = 1 / (1 + np.exp(-(features[items, 0] + (j-2)*.3)))
            tasks.append({"items": items, "labels": rng.binomial(1, p), "role": "train" if j < 4 else "validation",
                          "subject": str(j), "benchmark": "train" if j < 4 else "validation"})
        return features, tasks

    def test_nested_cells_train_only(self):
        _, tasks = self.fixture()
        small = core.training_cells(tasks, .25, 42)
        full = core.training_cells(tasks, 1., 42)
        self.assertTrue(set(small[0]) <= {0,1,2,3})
        self.assertTrue(set(zip(small[0],small[1])) <= set(zip(full[0],full[1])))
        self.assertEqual(len(full[0]), 320)

    def test_features_never_fit_on_heldout_items(self):
        features, _ = self.fixture()
        train_items = np.arange(80)
        phi, metadata = core.feature_map(features, train_items, "bayesian_rbf", 42, kernel_dim=8)
        changed = features.copy()
        changed[80:] *= 1000
        phi2, metadata2 = core.feature_map(changed, train_items, "bayesian_rbf", 42, kernel_dim=8)
        np.testing.assert_array_equal(metadata["mean"], metadata2["mean"])
        np.testing.assert_array_equal(phi[:80], phi2[:80])

    def test_five_models_produce_probabilities_and_condition_locally(self):
        features, tasks = self.fixture()
        cells = core.training_cells(tasks, 1., 42)
        for kind in core.MODELS:
            phi = None if kind in ("beta_binomial", "rasch") else core.feature_map(features, np.arange(80), kind, 42, kernel_dim=8)[0]
            state = core.fit(kind, cells, phi, len(tasks), len(features), iterations=8)
            for ci, cy in ((np.array([], dtype=int), np.array([])), (np.array([81,82,83]), np.array([0,1,1]))):
                p = core.predict(state, ci, cy, np.array([90,91,92]), phi)
                self.assertTrue(np.isfinite(p).all(), kind)
                self.assertTrue(((p >= 0) & (p <= 1)).all(), kind)
            before = core.predict(state, np.array([81]), np.array([0]), np.array([90,91]), phi)
            core.predict(state, np.array([81]), np.array([1]), np.array([90,91]), phi)
            after = core.predict(state, np.array([81]), np.array([0]), np.array([90,91]), phi)
            np.testing.assert_array_equal(before, after)

    def test_rasch_unseen_items_are_exchangeable(self):
        _, tasks = self.fixture()
        state = core.fit("rasch", core.training_cells(tasks, 1., 42), None, 6, 120, iterations=8)
        p = core.predict(state, np.array([81,82]), np.array([1,0]), np.array([90,91,92]), None)
        np.testing.assert_allclose(p, p[0])

    def test_schur_solution_matches_joint_gaussian_system(self):
        features, tasks = self.fixture()
        phi, _ = core.feature_map(features, np.arange(80), "bayesian_rbf", 42, kernel_dim=4)
        cells = core.training_cells(tasks, .1, 42)
        state = core.fit("bayesian_rbf", cells, phi, len(tasks), len(features))
        task, item, y = cells
        active = sorted(set(task))
        design = np.column_stack([phi[item], np.eye(len(tasks))[task][:,active]])
        penalty = np.ones(design.shape[1])
        penalty[0] = 1e-8
        penalty[phi.shape[1]:] = 4.
        expected = np.linalg.solve(design.T @ design / .25 + np.diag(penalty), design.T @ y / .25)
        np.testing.assert_allclose(state["w"], expected[:phi.shape[1]], rtol=1e-8, atol=1e-8)

    def test_linear_fit_matches_ordinary_least_squares(self):
        features, tasks = self.fixture()
        phi, _ = core.feature_map(features, np.arange(80), "linear_regression", 42)
        cells = core.training_cells(tasks, .5, 42)
        state = core.fit("linear_regression", cells, phi, len(tasks), len(features))
        task, item, y = cells
        n = np.bincount(task, minlength=len(tasks))
        s = np.bincount(task, weights=y, minlength=len(tasks))
        rate = (2+s[task]-y)/(3+n[task])
        design = np.column_stack([phi[item], rate])
        expected = np.linalg.lstsq(design, y, rcond=None)[0]
        np.testing.assert_allclose(np.r_[state["w"], state["context_coefficient"]], expected, atol=1e-8)

    def test_metric_names_and_single_class_auc(self):
        _, tasks = self.fixture()
        task = tasks[0]
        bank = [{"task": 0, "acquisition": np.arange(40), "targets": np.arange(40,80), "weight": 1.}]
        state = {"kind": "beta_binomial", "alpha": 2., "beta": 2.}
        result = core.diagnostics(state, tasks, bank, None, "training_subjects")
        for suffix in ("same_subject_split_half/brier_score", "same_subject_split_half/roc_auc", "same_subject_response_holdout/roc_auc"):
            self.assertIn("diagnostic/training_subjects/" + suffix, result)
        self.assertIsNone(core.metrics(np.zeros(4), np.full(4,.5))["roc_auc"])
