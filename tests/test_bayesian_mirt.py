"""Unit tests for the Bayesian multidimensional IRT model.

``modal_bayesian_mirt.py`` has never been run end to end -- the only numbers
attached to it in the README sit in the table explicitly labelled fabricated --
so these tests pin the pure pieces before any result is produced from them.

They cover the parts where a silent error would survive a full HMC fit and
still produce plausible output: the pack/unpack round trip that every sampler
step depends on, the convergence diagnostic that decides whether a fit is
trusted, the quadrature used to integrate the posterior, and the PCA's
training-only contract.

Nothing here touches Modal, a GPU, or the corpus.
"""

import unittest

import numpy as np

try:
    import torch

    import modal_bayesian_mirt as mirt
except ModuleNotFoundError:  # torch and modal are experiment-only dependencies
    torch = None
    mirt = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class LogitTests(unittest.TestCase):
    def test_logit_of_one_half_is_zero(self):
        self.assertAlmostEqual(float(mirt._logit(0.5)), 0.0, places=9)

    def test_logit_is_the_inverse_of_sigmoid(self):
        probabilities = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
        recovered = torch.sigmoid(torch.as_tensor(mirt._logit(probabilities)))
        np.testing.assert_allclose(recovered.numpy(), probabilities, atol=1e-6)

    def test_logit_clips_rather_than_returning_infinity(self):
        for probability in (0.0, 1.0):
            value = float(mirt._logit(probability))
            self.assertTrue(np.isfinite(value))

    def test_logit_is_monotone(self):
        values = mirt._logit(np.array([0.1, 0.3, 0.6, 0.9]))
        self.assertTrue(np.all(np.diff(values) > 0))


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class PackUnpackTests(unittest.TestCase):
    """Every HMC step packs the parameter blocks into one flat vector and
    unpacks them again; a mismatch there corrupts the fit silently."""

    def setUp(self):
        torch.manual_seed(3)
        self.shapes = [(4, 2), (6, 2), (6, 1), (1,)]
        self.blocks = [torch.randn(shape) for shape in self.shapes]

    def test_pack_produces_a_flat_vector_of_the_right_length(self):
        packed = mirt._pack(self.blocks)
        self.assertEqual(packed.dim(), 1)
        self.assertEqual(packed.numel(), sum(int(np.prod(s)) for s in self.shapes))

    def test_round_trip_is_exact(self):
        packed = mirt._pack(self.blocks)
        restored = mirt._unpack(packed, self.shapes, torch.device("cpu"))
        for original, copy in zip(self.blocks, restored, strict=True):
            self.assertTrue(torch.equal(original, copy))

    def test_block_order_is_preserved(self):
        blocks = [torch.full(shape, float(index)) for index, shape in enumerate(self.shapes)]
        restored = mirt._unpack(mirt._pack(blocks), self.shapes, torch.device("cpu"))
        for index, block in enumerate(restored):
            self.assertTrue(torch.all(block == float(index)))

    def test_unpacked_blocks_have_the_requested_shapes(self):
        restored = mirt._unpack(mirt._pack(self.blocks), self.shapes, torch.device("cpu"))
        self.assertEqual([tuple(b.shape) for b in restored], self.shapes)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class SplitRhatTests(unittest.TestCase):
    """R-hat decides whether a posterior is trusted. A version that cannot
    report non-convergence is worse than none at all."""

    def test_identical_chains_give_rhat_near_one(self):
        torch.manual_seed(11)
        draws = torch.randn(1, 400, 3).repeat(4, 1, 1)
        rhat = mirt._split_rhat(draws)
        self.assertTrue(torch.all(rhat < 1.05), f"rhat = {rhat}")

    def test_well_mixed_independent_chains_give_rhat_near_one(self):
        torch.manual_seed(12)
        rhat = mirt._split_rhat(torch.randn(4, 600, 3))
        self.assertTrue(torch.all(rhat < 1.1), f"rhat = {rhat}")

    def test_chains_stuck_at_different_locations_give_large_rhat(self):
        torch.manual_seed(13)
        offsets = torch.tensor([-8.0, -3.0, 3.0, 8.0]).view(4, 1, 1)
        chains = torch.randn(4, 400, 2) * 0.05 + offsets
        rhat = mirt._split_rhat(chains)
        self.assertTrue(torch.all(rhat > 1.5), f"rhat = {rhat}")

    def test_rhat_has_one_entry_per_parameter(self):
        self.assertEqual(mirt._split_rhat(torch.randn(2, 100, 7)).shape, (7,))

    def test_rhat_is_finite_for_a_constant_chain(self):
        # Zero within-chain variance must be clamped, not divided by.
        rhat = mirt._split_rhat(torch.zeros(4, 200, 2))
        self.assertTrue(torch.isfinite(rhat).all())


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class SobolNormalTests(unittest.TestCase):
    """The predictive integral is a quadrature over these draws, so their
    shape, moments, and reproducibility all feed straight into the forecast."""

    def test_shape_matches_the_request(self):
        self.assertEqual(tuple(mirt._sobol_normal(64, 3, seed=5).shape), (64, 3))

    def test_draws_are_approximately_standard_normal(self):
        draws = mirt._sobol_normal(4096, 2, seed=7).numpy()
        np.testing.assert_allclose(draws.mean(axis=0), np.zeros(2), atol=0.1)
        np.testing.assert_allclose(draws.std(axis=0), np.ones(2), atol=0.1)

    def test_same_seed_reproduces_the_same_draws(self):
        first = mirt._sobol_normal(32, 2, seed=9)
        second = mirt._sobol_normal(32, 2, seed=9)
        self.assertTrue(torch.equal(first, second))

    def test_different_seeds_give_different_draws(self):
        first = mirt._sobol_normal(32, 2, seed=1)
        second = mirt._sobol_normal(32, 2, seed=2)
        self.assertFalse(torch.equal(first, second))

    def test_draws_are_finite(self):
        self.assertTrue(torch.isfinite(mirt._sobol_normal(256, 4, seed=3)).all())


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class LogProbTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.n_tasks, self.n_features, self.dim = 3, 5, 2
        self.theta = torch.randn(self.n_tasks, self.dim)
        self.W_alpha = torch.randn(self.n_features, self.dim) * 0.3
        self.w_beta = torch.randn(self.n_features) * 0.3
        self.mu = torch.zeros(1)
        self.E_rows = torch.randn(12, self.n_features)
        self.task_index = torch.randint(0, self.n_tasks, (12,))
        self.labels = (torch.rand(12) < 0.5).float()
        self.priors = {
            "prior_theta": 1.0,
            "prior_alpha": 1.0,
            "prior_beta": 1.0,
            "prior_mu": 2.0,
        }

    def _log_prob(self, **overrides):
        kwargs = {
            "theta": self.theta,
            "W_alpha": self.W_alpha,
            "w_beta": self.w_beta,
            "mu": self.mu,
            **overrides,
        }
        return mirt._log_prob_blocks(
            kwargs["theta"],
            kwargs["W_alpha"],
            kwargs["w_beta"],
            kwargs["mu"],
            self.E_rows,
            self.task_index,
            self.labels,
            **self.priors,
        )

    def test_log_probability_is_finite_and_negative(self):
        value = self._log_prob()
        self.assertTrue(torch.isfinite(value))
        self.assertLess(float(value), 0.0)

    def test_moving_parameters_far_from_the_prior_lowers_the_density(self):
        far = self._log_prob(theta=self.theta + 25.0)
        self.assertLess(float(far), float(self._log_prob()))

    def test_gradients_flow_to_every_block(self):
        theta = self.theta.clone().requires_grad_(True)
        W_alpha = self.W_alpha.clone().requires_grad_(True)
        w_beta = self.w_beta.clone().requires_grad_(True)
        mu = self.mu.clone().requires_grad_(True)
        self._log_prob(theta=theta, W_alpha=W_alpha, w_beta=w_beta, mu=mu).backward()
        for block in (theta, W_alpha, w_beta, mu):
            self.assertIsNotNone(block.grad)
            self.assertTrue(torch.isfinite(block.grad).all())

    def test_perfectly_fitted_labels_score_higher_than_inverted_ones(self):
        strong = torch.full((self.n_tasks, self.dim), 4.0)
        alpha = self.E_rows @ self.W_alpha
        logits = (alpha * strong[self.task_index]).sum(dim=1)
        agreeing = (logits > 0).float()
        kwargs = dict(self.priors)
        good = mirt._log_prob_blocks(
            strong, self.W_alpha, self.w_beta, self.mu,
            self.E_rows, self.task_index, agreeing, **kwargs,
        )
        bad = mirt._log_prob_blocks(
            strong, self.W_alpha, self.w_beta, self.mu,
            self.E_rows, self.task_index, 1.0 - agreeing, **kwargs,
        )
        self.assertGreater(float(good), float(bad))


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class DiagonalMassTests(unittest.TestCase):
    """The HMC preconditioner. A non-positive entry makes the step size
    nonsensical, and the sampler would keep running regardless."""

    def setUp(self):
        torch.manual_seed(23)
        self.dim, self.n_features, self.n_tasks = 2, 4, 3
        self.theta = torch.randn(self.n_tasks, self.dim)
        self.W_alpha = torch.randn(self.n_features, self.dim)
        self.w_beta = torch.randn(self.n_features)
        self.mu = torch.zeros(1)
        self.E_rows = torch.randn(10, self.n_features)
        self.task_index = torch.randint(0, self.n_tasks, (10,))
        self.labels = (torch.rand(10) < 0.5).float()
        self.shapes = [
            (self.n_tasks, self.dim),
            (self.n_features, self.dim),
            (self.n_features,),
            (1,),
        ]
        self.priors = {
            "prior_theta": 1.0,
            "prior_alpha": 1.0,
            "prior_beta": 1.0,
            "prior_mu": 2.0,
        }

    def _mass(self):
        return mirt._diagonal_mass(
            self.theta, self.W_alpha, self.w_beta, self.mu,
            self.E_rows, self.task_index, self.labels, self.shapes, **self.priors,
        )

    def test_mass_is_strictly_positive(self):
        self.assertTrue(torch.all(self._mass() > 0.0))

    def test_mass_is_finite(self):
        self.assertTrue(torch.isfinite(self._mass()).all())

    def test_mass_length_matches_the_packed_parameter_vector(self):
        expected = sum(int(np.prod(shape)) for shape in self.shapes)
        self.assertEqual(self._mass().numel(), expected)

    def test_mass_respects_the_clamp_floor(self):
        self.assertGreaterEqual(float(self._mass().min()), 1e-3)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class PcaTests(unittest.TestCase):
    """The PCA must be fitted on training items only. Fitting it on every item
    leaks evaluation content into the features and would inflate every number
    the model reports."""

    def setUp(self):
        rng = np.random.default_rng(29)
        self.embeddings = rng.normal(size=(120, 16)).astype(np.float32)
        self.train_indices = np.arange(80)
        self.device = torch.device("cpu")

    def _fit(self, embeddings=None, indices=None, dim=3):
        return mirt._fit_pca(
            self.embeddings if embeddings is None else embeddings,
            self.train_indices if indices is None else indices,
            dim,
            self.device,
        )

    def test_output_covers_every_item_not_just_the_training_rows(self):
        features, _ = self._fit()
        self.assertEqual(features.shape[0], self.embeddings.shape[0])

    def test_output_width_is_the_requested_dimension(self):
        features, _ = self._fit(dim=4)
        self.assertEqual(features.shape[1], 4)

    def test_held_out_rows_do_not_affect_the_projection(self):
        # Corrupting rows outside train_item_indices must leave the training
        # rows' features untouched; if it does not, the fit saw them.
        baseline, _ = self._fit()
        corrupted = self.embeddings.copy()
        corrupted[80:] += 50.0
        altered, _ = self._fit(embeddings=corrupted)
        np.testing.assert_allclose(
            baseline[:80].numpy(), altered[:80].numpy(), atol=1e-4
        )

    def test_training_features_are_standardized(self):
        features, _ = self._fit()
        train = features[:80]
        self.assertTrue(torch.all(train.std(dim=0, unbiased=False) < 1.2))

    def test_features_are_clamped_to_five_sigma(self):
        outlier = self.embeddings.copy()
        outlier[100] *= 500.0
        features, _ = self._fit(embeddings=outlier)
        self.assertLessEqual(float(features.abs().max()), 5.0 + 1e-5)

    def test_metadata_records_the_training_item_count(self):
        _, metadata = self._fit()
        self.assertEqual(metadata["pca_fit_items"], len(self.train_indices))
        self.assertEqual(metadata["pca_dim"], 3)

    def test_fit_is_deterministic(self):
        first, _ = self._fit()
        second, _ = self._fit()
        np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-5)

    def test_features_are_finite(self):
        features, _ = self._fit()
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
