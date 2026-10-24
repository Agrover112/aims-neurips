import math
import unittest

try:
    import numpy as np
    import torch

    from aims_starter.faithful_np import (
        CNPAIRT,
        TNPDAIRT,
        Amortized1PL,
        Amortized2PL,
        FaithfulNPAIRT,
        _episode,
        binary_auc,
        map_theta_1pl,
        map_theta_2pl,
    )
except ModuleNotFoundError:  # Torch is an experiment/Modal dependency, not a submission dependency.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class CanonicalNotationModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x_C = torch.randn(3, 8)
        self.y_C = torch.tensor([0.0, 1.0, 1.0])
        self.x_T = torch.randn(6, 8)
        self.y_T = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    def test_np_family_contract_and_shapes(self):
        for variant in ("np", "anp", "stacked_mhsa_anp"):
            with self.subTest(variant=variant):
                model = FaithfulNPAIRT(8, variant=variant, d_model=16, d_z=4, n_heads=4)
                loss, diagnostics = model.training_loss(
                    self.x_C,
                    self.y_C,
                    self.x_T,
                    self.y_T,
                    n_z_samples=2,
                )
                p_y_T, var_p_y_T = model.predict(self.x_C, self.y_C, self.x_T, n_z_samples=3)
                self.assertEqual(loss.ndim, 0)
                self.assertEqual(set(diagnostics), {"nll", "nll_C", "nll_T", "kl", "ability_aux"})
                self.assertEqual(tuple(p_y_T.shape), (6,))
                self.assertEqual(tuple(var_p_y_T.shape), (6,))
                self.assertTrue(torch.isfinite(p_y_T).all())

    def test_item_encoding_excludes_subject_block(self):
        model = FaithfulNPAIRT(
            8,
            variant="np",
            d_model=16,
            d_z=4,
            n_heads=4,
            subject_feature_slice=(3, 5),
        )
        changed_subject = self.x_T.clone()
        changed_subject[:, 3:5] += 100.0
        encoded = model.encode_item_x(self.x_T)
        changed_encoded = model.encode_item_x(changed_subject)
        self.assertTrue(torch.allclose(encoded, changed_encoded))
        self.assertTrue(torch.allclose(model.encode_x(self.x_T), model.encode_x(changed_subject)))
        changed_context = self.x_C.clone()
        changed_context[:, 3:5] += 100.0
        z = torch.randn(2, 4)
        logits = model.decode_logits(
            z,
            model.encode_item_x(self.x_T),
            None,
            model.map_theta(model.encode_item_x(self.x_C), self.y_C),
        )
        changed_logits = model.decode_logits(
            z,
            model.encode_item_x(changed_subject),
            None,
            model.map_theta(model.encode_item_x(changed_context), self.y_C),
        )
        self.assertTrue(torch.allclose(logits, changed_logits, atol=1e-6))

    def test_budget_zero_context(self):
        model = FaithfulNPAIRT(8, variant="anp", d_model=16, d_z=4, n_heads=4)
        p_y_T, var_p_y_T = model.predict(self.x_C[:0], self.y_C[:0], self.x_T, n_z_samples=3)
        self.assertEqual(tuple(p_y_T.shape), (6,))
        self.assertEqual(tuple(var_p_y_T.shape), (6,))

    def test_per_item_latent_decoder_has_more_than_rank_one(self):
        scalar = FaithfulNPAIRT(8, variant="np", d_model=16, d_z=4, n_heads=4)
        per_item = FaithfulNPAIRT(
            8,
            variant="np",
            d_model=16,
            d_z=4,
            n_heads=4,
            latent_item_parameters=True,
        )
        z = torch.randn(4, requires_grad=True)
        scalar_s = scalar.decoder_jacobian_singular_values(self.x_T, z)
        per_item_s = per_item.decoder_jacobian_singular_values(self.x_T, z)
        self.assertLessEqual(int((scalar_s > 1e-5).sum()), 1)
        self.assertGreater(int((per_item_s > 1e-5).sum()), 1)

    def test_anp_residual_path_carries_r_star_into_logits(self):
        torch.manual_seed(3)
        model = FaithfulNPAIRT(
            8,
            variant="anp",
            d_model=16,
            d_z=4,
            n_heads=4,
            residual_scale=0.1,
        )
        x_C_encoded = model.encode_item_x(self.x_C)
        x_T_encoded = model.encode_item_x(self.x_T)
        z = torch.randn(2, 4)
        anchor = model.map_theta(x_C_encoded, self.y_C)
        first = model.r_star(x_C_encoded, self.y_C, x_T_encoded)
        second = model.r_star(model.encode_item_x(self.x_T[:3]), self.y_T[:3], x_T_encoded)
        logits_first = model.decode_logits(z, x_T_encoded, first, anchor)
        logits_second = model.decode_logits(z, x_T_encoded, second, anchor)
        self.assertFalse(torch.allclose(logits_first, logits_second))

        without_residual = FaithfulNPAIRT(8, variant="anp", d_model=16, d_z=4, n_heads=4)
        logits_plain = without_residual.decode_logits(z, x_T_encoded, first, anchor)
        logits_changed = without_residual.decode_logits(z, x_T_encoded, second, anchor)
        self.assertTrue(torch.allclose(logits_plain, logits_changed))

    def test_irt_and_tnp_contracts(self):
        airt = Amortized2PL(8, 2, d_model=16)
        logits = airt.logits(self.x_T, torch.zeros(6, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (6,))

        one_pl = Amortized1PL(8, 2, d_model=16)
        one_pl_logits = one_pl.logits(self.x_T, torch.zeros(6, dtype=torch.long))
        self.assertEqual(tuple(one_pl_logits.shape), (6,))
        discrimination, difficulty = one_pl.item_parameters(self.x_T)
        self.assertTrue(torch.allclose(discrimination, torch.ones_like(discrimination)))
        self.assertEqual(tuple(difficulty.shape), (6,))

        tnp = TNPDAIRT(8, d_model=16, n_heads=4, n_layers=1)
        loss = tnp.training_loss(self.x_C, self.y_C, self.x_T, self.y_T)
        p_y_T = tnp.predict(self.x_C, self.y_C, self.x_T)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(p_y_T.shape), (6,))

    def test_cnp_delta_can_be_disabled_without_changing_the_anchor(self):
        model = CNPAIRT(8, d_model=16)
        full = model.logits(self.x_C, self.y_C, self.x_T)
        without_delta = model.logits(self.x_C, self.y_C, self.x_T, delta_scale=0.0)
        self.assertFalse(torch.allclose(full, without_delta))
        x_C_encoded = model.encode_item_x(self.x_C)
        theta = model.map_theta(x_C_encoded, self.y_C)
        raw = model.irt_parameter_net(model.encode_item_x(self.x_T))
        a_T = torch.nn.functional.softplus(raw[:, 0]) + 0.05
        b_T = raw[:, 1]
        self.assertTrue(torch.allclose(without_delta, a_T * (theta - b_T)))

        disabled = CNPAIRT(8, d_model=16, use_theta_delta=False)
        self.assertTrue(
            torch.allclose(
                disabled.logits(self.x_C, self.y_C, self.x_T),
                disabled.logits(self.x_C, self.y_C, self.x_T, delta_scale=0.0),
            )
        )
        self.assertEqual(float(model.delta_theta(model.encode_item_x(self.x_T[:0]), self.y_C[:0])), 0.0)

    def test_all_irt_item_parameters_exclude_subject_features(self):
        changed_subject = self.x_T.clone()
        changed_subject[:, 3:5] += 100.0

        airt = Amortized2PL(8, 2, d_model=16, subject_feature_slice=(3, 5))
        a, b = airt.item_parameters(self.x_T)
        changed_a, changed_b = airt.item_parameters(changed_subject)
        self.assertTrue(torch.allclose(a, changed_a))
        self.assertTrue(torch.allclose(b, changed_b))
        task_indices = torch.zeros(len(self.x_T), dtype=torch.long)
        self.assertTrue(
            torch.allclose(
                airt.logits(self.x_T, task_indices),
                airt.logits(changed_subject, task_indices),
            )
        )

        for model in (
            CNPAIRT(8, d_model=16, subject_feature_slice=(3, 5)),
            TNPDAIRT(
                8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                subject_feature_slice=(3, 5),
            ),
        ):
            encoded = model.encode_item_x(self.x_T)
            changed_encoded = model.encode_item_x(changed_subject)
            self.assertTrue(torch.allclose(encoded, changed_encoded))
            logits = model.logits(self.x_C, self.y_C, self.x_T)
            changed_context = self.x_C.clone()
            changed_context[:, 3:5] += 100.0
            changed_logits = model.logits(changed_context, self.y_C, changed_subject)
            self.assertTrue(torch.allclose(logits, changed_logits, atol=1e-6))


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class BinaryAucTests(unittest.TestCase):
    """Direct coverage for the dependency-free pairwise ROC-AUC helper."""

    def test_perfect_ranking_is_one(self):
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        probability = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(binary_auc(labels, probability), 1.0, places=7)

    def test_inverted_ranking_is_zero(self):
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        probability = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(binary_auc(labels, probability), 0.0, places=7)

    def test_single_class_returns_nan(self):
        probability = np.array([0.2, 0.7, 0.9])
        self.assertTrue(math.isnan(binary_auc(np.ones(3), probability)))
        self.assertTrue(math.isnan(binary_auc(np.zeros(3), probability)))

    def test_ties_count_as_half_credit(self):
        # One positive and one negative share a score, so the single pair is a tie.
        labels = np.array([1.0, 0.0])
        probability = np.array([0.5, 0.5])
        self.assertAlmostEqual(binary_auc(labels, probability), 0.5, places=7)

    def test_partial_ties_are_averaged(self):
        # Pairs: (0.9,0.4) win, (0.9,0.9) tie, (0.5,0.4) win, (0.5,0.9) loss.
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        probability = np.array([0.9, 0.5, 0.4, 0.9])
        self.assertAlmostEqual(binary_auc(labels, probability), 0.625, places=7)

    def test_matches_a_brute_force_pair_count(self):
        rng = np.random.default_rng(101)
        labels = (rng.random(40) < 0.5).astype(float)
        probability = rng.random(40).round(2)

        wins = 0.0
        total = 0
        for positive in probability[labels == 1.0]:
            for negative in probability[labels == 0.0]:
                total += 1
                if positive > negative:
                    wins += 1.0
                elif positive == negative:
                    wins += 0.5

        self.assertAlmostEqual(binary_auc(labels, probability), wins / total, places=7)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class MapThetaTests(unittest.TestCase):
    """The ability MAP solve is reimplemented in several classes."""

    def setUp(self):
        torch.manual_seed(5)
        self.x_dim = 12
        self.x_C = torch.randn(9, self.x_dim)

    def test_empty_context_returns_the_prior_mean(self):
        empty = torch.zeros(0, self.x_dim)
        empty_labels = torch.zeros(0)

        cnp = CNPAIRT(self.x_dim)
        encoded = cnp.encode_item_x(empty)
        self.assertAlmostEqual(float(cnp.map_theta(encoded, empty_labels)), 0.0, places=7)
        self.assertAlmostEqual(
            float(cnp.map_theta(encoded, empty_labels, prior_mean=0.75)), 0.75, places=7
        )

    def test_free_function_map_theta_handles_an_empty_context(self):
        model_2pl = Amortized2PL(self.x_dim, 4)
        model_1pl = Amortized1PL(self.x_dim, 4)
        empty = torch.zeros(0, self.x_dim)
        empty_labels = torch.zeros(0)

        self.assertAlmostEqual(
            float(map_theta_2pl(model_2pl, empty, empty_labels, prior_mean=0.4)), 0.4, places=7
        )
        self.assertAlmostEqual(
            float(map_theta_1pl(model_1pl, empty, empty_labels, prior_mean=0.4)), 0.4, places=7
        )

    def test_map_theta_is_monotone_in_the_context_labels(self):
        cnp = CNPAIRT(self.x_dim)
        encoded = cnp.encode_item_x(self.x_C)

        all_wrong = float(cnp.map_theta(encoded, torch.zeros(9)))
        mixed = float(cnp.map_theta(encoded, torch.tensor([1.0, 0.0] * 4 + [1.0])))
        all_right = float(cnp.map_theta(encoded, torch.ones(9)))

        self.assertLess(all_wrong, mixed)
        self.assertLess(mixed, all_right)

    def test_free_function_map_theta_is_monotone(self):
        model = Amortized2PL(self.x_dim, 4)
        all_wrong = float(map_theta_2pl(model, self.x_C, torch.zeros(9)))
        all_right = float(map_theta_2pl(model, self.x_C, torch.ones(9)))
        self.assertLess(all_wrong, all_right)

    def test_map_theta_stays_within_the_clamp(self):
        cnp = CNPAIRT(self.x_dim)
        encoded = cnp.encode_item_x(torch.randn(64, self.x_dim))
        for labels in (torch.zeros(64), torch.ones(64)):
            theta = float(cnp.map_theta(encoded, labels))
            self.assertGreaterEqual(theta, -4.0)
            self.assertLessEqual(theta, 4.0)

    def test_the_three_newton_implementations_agree(self):
        """CNPAIRT, FaithfulNPAIRT, and TNPDAIRT each carry their own solve.

        Feeding one shared set of item parameters and labels through all three
        must give the same ability. This fails loudly if one copy is edited
        without the others.
        """

        torch.manual_seed(21)
        cnp = CNPAIRT(self.x_dim)
        faithful = FaithfulNPAIRT(self.x_dim, variant="np")
        tnpd = TNPDAIRT(self.x_dim)

        # Share one item network so all three solve the identical problem.
        faithful.irt_parameter_net.load_state_dict(cnp.irt_parameter_net.state_dict())
        tnpd.irt_parameter_net.load_state_dict(cnp.irt_parameter_net.state_dict())

        encoded = cnp.encode_item_x(self.x_C)
        y_C = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])

        cnp_theta = float(cnp.map_theta(encoded, y_C))
        faithful_theta = float(faithful.map_theta(encoded, y_C))
        tnpd_theta = float(tnpd.map_theta(encoded, y_C))

        self.assertAlmostEqual(cnp_theta, faithful_theta, delta=1e-3)
        self.assertAlmostEqual(cnp_theta, tnpd_theta, delta=1e-3)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class EpisodeSamplingTests(unittest.TestCase):
    """``faithful_np._episode`` is distinct from ``meta_training.sample_episode``."""

    def setUp(self):
        self.task = {
            "x": np.random.default_rng(0).normal(size=(120, 6)).astype("float32"),
            "y": (np.random.default_rng(1).random(120) < 0.5).astype("float32"),
        }
        self.device = torch.device("cpu")

    def test_disjoint_flag_keeps_context_out_of_the_targets(self):
        rng = np.random.default_rng(4)
        for _ in range(25):
            x_C, _, x_T, _ = _episode(
                self.task, rng, self.device, disjoint_context_target=True
            )
            if not len(x_C):
                continue
            context = {tuple(row.tolist()) for row in x_C}
            targets = {tuple(row.tolist()) for row in x_T}
            self.assertEqual(context & targets, set())

    def test_default_includes_the_context_in_the_targets(self):
        rng = np.random.default_rng(4)
        found_nonempty = False
        for _ in range(25):
            x_C, _, x_T, _ = _episode(self.task, rng, self.device)
            if not len(x_C):
                continue
            found_nonempty = True
            context = {tuple(row.tolist()) for row in x_C}
            targets = {tuple(row.tolist()) for row in x_T}
            self.assertTrue(context <= targets)
        self.assertTrue(found_nonempty)

    def test_context_never_consumes_the_whole_task(self):
        rng = np.random.default_rng(9)
        for _ in range(25):
            x_C, _, _, _ = _episode(self.task, rng, self.device)
            self.assertLessEqual(len(x_C), len(self.task["y"]) - 1)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class GradientFlowTests(unittest.TestCase):
    def test_training_loss_produces_finite_gradients(self):
        torch.manual_seed(13)
        model = FaithfulNPAIRT(8, variant="anp")
        x_C = torch.randn(5, 8)
        y_C = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        x_T = torch.randn(12, 8)
        y_T = (torch.rand(12) < 0.5).float()

        loss, _ = model.training_loss(x_C, y_C, x_T, y_T, n_z_samples=2)
        loss.backward()

        with_gradient = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None and torch.any(parameter.grad != 0.0)
        ]
        self.assertTrue(with_gradient)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
