import unittest

try:
    import torch

    from aims_starter.faithful_np import (
        CNPAIRT,
        TNPDAIRT,
        Amortized1PL,
        Amortized2PL,
        FaithfulNPAIRT,
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


if __name__ == "__main__":
    unittest.main()
