import unittest

try:
    import torch

    from aims_starter.faithful_np import TNPDAIRT, Amortized2PL, FaithfulNPAIRT
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
                model = FaithfulNPAIRT(
                    8, variant=variant, d_model=16, d_z=4, n_heads=4
                )
                loss, diagnostics = model.training_loss(
                    self.x_C,
                    self.y_C,
                    self.x_T,
                    self.y_T,
                    n_z_samples=2,
                )
                p_y_T, var_p_y_T = model.predict(
                    self.x_C, self.y_C, self.x_T, n_z_samples=3
                )
                self.assertEqual(loss.ndim, 0)
                self.assertEqual(
                    set(diagnostics), {"nll", "nll_C", "nll_T", "kl"}
                )
                self.assertEqual(tuple(p_y_T.shape), (6,))
                self.assertEqual(tuple(var_p_y_T.shape), (6,))
                self.assertTrue(torch.isfinite(p_y_T).all())

    def test_budget_zero_context(self):
        model = FaithfulNPAIRT(8, variant="anp", d_model=16, d_z=4, n_heads=4)
        p_y_T, var_p_y_T = model.predict(
            self.x_C[:0], self.y_C[:0], self.x_T, n_z_samples=3
        )
        self.assertEqual(tuple(p_y_T.shape), (6,))
        self.assertEqual(tuple(var_p_y_T.shape), (6,))

    def test_irt_and_tnp_contracts(self):
        airt = Amortized2PL(8, 2, d_model=16)
        logits = airt.logits(self.x_T, torch.zeros(6, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (6,))

        tnp = TNPDAIRT(8, d_model=16, n_heads=4, n_layers=1)
        loss = tnp.training_loss(self.x_C, self.y_C, self.x_T, self.y_T)
        p_y_T = tnp.predict(self.x_C, self.y_C, self.x_T)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(p_y_T.shape), (6,))


if __name__ == "__main__":
    unittest.main()
