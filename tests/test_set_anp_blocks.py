"""Contracts for the one-block Set-ANP ablation."""

import unittest

import torch

from organized.training.set_anp_blocks import SetAttentiveNeuralProcess


class SetAttentiveNeuralProcessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.x_context = torch.randn(7, 16)
        self.y_context = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        self.x_target = torch.randn(5, 16)
        self.y_target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    def _model(self):
        return SetAttentiveNeuralProcess(
            16, width=32, self_attention_blocks=1,
            cross_attention_heads=4, n_inducing=4,
            use_latent=True, latent_dim=12,
        )

    def test_context_order_does_not_change_logits(self):
        model = self._model().eval()
        permutation = torch.tensor([5, 2, 0, 6, 1, 4, 3])
        first = model.logits(self.x_context, self.y_context, self.x_target)
        second = model.logits(
            self.x_context[permutation], self.y_context[permutation], self.x_target
        )
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_target_order_only_reorders_logits(self):
        model = self._model().eval()
        permutation = torch.tensor([3, 1, 4, 0, 2])
        original = model.logits(self.x_context, self.y_context, self.x_target)
        reordered = model.logits(
            self.x_context, self.y_context, self.x_target[permutation]
        )
        torch.testing.assert_close(reordered, original[permutation], rtol=1e-5, atol=1e-6)

    def test_empty_context_and_elbo_are_finite(self):
        model = self._model()
        logits = model.logits(self.x_context[:0], self.y_context[:0], self.x_target)
        loss, terms = model.loss(
            self.x_context, self.y_context, self.x_target, self.y_target
        )
        self.assertEqual(logits.shape, (5,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(terms["kl"], 0.0)

    def test_context_labels_affect_predictions(self):
        model = self._model().eval()
        first = model.logits(self.x_context, self.y_context, self.x_target)
        flipped = model.logits(self.x_context, 1.0 - self.y_context, self.x_target)
        self.assertGreater(float((first - flipped).abs().max()), 1e-5)

    def test_invalid_inducing_count_fails_early(self):
        with self.assertRaises(ValueError):
            SetAttentiveNeuralProcess(16, n_inducing=0)


if __name__ == "__main__":
    unittest.main()
