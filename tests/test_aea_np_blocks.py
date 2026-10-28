"""Contracts for the Set-Transformer AEA-NP implementation."""

import unittest

import torch

from organized.training.aea_np_blocks import ActiveEvaluationNeuralProcess


class ActiveEvaluationNeuralProcessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x_context = torch.randn(6, 16)
        self.y_context = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        self.x_target = torch.randn(5, 16)
        self.y_target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    def _model(self):
        return ActiveEvaluationNeuralProcess(
            16, width=32, item_dim=16, label_dim=16,
            encoder_blocks=2, decoder_blocks=2, attention_heads=4,
            n_inducing=4, latent_dim=12,
        )

    def test_context_order_does_not_change_logits(self):
        model = self._model().eval()
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        first = model.logits(self.x_context, self.y_context, self.x_target)
        second = model.logits(
            self.x_context[permutation], self.y_context[permutation], self.x_target
        )
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_target_order_only_reorders_logits(self):
        model = self._model().eval()
        permutation = torch.tensor([4, 1, 0, 3, 2])
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
        self.assertTrue(torch.isfinite(torch.tensor(terms["nll"])))
        self.assertTrue(torch.isfinite(torch.tensor(terms["kl"])))

    def test_invalid_attention_configuration_fails_early(self):
        with self.assertRaises(ValueError):
            ActiveEvaluationNeuralProcess(16, width=30, attention_heads=8)


if __name__ == "__main__":
    unittest.main()
