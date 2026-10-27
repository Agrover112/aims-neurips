import unittest

import torch

from organized.training.anp_crossattn_blocks import AttentiveNeuralProcess


class AttentiveNeuralProcessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x_context = torch.randn(5, 16)
        self.y_context = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        self.x_target = torch.randn(4, 16)
        self.y_target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    def test_target_logits_are_context_order_invariant(self):
        model = AttentiveNeuralProcess(
            16, width=32, self_attention_blocks=2,
            cross_attention_heads=4, use_latent=False,
        ).eval()
        permutation = torch.tensor([3, 0, 4, 1, 2])
        first = model.logits(self.x_context, self.y_context, self.x_target)
        second = model.logits(
            self.x_context[permutation], self.y_context[permutation], self.x_target
        )
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_empty_context_and_latent_loss_are_finite(self):
        model = AttentiveNeuralProcess(
            16, width=32, self_attention_blocks=1,
            cross_attention_heads=4, use_latent=True,
        )
        empty_logits = model.logits(
            self.x_context[:0], self.y_context[:0], self.x_target
        )
        loss, terms = model.loss(
            self.x_context, self.y_context, self.x_target, self.y_target
        )
        self.assertEqual(empty_logits.shape, (4,))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.tensor(terms["nll"])))
        self.assertTrue(torch.isfinite(torch.tensor(terms["kl"])))

    def test_width_must_divide_across_attention_heads(self):
        with self.assertRaises(ValueError):
            AttentiveNeuralProcess(16, width=30, cross_attention_heads=4)


if __name__ == "__main__":
    unittest.main()
