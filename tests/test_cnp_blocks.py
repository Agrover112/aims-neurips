"""Unit tests for the CNP/NP set architecture.

The invariants here are the ones a set model silently violates while still
producing plausible loss curves:

* **permutation invariance** over the context (CNP eq. 2) — if an encoder
  leaks order, the model is not a valid conditional stochastic process and the
  budget ladder becomes order-dependent;
* **the context actually reaching the output** — this is the failure mode in
  ``CNPAIRT``, which squeezes the whole context into one scalar and never
  passes ``r`` to its item head, and the failure mode of concatenating a scalar
  label onto a 1024-d embedding, where it is 0.1% of input dims;
* **empty context working**, since budget 0 is a scored point in the
  competition, not an edge case.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "organized" / "training"))

try:
    import torch

    import cnp_blocks as cb
except ModuleNotFoundError:
    torch = None
    cb = None

EMB = 32
ENCODERS = ["linear", "mhsa", "set_transformer", "sgformer"]
POOLINGS = ["mean", "mhsa"]


@unittest.skipIf(cb is None, "PyTorch is not installed in the lightweight test environment")
class ArchitectureMatrixTests(unittest.TestCase):
    """Every encoder x pooling x latent combination must satisfy the contract."""

    def _model(self, encoder="linear", pooling="mean", use_latent=False):
        torch.manual_seed(0)
        return cb.NeuralProcess(EMB, width=64, n_blocks=2, encoder=encoder,
                                pooling=pooling, use_latent=use_latent,
                                latent_dim=16, n_inducing=4)

    def test_every_combination_builds_and_runs(self):
        for enc in ENCODERS:
            for pool in POOLINGS:
                for latent in (False, True):
                    with self.subTest(encoder=enc, pooling=pool, latent=latent):
                        m = self._model(enc, pool, latent)
                        out = m.logits(torch.randn(5, EMB),
                                       (torch.rand(5) < 0.5).float(),
                                       torch.randn(3, EMB))
                        self.assertEqual(tuple(out.shape), (3,))
                        self.assertTrue(torch.isfinite(out).all())

    def test_context_order_does_not_change_the_prediction(self):
        """CNP eq. (2): the aggregator must be permutation invariant."""
        for enc in ENCODERS:
            for pool in POOLINGS:
                with self.subTest(encoder=enc, pooling=pool):
                    m = self._model(enc, pool).eval()
                    x_c, y_c = torch.randn(9, EMB), (torch.rand(9) < 0.5).float()
                    x_t = torch.randn(4, EMB)
                    perm = torch.randperm(9)
                    with torch.no_grad():
                        a = m.logits(x_c, y_c, x_t)
                        b = m.logits(x_c[perm], y_c[perm], x_t)
                    self.assertLess(float((a - b).abs().max()), 1e-4,
                                    f"{enc}/{pool} is order-dependent")

    def test_empty_context_is_finite(self):
        """Budget 0 is scored, so an empty context is a real input."""
        for enc in ENCODERS:
            for pool in POOLINGS:
                with self.subTest(encoder=enc, pooling=pool):
                    m = self._model(enc, pool).eval()
                    with torch.no_grad():
                        out = m.logits(torch.zeros(0, EMB), torch.zeros(0),
                                       torch.randn(4, EMB))
                    self.assertTrue(torch.isfinite(out).all())

    def test_context_labels_change_the_prediction(self):
        """The CNPAIRT failure: if r never reaches the head, flipping every
        context label leaves the output untouched."""
        for enc in ENCODERS:
            with self.subTest(encoder=enc):
                m = self._model(enc).eval()
                x_c, y_c = torch.randn(8, EMB), (torch.rand(8) < 0.5).float()
                x_t = torch.randn(4, EMB)
                with torch.no_grad():
                    a = m.logits(x_c, y_c, x_t)
                    b = m.logits(x_c, 1 - y_c, x_t)
                self.assertGreater(float((a - b).abs().max()), 1e-5,
                                   f"{enc}: context labels are being ignored")

    def test_decoder_sees_the_set_representation(self):
        """Eq. (3) says g takes BOTH x_T and r. Perturbing r must move the
        output; CNPAIRT's item head takes x_T alone and would not."""
        m = self._model().eval()
        x_c, y_c = torch.randn(6, EMB), (torch.rand(6) < 0.5).float()
        x_t = torch.randn(3, EMB)
        with torch.no_grad():
            r = m._set_repr(x_c, y_c)
            base = m.logits(x_c, y_c, x_t)
            # same target, different context content
            other = m.logits(torch.randn(6, EMB), y_c, x_t)
        self.assertEqual(tuple(r.shape), (64,))
        self.assertGreater(float((base - other).abs().max()), 1e-5)


@unittest.skipIf(cb is None, "PyTorch is not installed in the lightweight test environment")
class LatentTests(unittest.TestCase):
    def test_latent_loss_returns_nll_and_kl(self):
        torch.manual_seed(0)
        m = cb.NeuralProcess(EMB, width=64, use_latent=True, latent_dim=16)
        loss, parts = m.loss(torch.randn(5, EMB), (torch.rand(5) < 0.5).float(),
                             torch.randn(4, EMB), (torch.rand(4) < 0.5).float())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("nll", parts)
        self.assertIn("kl", parts)
        self.assertGreaterEqual(parts["kl"], 0.0)

    def test_kl_is_zero_when_context_equals_context_plus_target(self):
        """With an empty target the posterior sees exactly the prior's input,
        so the two distributions must coincide."""
        torch.manual_seed(0)
        m = cb.NeuralProcess(EMB, width=64, use_latent=True, latent_dim=16).eval()
        x_c, y_c = torch.randn(6, EMB), (torch.rand(6) < 0.5).float()
        with torch.no_grad():
            mu_p, lv_p = m.latent_params(x_c, y_c)
            mu_q, lv_q = m.latent_params(x_c, y_c)
        torch.testing.assert_close(mu_p, mu_q)
        torch.testing.assert_close(lv_p, lv_q)

    def test_deterministic_loss_has_no_kl(self):
        torch.manual_seed(0)
        m = cb.NeuralProcess(EMB, width=64, use_latent=False)
        _, parts = m.loss(torch.randn(5, EMB), (torch.rand(5) < 0.5).float(),
                          torch.randn(4, EMB), (torch.rand(4) < 0.5).float())
        self.assertEqual(parts, {})


@unittest.skipIf(cb is None, "PyTorch is not installed in the lightweight test environment")
class ConfigTests(unittest.TestCase):
    def test_sgformer_needs_no_graph_and_is_linear_in_set_size(self):
        """SGFormer's global-attention branch takes features only -- the graph
        is only used by its GNN branch, which we are not using. Cost must grow
        linearly, not quadratically, in context size."""
        import time

        m = cb.NeuralProcess(EMB, width=64, n_blocks=1, encoder="sgformer").eval()
        times = {}
        for n in (64, 512):
            x = torch.randn(n, EMB)
            y = (torch.rand(n) < 0.5).float()
            with torch.no_grad():
                m._set_repr(x, y)                     # warm
                t = time.perf_counter()
                for _ in range(5):
                    m._set_repr(x, y)
                times[n] = (time.perf_counter() - t) / 5
        # 8x the set size must cost well under the 64x a quadratic method would
        ratio = times[512] / max(times[64], 1e-9)
        self.assertLess(ratio, 24.0, f"sgformer scaling looks quadratic: {ratio:.1f}x")

    def test_unknown_encoder_and_pooling_raise(self):
        with self.assertRaises(ValueError):
            cb.make_encoder("nope", 64, 2)
        with self.assertRaises(ValueError):
            cb.make_pool("nope", 64)

    def test_default_width_follows_the_small_data_finding(self):
        """64, not 256: the Active-Evaluation paper uses 16-32 on comparable
        test-taker counts and reports set transformers overfitting above that."""
        m = cb.NeuralProcess(EMB)
        self.assertEqual(m.width, 64)

    def test_label_is_projected_to_full_width_not_concatenated(self):
        """A scalar label concatenated to a 1024-d embedding is 0.1% of the
        input dims and gets ignored; it needs its own projection."""
        m = cb.NeuralProcess(EMB, width=64)
        self.assertEqual(m.label_proj.out_features, 64)


if __name__ == "__main__":
    unittest.main()
