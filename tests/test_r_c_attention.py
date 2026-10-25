from __future__ import annotations

import unittest

try:
    import torch

    from aims_starter.r_c_attention import (
        ContextConditionedIRT,
        MultidimContextIRT,
        marginalize_logits,
    )
except ModuleNotFoundError:  # Torch is an experiment dependency, not a submission dependency.
    torch = None
    ContextConditionedIRT = None
    MultidimContextIRT = None
    marginalize_logits = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class AdditiveAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ContextConditionedIRT(6, mode="additive_ab", d_model=8)
        # Make the context offsets non-zero so a circular anchor is visible.
        with torch.no_grad():
            self.model.ab_head[-1].weight.normal_(std=0.1)
            self.model.ab_head[-1].bias.normal_(std=0.1)
        self.x_C = torch.randn(5, 6)
        self.y_C = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        self.x_T = torch.randn(4, 6)

    def _capture_anchor(self, y_C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        original = self.model.map_theta

        def spy(a_C, b_C, labels, **kwargs):
            captured["a"] = a_C.detach().clone()
            captured["b"] = b_C.detach().clone()
            return original(a_C, b_C, labels, **kwargs)

        self.model.map_theta = spy
        with torch.no_grad():
            self.model(self.x_C, y_C, self.x_T)
        return captured["a"], captured["b"]

    def test_additive_anchor_uses_item_only_parameters(self) -> None:
        a_first, b_first = self._capture_anchor(self.y_C)
        a_second, b_second = self._capture_anchor(1.0 - self.y_C)

        expected_a, expected_b = self.model._anchor_parameters(self.model.encode_item(self.x_C))
        torch.testing.assert_close(a_first, expected_a)
        torch.testing.assert_close(b_first, expected_b)
        torch.testing.assert_close(a_first, a_second)
        torch.testing.assert_close(b_first, b_second)

    def test_additive_context_still_affects_target_parameters(self) -> None:
        _, r_C = self.model._context_repr(self.x_C, self.y_C)
        h_T = self.model.encode_item(self.x_T)
        with torch.no_grad():
            a_plain, b_plain = self.model._target_parameters(h_T, torch.zeros_like(r_C))
            a_context, b_context = self.model._target_parameters(h_T, r_C)
        self.assertFalse(torch.allclose(a_plain, a_context))
        self.assertFalse(torch.allclose(b_plain, b_context))


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class MultidimPriorTests(unittest.TestCase):
    def test_tighter_prior_shrinks_map_ability(self) -> None:
        a_c = torch.ones((4, 2))
        d_c = torch.zeros(4)
        y_c = torch.ones(4)

        theta_tau1 = MultidimContextIRT.map_theta(a_c, d_c, y_c, prior_variance=1.0)
        theta_tau4 = MultidimContextIRT.map_theta(a_c, d_c, y_c, prior_variance=4.0)

        self.assertLess(float(theta_tau1.norm()), float(theta_tau4.norm()))
        self.assertLessEqual(float(theta_tau1.abs().max()), 2.0)

    def test_prior_variance_is_kept_on_the_model(self) -> None:
        model = MultidimContextIRT(6, mode="scalar", ability_dim=4, d_model=8, prior_variance=1.0)
        self.assertEqual(model.prior_variance, 1.0)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class JointDifficultyUpdateTests(unittest.TestCase):
    def test_centering_removes_constant_difficulty_direction(self) -> None:
        context = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
        target = torch.tensor([[3.0, 1.0]])

        centered_context, centered_target = MultidimContextIRT.center_difficulty_loadings(
            context, target
        )

        torch.testing.assert_close(centered_context, torch.zeros_like(context))
        torch.testing.assert_close(centered_target, torch.tensor([[1.0, 2.0]]))

    def test_one_label_updates_theta_but_not_centered_difficulty(self) -> None:
        a_c = torch.tensor([[1.0, 0.5]])
        d_c = torch.zeros(1)
        u_c, _ = MultidimContextIRT.center_difficulty_loadings(
            torch.tensor([[0.4, -0.8]]), torch.zeros((0, 2))
        )

        theta, adapter = MultidimContextIRT.map_theta_and_difficulty(
            a_c, d_c, u_c, torch.ones(1), theta_prior_variance=1.0
        )

        self.assertGreater(float(theta.norm()), 0.0)
        torch.testing.assert_close(adapter, torch.zeros_like(adapter))

    def test_joint_map_captures_benchmark_relative_difficulty(self) -> None:
        a_c = torch.ones((8, 1))
        d_c = torch.zeros(8)
        u_c = torch.tensor([[-1.5], [-1.2], [-0.9], [-0.6], [0.6], [0.9], [1.2], [1.5]])
        y_c = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        theta_only = MultidimContextIRT.map_theta(a_c, d_c, y_c, prior_variance=1.0)
        theta, adapter = MultidimContextIRT.map_theta_and_difficulty(
            a_c,
            d_c,
            u_c,
            y_c,
            theta_prior_variance=1.0,
            adapter_prior_variance=1.0,
        )
        theta_only_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            a_c @ theta_only - d_c, y_c
        )
        joint_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            a_c @ theta - d_c - u_c @ adapter, y_c
        )

        self.assertGreater(float(adapter[0]), 0.0)
        self.assertLess(float(joint_loss), float(theta_only_loss))

    def test_adapter_is_opt_in_and_forward_is_finite(self) -> None:
        baseline = MultidimContextIRT(6, mode="scalar", ability_dim=4, d_model=8)
        adapted = MultidimContextIRT(
            6,
            mode="scalar",
            ability_dim=4,
            d_model=8,
            difficulty_adapter_dim=4,
            adapter_prior_variance=1.0,
        )
        self.assertFalse(hasattr(baseline, "difficulty_adapter_head"))
        self.assertTrue(hasattr(adapted, "difficulty_adapter_head"))

        logits = adapted(
            torch.randn(4, 6),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.randn(5, 6),
        )
        self.assertEqual(tuple(logits.shape), (5,))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_adapter_head_receives_training_gradient(self) -> None:
        torch.manual_seed(7)
        model = MultidimContextIRT(
            6,
            mode="scalar",
            ability_dim=4,
            d_model=8,
            prior_variance=1.0,
            difficulty_adapter_dim=4,
        )
        logits = model(
            torch.randn(6, 6),
            torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            torch.randn(7, 6),
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        )
        loss.backward()

        gradient = model.difficulty_adapter_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.norm()), 0.0)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class MultidimPosteriorPredictiveTests(unittest.TestCase):
    def test_zero_variance_recovers_plugin_logits(self) -> None:
        logits = torch.tensor([-2.0, 0.0, 3.0])
        torch.testing.assert_close(marginalize_logits(logits, torch.zeros(3)), logits)

    def test_context_reduces_laplace_uncertainty(self) -> None:
        theta = torch.zeros(2)
        empty_a = torch.zeros((0, 2))
        empty_d = torch.zeros(0)
        prior = MultidimContextIRT.laplace_covariance(empty_a, empty_d, theta, prior_variance=1.0)
        a_c = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        d_c = torch.zeros(3)
        posterior = MultidimContextIRT.laplace_covariance(a_c, d_c, theta, prior_variance=1.0)

        torch.testing.assert_close(prior, torch.eye(2))
        self.assertLess(float(torch.trace(posterior)), float(torch.trace(prior)))
        self.assertGreater(float(torch.linalg.eigvalsh(posterior).min()), 0.0)

    def test_posterior_prediction_shrinks_logits_without_changing_sign(self) -> None:
        torch.manual_seed(4)
        model = MultidimContextIRT(6, mode="scalar", ability_dim=4, d_model=8, prior_variance=1.0)
        model.eval()
        x_c = torch.randn(3, 6)
        y_c = torch.tensor([1.0, 0.0, 1.0])
        x_t = torch.randn(5, 6)

        with torch.no_grad():
            plugin = model(x_c, y_c, x_t)
            posterior = model(x_c, y_c, x_t, marginalize_theta=True)

        self.assertTrue(torch.all(posterior.abs() <= plugin.abs() + 1e-7))
        self.assertTrue(torch.all(plugin * posterior >= 0.0))
        self.assertFalse(torch.allclose(plugin, posterior))


if __name__ == "__main__":
    unittest.main()
