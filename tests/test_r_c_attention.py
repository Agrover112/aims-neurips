from __future__ import annotations

import unittest

try:
    import torch

    from aims_starter.r_c_attention import ContextConditionedIRT
except ModuleNotFoundError:  # Torch is an experiment dependency, not a submission dependency.
    torch = None
    ContextConditionedIRT = None


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

        expected_a, expected_b = self.model._anchor_parameters(
            self.model.encode_item(self.x_C)
        )
        torch.testing.assert_close(a_first, expected_a)
        torch.testing.assert_close(b_first, expected_b)
        torch.testing.assert_close(a_first, a_second)
        torch.testing.assert_close(b_first, b_second)

    def test_additive_context_still_affects_target_parameters(self) -> None:
        _, r_C = self.model._context_repr(self.x_C, self.y_C)
        h_T = self.model.encode_item(self.x_T)
        with torch.no_grad():
            a_plain, b_plain = self.model._target_parameters(
                h_T, torch.zeros_like(r_C)
            )
            a_context, b_context = self.model._target_parameters(h_T, r_C)
        self.assertFalse(torch.allclose(a_plain, a_context))
        self.assertFalse(torch.allclose(b_plain, b_context))


if __name__ == "__main__":
    unittest.main()
