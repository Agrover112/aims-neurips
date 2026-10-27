"""Unit tests for the Amortized IRT model (item difficulty predicted from
question-text embeddings instead of fit as a free per-item scalar).

Pure-function / small-tensor tests only: no Modal, no GPU, no corpus. They pin
the three things a silent bug here would corrupt without any error surfacing
in a full run:

  * the 1PL/2PL/3PL parameter-count contract (which heads exist per kind),
  * that cold-start scoring (zero responses for a held-out item) actually uses
    only the embedding, not anything fit from responses,
  * that amortization can recover a difficulty signal that is a real function
    of the embedding, which is the entire premise of the model.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "organized" / "testing" / "offline"))

try:
    import torch

    import modal_amortized_irt as amort
except ModuleNotFoundError:  # torch and modal are experiment-only dependencies
    torch = None
    amort = None


@unittest.skipIf(amort is None, "PyTorch/modal are not installed in the lightweight test environment")
class ParameterCountTests(unittest.TestCase):
    """d is amortized for 2PL/3PL and fixed for 1PL; c is free-per-item for 3PL
    only. Getting this wrong silently turns one model into another."""

    def setUp(self):
        torch.manual_seed(0)
        self.n_subjects, self.emb_dim, self.n_items = 5, 8, 6

    def _make(self, kind):
        return amort.AmortizedIRT(self.n_subjects, self.emb_dim, kind, self.n_items, seed=0)

    def test_1pl_has_no_discrimination_head(self):
        self.assertIsNone(self._make("1pl").d_head)

    def test_2pl_has_a_discrimination_head(self):
        self.assertIsNotNone(self._make("2pl").d_head)

    def test_3pl_has_a_discrimination_head(self):
        self.assertIsNotNone(self._make("3pl").d_head)

    def test_only_3pl_has_a_guessing_parameter(self):
        self.assertIsNone(self._make("1pl").logit_c)
        self.assertIsNone(self._make("2pl").logit_c)
        self.assertIsNotNone(self._make("3pl").logit_c)

    def test_3pl_guessing_has_one_entry_per_item(self):
        self.assertEqual(self._make("3pl").logit_c.shape, (self.n_items,))

    def test_1pl_discrimination_is_always_one(self):
        # a is (n_items, D) since the model gained an ability dimension.
        model = self._make("1pl")
        emb = torch.randn(4, self.emb_dim)
        _, a = model.item_params(emb)
        torch.testing.assert_close(a, torch.ones(4, model.d_dim))

    def test_2pl_discrimination_is_positive(self):
        model = self._make("2pl")
        emb = torch.randn(20, self.emb_dim) * 5  # wide range, including large negatives
        _, d = model.item_params(emb)
        self.assertTrue(torch.all(d > 0))

    def test_parameters_include_theta_and_every_head(self):
        model = self._make("3pl")
        ids = {id(p) for p in model.parameters()}
        self.assertIn(id(model.theta), ids)
        for p in model.b_head.parameters():
            self.assertIn(id(p), ids)
        for p in model.d_head.parameters():
            self.assertIn(id(p), ids)
        self.assertIn(id(model.logit_c), ids)


@unittest.skipIf(amort is None, "PyTorch/modal are not installed in the lightweight test environment")
class ProbsTests(unittest.TestCase):
    """``probs`` takes per-pair tensors (theta[s_idx], emb[i_idx]), same
    length -- it is not a cross product of every subject against every item."""

    def setUp(self):
        torch.manual_seed(1)
        self.model = amort.AmortizedIRT(3, 6, "3pl", n_items=4, seed=1)
        self.item_idx = torch.tensor([0, 1, 2, 3])
        self.emb = torch.randn(4, 6)
        self.theta_pairs = self.model.theta[torch.tensor([0, 0, 1, 2])]  # (4, D)

    def test_probabilities_are_in_open_unit_interval(self):
        p = self.model.probs(self.theta_pairs, self.emb, self.item_idx)
        self.assertTrue(torch.all(p > 0))
        self.assertTrue(torch.all(p < 1))

    def test_guessing_floors_the_probability(self):
        # Drive theta far below every item's difficulty so the 2PL term is
        # ~0; the 3PL mixture must still sit at or above the guessing floor.
        model = amort.AmortizedIRT(1, 6, "3pl", n_items=4, seed=2)
        with torch.no_grad():
            model.theta.fill_(-50.0)
            model.logit_c.fill_(2.0)  # sigmoid(2)*0.5 ~= 0.44
        item_idx = torch.arange(4)
        theta_pairs = model.theta[torch.zeros(4, dtype=torch.long)]  # (4, D)
        p = model.probs(theta_pairs, self.emb, item_idx)
        floor = float(torch.sigmoid(model.logit_c[0]) * 0.5)
        self.assertTrue(torch.all(p >= floor - 1e-4))

    def test_without_item_index_3pl_falls_back_to_the_2pl_curve(self):
        # item_idx=None means "no guessing applied" -- used nowhere in the
        # training loop, but the fallback must not silently misbehave.
        p_no_idx = self.model.probs(self.theta_pairs, self.emb)
        b, a = self.model.item_params(self.emb)
        # paper's difficulty form, summed over the ability dimensions
        expected = torch.sigmoid(
            (a * (self.theta_pairs - b.unsqueeze(-1))).sum(-1)).clamp(1e-6, 1 - 1e-6)
        torch.testing.assert_close(p_no_idx, expected)


@unittest.skipIf(amort is None, "PyTorch/modal are not installed in the lightweight test environment")
class FitRecoversSignalTests(unittest.TestCase):
    """The entire premise of amortization: b_j = f(e_j) must be learnable
    through the IRT likelihood alone, with no free per-item difficulty."""

    def test_2pl_fit_reduces_loss_on_separable_synthetic_data(self):
        rng = np.random.default_rng(0)
        n_s, n_i, dim = 40, 80, 12
        emb = rng.normal(size=(n_i, dim)).astype(np.float32)
        w = rng.normal(size=dim)
        b_true = np.tanh(emb @ w) * 1.5
        theta_true = rng.normal(size=n_s)
        p = 1.0 / (1.0 + np.exp(-(theta_true[:, None] - b_true[None, :])))
        y = (rng.random(p.shape) < p).astype(np.float32)
        si, ii = np.meshgrid(np.arange(n_s), np.arange(n_i), indexing="ij")
        si, ii, yy = si.ravel(), ii.ravel(), y.ravel()

        model = amort.AmortizedIRT(n_s, dim, "2pl", n_i, seed=0)
        hist = amort._fit(model, torch.as_tensor(emb), si, ii, yy, steps=150, desc="")
        self.assertLess(hist[-1], hist[0])
        self.assertTrue(np.isfinite(hist[-1]))

    def test_coldstart_item_prediction_uses_only_the_embedding(self):
        """Score an item the fit never saw a single response for. If the
        prediction depends on anything but the embedding, this cannot work --
        there is nothing else available for a genuinely new item."""
        rng = np.random.default_rng(1)
        n_s, n_i, dim = 30, 60, 10
        emb = rng.normal(size=(n_i, dim)).astype(np.float32)
        w = rng.normal(size=dim)
        b_true = np.tanh(emb @ w) * 1.2
        theta_true = rng.normal(size=n_s)
        p = 1.0 / (1.0 + np.exp(-(theta_true[:, None] - b_true[None, :])))
        y = (rng.random(p.shape) < p).astype(np.float32)
        si, ii = np.meshgrid(np.arange(n_s), np.arange(n_i), indexing="ij")
        si, ii, yy = si.ravel(), ii.ravel(), y.ravel()

        held = rng.choice(n_i, n_i // 4, replace=False)
        core = ~np.isin(ii, held)

        model = amort.AmortizedIRT(n_s, dim, "1pl", n_i, seed=0)
        amort._fit(model, torch.as_tensor(emb), si[core], ii[core], yy[core],
                   steps=150, desc="")

        with torch.no_grad():
            b_hat, _ = model.item_params(torch.as_tensor(emb))
        # Correlation on items that contributed zero responses to the fit.
        corr = np.corrcoef(b_hat.numpy()[held], b_true[held])[0, 1]
        self.assertGreater(corr, 0.3, f"cold-start b_hat barely tracks b_true (r={corr:.3f})")

    def test_fit_theta_only_does_not_move_item_parameters(self):
        model = amort.AmortizedIRT(2, 6, "2pl", n_items=5, seed=3)
        emb = torch.randn(5, 6)
        before = {k: v.clone() for k, v in model.b_head.state_dict().items()}
        amort._fit_theta_only(model, emb, np.array([0, 1, 2]),
                              np.array([1.0, 0.0, 1.0], dtype=np.float32), steps=20)
        after = model.b_head.state_dict()
        for key in before:
            torch.testing.assert_close(before[key], after[key])


@unittest.skipIf(amort is None, "PyTorch/modal are not installed in the lightweight test environment")
class ScoreTests(unittest.TestCase):
    def test_perfect_predictions_give_auc_one_and_zero_brier(self):
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        p = np.array([0.01, 0.02, 0.98, 0.99], dtype=np.float32)
        auc, brier = amort._score(y, p)
        self.assertAlmostEqual(auc, 1.0)
        self.assertLess(brier, 0.01)

    def test_single_class_labels_give_nan_auc_not_an_exception(self):
        import warnings

        y = np.array([1, 1, 1], dtype=np.float32)
        p = np.array([0.6, 0.7, 0.8], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn warns on single-class AUC; we handle it
            auc, _ = amort._score(y, p)
        self.assertTrue(np.isnan(auc))

    def test_brier_is_mean_squared_error(self):
        y = np.array([0, 1], dtype=np.float32)
        p = np.array([0.3, 0.9], dtype=np.float32)
        _, brier = amort._score(y, p)
        self.assertAlmostEqual(brier, float(np.mean((p - y) ** 2)), places=6)


if __name__ == "__main__":
    unittest.main()
