"""Regression tests for the bug classes that actually bit us in the IRT family.

Every test here corresponds to a real bug that shipped, produced plausible
numbers, and was only caught by inspection. They are written as *invariants*
over the whole family rather than against one file, so a new IRT variant gets
checked for free by adding it to the module list.

The bugs, and why each was invisible:

1. **MIRT was not multidimensional.** ``z = a[i] * theta[s].sum(dim=1)`` --
   a scalar discrimination against the *sum* of theta's components. Since
   sum(theta) is one scalar, D=2 and D=4 were algebraically identical to D=1.
   Every run completed and reported sensible-looking AUCs.

2. **3PL had no guessing parameter.** The branch read
   ``kind in ("2pl", "3pl")`` with no ``c`` anywhere, so "3PL" was a duplicate
   2PL differing only by init seed.

3. **A prior that silently deletes a parameter.** The Bayesian jobs used a
   ``.mean()`` likelihood with a ``.sum()`` log-prior, so with thousands of
   items the prior outweighed the data ~n_obs and crushed ``log_a`` to 0,
   collapsing every 2PL into a 1PL. The two agreed to 8e-6 AUC.

4. **Undefined helper.** Three jobs called ``_fit_core()``, which was defined
   in none of them -- an instant NameError, so those jobs produced nothing.

5. **A GPU that was requested but never used.** ``gpu="L4"`` with no ``.cuda()``
   anywhere: CPU-speed training at GPU prices, ~17 s/step, and it could never
   have finished.

The shape of all of these is the same: *two things that should differ, don't*.
That is what these tests check.
"""

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT / "organized" / "testing" / "offline"
sys.path.insert(0, str(OFFLINE))

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _load(name):
    """Import an offline job module, with modal stubbed so no network is hit."""
    import importlib.util
    import types

    if "modal" not in sys.modules:
        m = types.ModuleType("modal")

        class _Img:
            def pip_install(self, *a, **k):
                return self

            def add_local_file(self, *a, **k):
                return self

            def apt_install(self, *a, **k):
                return self

            @staticmethod
            def debian_slim(*a, **k):
                return _Img()

        class _App:
            def __init__(self, *a, **k):
                pass

            def function(self, *a, **k):
                return lambda f: f

            def local_entrypoint(self, *a, **k):
                return lambda f: f

        m.App = _App
        m.Image = _Img
        m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: None)
        sys.modules["modal"] = m

    path = OFFLINE / name
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


JOB_FILES = sorted(p.name for p in OFFLINE.glob("modal_*.py")) if OFFLINE.exists() else []


class StaticContractTests(unittest.TestCase):
    """Cheap whole-directory checks -- no model fitting, no imports executed."""

    def test_no_offline_job_requests_a_gpu(self):
        """Bug 5. Every offline job is CPU-only; a stray gpu= is a cost bug.

        These are small free-parameter / small-MLP fits. The one time a GPU was
        requested here it was never actually used, and burned ~126 L4
        container-hours at CPU speed.
        """
        offenders = [f for f in JOB_FILES if 'gpu="' in (OFFLINE / f).read_text()]
        self.assertEqual(offenders, [], f"offline jobs requesting a GPU: {offenders}")

    def test_no_job_calls_an_undefined_module_level_helper(self):
        """Bug 4. ``_fit_core`` was called in three files and defined in none."""
        import builtins

        bad = {}
        for f in JOB_FILES:
            tree = ast.parse((OFFLINE / f).read_text())
            defined = {n.name for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            # names bound anywhere (imports, assignments, args) are fine
            for n in ast.walk(tree):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    defined.add(n.id)
                elif isinstance(n, ast.arg):
                    defined.add(n.arg)
                elif isinstance(n, ast.Import):
                    for a in n.names:
                        defined.add((a.asname or a.name).split(".")[0])
                elif isinstance(n, ast.ImportFrom):
                    for a in n.names:
                        defined.add(a.asname or a.name)
            called = {n.func.id for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            missing = sorted(c for c in called
                             if c not in defined and not hasattr(builtins, c))
            if missing:
                bad[f] = missing
        self.assertEqual(bad, {}, f"calls to undefined helpers: {bad}")

    def test_a_mean_likelihood_is_never_paired_with_a_sum_prior(self):
        """Bug 3. Mixing the two scales the prior by the observation count.

        Unscaled, that crushed log_a to 0 and turned every 2PL into a 1PL. A
        .sum() prior is fine only if divided by the observation count.
        """
        offenders = []
        for f in JOB_FILES:
            text = (OFFLINE / f).read_text()
            for line in text.splitlines():
                if "square().sum()" not in line:
                    continue
                # acceptable only when normalised by an observation count
                if "len(y" not in line and "/ n_obs" not in line:
                    # the division may sit on the following line of the expression
                    idx = text.index(line)
                    window = text[idx:idx + 400]
                    if "len(y" not in window and "/ n_obs" not in window:
                        offenders.append(f"{f}: {line.strip()[:70]}")
        self.assertEqual(offenders, [],
                         "summed priors not normalised by observation count:\n"
                         + "\n".join(offenders))


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class AmortizedIRTInvariantTests(unittest.TestCase):
    """Behavioural invariants for the amortized model.

    Each of these would have failed on the shipped buggy code.
    """

    @classmethod
    def setUpClass(cls):
        cls.ai = _load("modal_amortized_irt.py")
        if cls.ai is None:
            raise unittest.SkipTest("modal_amortized_irt.py not present")
        torch.set_num_threads(2)

    def _world(self, n_s=60, n_i=150, dim=32, D=2, seed=0):
        """A genuinely D-dimensional world whose difficulty depends on the text."""
        rng = np.random.default_rng(seed)
        emb = rng.normal(size=(n_i, dim)).astype(np.float32)
        W = rng.normal(size=(dim, D))
        w = rng.normal(size=dim)
        a = np.log1p(np.exp(emb @ W)) + 1e-3
        b = np.tanh(emb @ w) * 1.5
        th = rng.normal(size=(n_s, D))
        z = (a[None, :, :] * (th[:, None, :] - b[None, :, None])).sum(-1)
        y = (rng.random(z.shape) < 1 / (1 + np.exp(-z))).astype(np.float32)
        si, ii = np.meshgrid(np.arange(n_s), np.arange(n_i), indexing="ij")
        return emb, si.ravel(), ii.ravel(), y.ravel()

    def test_ability_components_are_not_interchangeable(self):
        """Bug 1, stated precisely. THIS is the test that catches the collapse.

        The original bug was ``a[i] * theta[s].sum(dim=1)``: a scalar
        discrimination against the *sum* of theta's components. Under that
        form the prediction depends only on sum(theta), so permuting theta's
        components leaves it unchanged.

        Comparing D=1 vs D=2 *loss* does NOT catch this -- the losses differ
        anyway because the parameter count differs. Verified: that weaker test
        passes against the buggy implementation. Holding the sum fixed and
        permuting the components is the real invariant.
        """
        cpu = torch.device("cpu")
        m = self.ai.AmortizedIRT(2, 16, "2pl", 4, seed=0, device=cpu, d_dim=2)
        emb = torch.randn(4, 16)
        # two ability vectors with the SAME sum but swapped components
        theta_a = torch.tensor([[2.0, -1.0]]).repeat(4, 1)
        theta_b = torch.tensor([[-1.0, 2.0]]).repeat(4, 1)
        self.assertAlmostEqual(float(theta_a.sum()), float(theta_b.sum()), places=6)
        with torch.no_grad():
            pa = m.probs(theta_a, emb)
            pb = m.probs(theta_b, emb)
        self.assertGreater(float((pa - pb).abs().max()), 1e-4,
                           "swapping theta's components left the prediction "
                           "unchanged -- the model depends only on sum(theta), "
                           "so the ability dimensions have collapsed to one")

    def test_discrimination_is_applied_per_dimension(self):
        """Bug 1, complementary check on the item side.

        Perturbing a single component of one item's discrimination must change
        that item's prediction. If only an aggregate of ``a`` is used, it will
        not.
        """
        cpu = torch.device("cpu")
        m = self.ai.AmortizedIRT(1, 16, "2pl", 2, seed=0, device=cpu, d_dim=2)
        emb = torch.randn(2, 16)
        theta = torch.tensor([[1.5, -0.5], [1.5, -0.5]])
        with torch.no_grad():
            b_all, a_all = m.item_params(emb)
            base = m.probs_gathered(theta, b_all, a_all, torch.tensor([0, 1]))
            bumped = a_all.clone()
            bumped[0, 0] += 1.0          # one component of one item only
            moved = m.probs_gathered(theta, b_all, bumped, torch.tensor([0, 1]))
        self.assertGreater(float((base[0] - moved[0]).abs()), 1e-4,
                           "changing one component of a_j did not move its "
                           "prediction -- discrimination is not per-dimension")
        self.assertAlmostEqual(float(base[1]), float(moved[1]), places=6,
                               msg="perturbing item 0 changed item 1")

    def test_theta_and_discrimination_have_the_requested_dimension(self):
        """Bug 1, structurally: shapes must carry D, not a scalar."""
        cpu = torch.device("cpu")
        for d in (1, 2, 4):
            m = self.ai.AmortizedIRT(5, 16, "2pl", 7, 0, device=cpu, d_dim=d)
            self.assertEqual(tuple(m.theta.shape), (5, d))
            _, a = m.item_params(torch.randn(7, 16))
            self.assertEqual(tuple(a.shape), (7, d))

    def test_probability_is_one_half_when_ability_equals_difficulty(self):
        """Pins the paper's difficulty form p = sigma(a*(theta - b)).

        The slope-intercept form sigma(<a,theta> - b) is equally expressive but
        makes b an intercept (= a*difficulty), so the embedding head would be
        predicting a product rather than difficulty, and b would not be
        comparable with the paper's z. At theta == b the difficulty form gives
        exactly 0.5 regardless of a; the intercept form does not.
        """
        cpu = torch.device("cpu")
        m = self.ai.AmortizedIRT(4, 16, "2pl", 4, 0, device=cpu, d_dim=1)
        emb = torch.randn(4, 16)
        with torch.no_grad():
            b, _ = m.item_params(emb)
            p = m.probs(b.unsqueeze(-1), emb)
        np.testing.assert_allclose(p.numpy(), np.full(4, 0.5), atol=1e-6)

    def test_guessing_floors_the_curve_for_3pl_only(self):
        """Bug 2. 3PL must actually carry c; 1PL/2PL must not."""
        cpu = torch.device("cpu")
        self.assertIsNone(self.ai.AmortizedIRT(2, 16, "1pl", 3, 0, device=cpu).logit_c)
        self.assertIsNone(self.ai.AmortizedIRT(2, 16, "2pl", 3, 0, device=cpu).logit_c)
        m = self.ai.AmortizedIRT(1, 16, "3pl", 3, 0, device=cpu, d_dim=1)
        self.assertIsNotNone(m.logit_c)
        with torch.no_grad():
            m.logit_c.fill_(2.0)                      # floor ~= 0.44
            emb = torch.randn(3, 16)
            theta = torch.full((3, 1), -50.0)         # drive the 2PL term to ~0
            p = m.probs(theta, emb, torch.arange(3))
        floor = float(torch.sigmoid(m.logit_c[0]) * 0.5)
        self.assertTrue(bool((p >= floor - 1e-4).all()),
                        f"3PL dropped below its guessing floor: {p} < {floor}")

    def test_gathered_and_per_pair_paths_agree(self):
        """The fast path (heads over unique items) must equal the slow path.

        ``_fit`` evaluates the heads once over the (n_items, dim) matrix and
        gathers; ``probs`` evaluates per response. Same maths, 17x cheaper --
        but only if they actually agree.
        """
        cpu = torch.device("cpu")
        m = self.ai.AmortizedIRT(6, 16, "2pl", 5, 0, device=cpu, d_dim=2)
        emb = torch.randn(5, 16)
        idx = torch.tensor([0, 2, 4, 1])
        theta = m.theta[torch.tensor([0, 1, 2, 3])].detach()
        with torch.no_grad():
            b_all, a_all = m.item_params(emb)
            fast = m.probs_gathered(theta, b_all, a_all, idx)
            slow = m.probs(theta, emb[idx], idx)
        torch.testing.assert_close(fast, slow)


if __name__ == "__main__":
    unittest.main()
