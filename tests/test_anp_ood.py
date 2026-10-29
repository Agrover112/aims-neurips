import unittest
import numpy as np
import torch
from organized.training import anp_ood_core as core

class ANPOODTests(unittest.TestCase):
    def test_balanced_disjoint_repeatable_training_only(self):
        tasks = [{"role": "train", "benchmark": b, "items": np.arange(80), "labels": np.arange(80)%2}
                 for b in ("a", "a", "b", "c")]
        tasks.append({"role": "validation", "benchmark": "d", "items": np.arange(80), "labels": np.zeros(80)})
        groups = core.training_groups(tasks)
        self.assertEqual(set(groups), {"a", "b", "c"})
        features = torch.randn(80, 4)
        first, meta = core.sample_batch(tasks, groups, features, np.random.default_rng(42))
        again, meta2 = core.sample_batch(tasks, groups, features, np.random.default_rng(42))
        self.assertEqual(len(first), 3)
        for a,b, m in zip(first, again, meta):
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y)
            self.assertLess(m[0],4)
            self.assertFalse(set(m[2]) & set(m[3]))

    def test_dro_normalized_and_upweights_hard_group(self):
        state = None
        for _ in range(20):
            state, weights = core.dro_weights(state, torch.tensor([.1, 2., .2], requires_grad=True), eta=.1)
        self.assertAlmostEqual(float(weights.sum()),1.,places=6)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertFalse(weights.requires_grad)

    def test_weight_average_math_and_resume(self):
        avg, n = core.average_state(None, 0, {"w": torch.tensor([1.,3.])})
        avg, n = core.average_state(avg, n, {"w": torch.tensor([3.,5.])})
        torch.testing.assert_close(avg["w"], torch.tensor([2.,4.]))
        avg, n = core.average_state(avg, n, {"w": torch.tensor([5.,7.])})
        torch.testing.assert_close(avg["w"], torch.tensor([3.,5.]))
        self.assertEqual(n,3)

    def test_sam_reuses_noise_and_restores_on_failure(self):
        torch.manual_seed(4)
        model = torch.nn.Linear(2,1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        x = torch.randn(4,2)
        noises = []
        def closure():
            noise = torch.randn(4,1)
            noises.append(noise.clone())
            return (model(x)-noise).square().mean()
        before = model.weight.detach().clone()
        first, second = core.sam_step(model, optimizer, closure)
        torch.testing.assert_close(noises[0], noises[1])
        self.assertFalse(torch.equal(before,model.weight))
        originals = [p.detach().clone() for p in model.parameters()]
        count = 0
        def fail():
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("test second pass")
            return model(x).square().mean()
        with self.assertRaisesRegex(RuntimeError,"test second pass"):
            core.sam_step(model,optimizer,fail)
        for p,v in zip(model.parameters(), originals):
            torch.testing.assert_close(p,v,rtol=0,atol=0)

    def test_evaluation_preserves_pools_and_probabilities(self):
        class Constant:
            def eval(self):
                return self
            def logits(self,xc,yc,xt):
                return torch.zeros(len(xt))
        tasks = [{"subject":"s","benchmark":"b","items":np.arange(80),"labels":np.arange(80)%2}]
        bank = [{"task":0,"acquisition":np.arange(40),"targets":np.arange(40,80),"weight":1.}]
        features = torch.zeros(80,4)
        result = core.evaluate(Constant(),tasks,bank,features)
        self.assertAlmostEqual(result["brier_alc"],.25)
        diagnostic = core.diagnostics(Constant(),tasks,bank,features,"heldout_subjects")
        self.assertAlmostEqual(diagnostic["diagnostic/heldout_subjects/same_subject_split_half/brier_score"],.25)
