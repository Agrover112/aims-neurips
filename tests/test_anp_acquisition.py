import unittest
import numpy as np
import torch
from organized.training import anp_acquisition as core

class Toy(torch.nn.Module):
    latent_dim=2
    def logits(self, xc, yc, xt, z=None):
        base = (yc.sum()+2)/(len(yc)+4)
        result = torch.logit(base).expand(len(xt))
        return result if z is None else result+z[0]
    def latent_params(self, xc, yc):
        return torch.zeros(2), torch.zeros(2)

class AcquisitionTests(unittest.TestCase):
    def test_bald_distinguishes_disagreement(self):
        m=Toy(); x=torch.ones(3,2); xc=x[:0]; y=torch.empty(0)
        zero=core.score(m,xc,y,x,'bald',x,torch.zeros(8,2))
        disagreement=core.score(m,xc,y,x,'bald',x,torch.tensor([[-2.,0.],[2.,0.]]))
        self.assertTrue(torch.all(zero<1e-6))
        self.assertTrue(torch.all(disagreement>.1))

    def test_brier_change_formula(self):
        x=torch.ones(3,2)
        s=core.score(Toy(),x[:0],torch.empty(0),x,'expected_brier',x,torch.zeros(2,2))
        torch.testing.assert_close(s,torch.full((3,),.01))

    def test_stream_reveals_only_accepted_and_scores_nested_contexts(self):
        features=torch.ones(100,2); ids=np.arange(100); labels=np.arange(100)%2
        for policy in core.POLICIES:
            revealed=[]
            def reveal(i):
                self.assertLess(i,50); self.assertNotIn(i,revealed)
                revealed.append(i); return labels[i]
            row=core.trajectory(Toy(),features,ids,np.arange(50),np.arange(50,100),reveal,
                                labels[50:],policy,features[:3],features[:5],42)
            self.assertEqual(len(revealed),31)
            self.assertEqual(set(row['by_budget']),set(map(str,core.BUDGETS)))
            self.assertEqual(revealed,[p-1 for p in row['purchase_positions']])
            self.assertLessEqual(row['purchase_positions'][-1],50)

    def test_collapsed_score_ties_do_not_force_early_buys(self):
        outcomes=[core.accept(0.,np.zeros(32),1,100,u) for u in np.linspace(0,1,1000)]
        self.assertLess(sum(outcomes),20)
        self.assertTrue(core.accept(-1.,np.zeros(32),3,3,.99))

if __name__=='__main__':
    unittest.main()
