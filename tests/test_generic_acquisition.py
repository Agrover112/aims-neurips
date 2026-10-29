import unittest
import numpy as np
import torch
from organized.acquisition.expected_brier import expected_brier_scores, streaming_accept
from organized.training.anp_acquisition import score, accept
from organized.training.anp_crossattn_blocks import AttentiveNeuralProcess

class GenericAcquisitionTests(unittest.TestCase):
    def test_exchangeable_beta_gives_equal_scores(self):
        def predict(xc,yc,xt):
            return np.full(len(xt),(2+yc.sum())/(4+len(yc)))
        s=expected_brier_scores(predict,np.empty((0,2)),np.empty(0),np.array([[0.,0.],[10.,20.]]),np.ones((4,2)))
        np.testing.assert_allclose(s,[.01,.01],atol=1e-12)

    def test_generic_matches_torch_anp_and_pacing(self):
        torch.set_num_threads(2); torch.manual_seed(3)
        model=AttentiveNeuralProcess(4,width=8,self_attention_blocks=1,cross_attention_heads=2,latent_dim=2).eval()
        xc=torch.randn(3,4); yc=torch.tensor([0.,1.,0.]); xt=torch.randn(2,4); ref=torch.randn(5,4)
        def predict(a,b,q):
            with torch.inference_mode():
                return model.logits(torch.tensor(a,dtype=torch.float32),torch.tensor(b,dtype=torch.float32),torch.tensor(q,dtype=torch.float32)).sigmoid().numpy()
        actual=expected_brier_scores(predict,xc.numpy(),yc.numpy(),xt.numpy(),ref.numpy())
        expected=score(model,xc,yc,xt,'expected_brier',ref,None).numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-8)
        for u in (.01,.5,.99):
            self.assertEqual(streaming_accept(actual[0],actual,7,20,u),accept(actual[0],actual,7,20,u))

    def test_invalid_callback_and_no_budget(self):
        with self.assertRaises(ValueError):
            expected_brier_scores(lambda a,b,c:np.full(len(c),np.nan),np.empty((0,2)),np.empty(0),np.ones((1,2)),np.ones((2,2)))
        self.assertFalse(streaming_accept(0,[0],0,5,.5))
        self.assertTrue(streaming_accept(0,[0],3,3,.5))

if __name__=='__main__':
    unittest.main()
