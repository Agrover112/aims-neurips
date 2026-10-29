import unittest
import numpy as np
from scipy.special import expit
from organized.training import bayesian_adapters as core

class BayesianAdapterTests(unittest.TestCase):
    def test_correction_symmetry_and_label_update(self):
        empty=np.array([])
        self.assertAlmostEqual(core.correction(empty,empty,np.array([0.]))[0],.5)
        positive=core.correction(np.zeros(3),np.ones(3),np.zeros(2))
        negative=core.correction(np.zeros(3),np.zeros(3),np.zeros(2))
        self.assertTrue(np.all(positive>.5))
        np.testing.assert_allclose(positive,1-negative,atol=1e-12)

    def test_zero_variance_correction_recovers_original(self):
        logits=np.array([-3.,0.,2.])
        result=core.correction(np.zeros(2),np.ones(2),logits,offset_sd=0.,slope_sd=0.)
        np.testing.assert_allclose(result,expit(logits),atol=1e-12)

    def test_gp_symmetry_and_semantic_locality(self):
        xc=np.zeros((3,2)); xq=np.array([[0.,0.],[10.,10.]])
        p=core.gp_predict(xc,np.ones(3),np.zeros(3),xq,np.zeros(2),offset_sd=.01)
        n=core.gp_predict(xc,np.zeros(3),np.zeros(3),xq,np.zeros(2),offset_sd=.01)
        np.testing.assert_allclose(p,1-n,atol=1e-8)
        self.assertGreater(p[0],p[1]+.1)
        self.assertAlmostEqual(p[1],.5,places=3)

    def test_gp_empty_duplicate_context_and_extreme_mean(self):
        for y in (np.array([]),np.ones(31)):
            xc=np.zeros((len(y),2)); mq=np.array([-30.,0.,30.])
            p=core.gp_predict(xc,y,np.zeros(len(y)),np.zeros((3,2)),mq)
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(np.all((p>=0)&(p<=1)))
            self.assertLess(p[0],p[1]); self.assertLess(p[1],p[2])

if __name__=='__main__':
    unittest.main()
