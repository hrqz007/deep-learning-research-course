"""Run: python tests.py. Includes numerical identities and saved-run acceptance."""
import unittest
import json
import numpy as np
from experiment import ROOT, MU, COV, SIGMA, score, fit, predict, sample
class ScoreTests(unittest.TestCase):
    def test_gradient_finite_difference(self):
        x=np.array([[1.2,-1.1]]); precision=np.linalg.inv(COV+SIGMA**2*np.eye(2))
        def logp(z): return -.5*np.einsum('ni,ij,nj->n',z-MU,precision,z-MU)
        grad=[]
        for j in range(2):
            d=np.zeros_like(x);d[0,j]=1e-5
            grad.append(float(((logp(x+d)-logp(x-d))/(2e-5))[0]))
        np.testing.assert_allclose(score(x)[0],grad,atol=1e-8)
    def test_target_identity(self):
        c,y,t=sample(np.random.default_rng(1),20)
        np.testing.assert_allclose(t,(c-y)/SIGMA**2,atol=1e-12)
    def test_exact_affine_recovery(self):
        x=np.random.default_rng(3).normal(size=(100,2));w=fit(x,score(x))
        np.testing.assert_allclose(predict(x,w),score(x),atol=1e-12)
    def test_saved_run(self):
        m=json.loads((ROOT/'outputs/metrics.json').read_text())
        self.assertLess(m['score_mse_per_coordinate'],.005)
        self.assertLess(m['coefficient_max_absolute_error'],.05)
        d=np.load(ROOT/'outputs/data.npz'); self.assertFalse(np.array_equal(d['test'][:100],d['noisy_train'][:100]))
        self.assertEqual(sum(v['n'] for v in m['groups'].values()),m['n_test'])
        self.assertTrue(np.isfinite(d['weights']).all())
if __name__=='__main__': unittest.main(verbosity=2)
