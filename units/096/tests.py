"""独立数值断言；运行 python tests.py。先执行 experiment.py 生成证据。"""
import unittest, json
from pathlib import Path
import numpy as np
import experiment as e
P=Path(__file__).resolve().parent
class TestExperiment(unittest.TestCase):
    def test_mixture_symmetry(self):
        x=np.array([-.7,0.,.7]);np.testing.assert_allclose(e.score_uncond(x,1),-e.score_uncond(-x,1),atol=1e-12)
    def test_conditional_value(self):
        self.assertAlmostEqual(float(e.score_cond(0,1,1)),2/1.36)
    def test_score_finite_difference(self):
        def logp(x):return np.log(.5*np.exp(-(x-2)**2/(2*1.36))+.5*np.exp(-(x+2)**2/(2*1.36)))
        x=.7;h=1e-5
        self.assertAlmostEqual(float(e.score_uncond(x,1)),float((logp(x+h)-logp(x-h))/(2*h)),places=7)
    def test_compression_and_generation(self):
        a=np.load(P/'outputs/data.npz');err=np.mean((a['observation']-a['reconstruction'])**2)
        self.assertTrue(.035<err<.055);self.assertEqual(a['w4'].shape,(4000,));self.assertTrue(np.isfinite(a['w4']).all())
        self.assertGreater(float(np.mean(a['w1']>0)),.98)

if __name__=='__main__': unittest.main(verbosity=2)
