import unittest,json
import numpy as np
from experiment import ROOT,V0,BETA,T,MU,variance,score,reverse,discrete_variance
class DiffusionTests(unittest.TestCase):
    def test_score(self):
        self.assertEqual(score(MU,.5),0)
        self.assertAlmostEqual(score(MU+1,0),-1/V0)
    def test_one_step_recurrence(self):
        a=1-BETA*T/variance(T)
        self.assertAlmostEqual(discrete_variance(1,'sde'),a*a*variance(T)+BETA*T)
    def test_ode_convergence(self):
        x=np.array([-2.,0.,3.]); exact=MU+(x-MU)*np.sqrt(V0/variance(T))
        e1=np.linalg.norm(reverse(x,20,'ode')-exact);e2=np.linalg.norm(reverse(x,200,'ode')-exact)
        self.assertLess(e2,e1/5)
    def test_variance_convergence(self):
        for kind in ('sde','ode'):
            self.assertLess(abs(discrete_variance(200,kind)-V0),abs(discrete_variance(20,kind)-V0)/5)
    def test_saved_distribution(self):
        m=json.loads((ROOT/'outputs/metrics.json').read_text())
        self.assertLess(abs(m['forward_final_variance']-variance(T)),.08)
        for r in m['results']:
            self.assertLess(abs(r['sample_variance']-r['discrete_theory_variance']),.025)
            if r['steps']==200:
                self.assertLess(r['ks_to_target'],.02)
                self.assertLess(abs(r['sample_mean']-MU),.03)
if __name__=='__main__':unittest.main(verbosity=2)
