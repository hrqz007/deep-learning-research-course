import unittest,numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):self.t=np.array([.2,-.4,.7])
 def test_sigmoid(self):self.assertEqual(float(e.sigmoid(0)),.5)
 def test_extreme(self):self.assertTrue(np.isfinite(e.sigmoid([-1000,1000])).all())
 def test_probabilities(self):self.assertAlmostEqual(sum(r['probability'] for r in e.enumerate_mdp(self.t)),1)
 def test_four_trajectories(self):self.assertEqual(len(e.enumerate_mdp(self.t)),4)
 def test_score_mean_zero(self):np.testing.assert_allclose(sum((r['probability']*r['score'] for r in e.enumerate_mdp(self.t))),0,atol=1e-14)
 def test_mdp_gradient(self):
  _,g=e.mdp_exact(self.t)
  for i in range(3):
   d=np.zeros(3);d[i]=1e-5;self.assertAlmostEqual(g[i],(e.mdp_exact(self.t+d)[0]-e.mdp_exact(self.t-d)[0])/2e-5,places=8)
 def test_bandit_gradient(self):self.assertAlmostEqual(e.bandit_exact(0)[1],.15)
 def test_baseline_expectation(self):
  rows=e.enumerate_mdp(self.t);g=sum((r['probability']*(r['return']-.37)*r['score'] for r in rows));np.testing.assert_allclose(g,e.mdp_exact(self.t)[1])
 def test_mc(self):
  a=e.mdp_mc(self.t);np.testing.assert_array_less(np.abs(a.mean(0)-e.mdp_exact(self.t)[1]),5*a.std(0)/np.sqrt(len(a)))
 def test_discount_zero(self):self.assertAlmostEqual(e.mdp_exact(self.t,0)[0],-.1*float(e.sigmoid(.2)))
 def test_improvement(self):
  t=self.t.copy();v=e.mdp_exact(t)[0]
  for _ in range(50):t+=e.mdp_exact(t)[1]
  self.assertGreater(e.mdp_exact(t)[0],v)
 def test_reproducible(self):np.testing.assert_array_equal(e.mdp_mc(self.t,50),e.mdp_mc(self.t,50))
 def test_off_policy(self):self.assertLess(abs(e.importance_bandit(.3,-1.3)['importance_corrected']-e.bandit_exact(.3)[1]),.01)
if __name__=='__main__':unittest.main()
