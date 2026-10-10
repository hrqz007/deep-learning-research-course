import unittest,numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):self.X,self.y=e.fixture();self.w=np.array([1.2,-.3])
 def test_initial_dpo(self):self.assertAlmostEqual(e.objective(np.array([.8,0]),self.X,self.y)[0],np.log(2))
 def test_gradient(self):
  for method in ['sft','dpo']:
   _,g=e.objective(self.w,self.X,self.y,method)
   for i in range(2):
    d=np.zeros(2);d[i]=1e-5;self.assertAlmostEqual(g[i],(e.objective(self.w+d,self.X,self.y,method)[0]-e.objective(self.w-d,self.X,self.y,method)[0])/2e-5,places=8)
 def test_pair_swap(self):self.assertAlmostEqual(e.objective(self.w,self.X,self.y)[0],e.objective(self.w,-self.X,1-self.y)[0])
 def test_extreme(self):self.assertTrue(np.isfinite(e.objective(np.array([1000.,-1000.]),self.X,self.y)[0]))
 def test_beta(self):
  with self.assertRaises(ValueError):e.objective(self.w,self.X,self.y,beta=0)
 def test_dpo_improves(self):
  w,h=e.train(self.X,self.y);self.assertLess(h[-1][1],h[0][1])
 def test_sft_improves(self):
  w,h=e.train(self.X,self.y,'sft');self.assertLess(h[-1][1],h[0][1])
 def test_kl_nonnegative(self):self.assertGreaterEqual(e.evaluate(self.w,self.X,self.y)['mean_reference_kl'],0)
 def test_reference_kl(self):self.assertAlmostEqual(e.evaluate(np.array([.8,0.]),self.X,self.y)['mean_reference_kl'],0)
 def test_fixture_repeat(self):np.testing.assert_array_equal(e.fixture()[0],self.X)
 def test_conflict(self):
  X,y=e.fixture(correlation=0);np.testing.assert_array_equal(X[:,0],-X[:,1])
 def test_prob_bounds(self):
  m=e.evaluate(self.w,self.X,self.y)
  for k,v in m.items():self.assertGreaterEqual(v,0);self.assertLessEqual(v,1 if k!='mean_reference_kl' else 100)
 def test_train_repeat(self):np.testing.assert_array_equal(e.train(self.X,self.y)[0],e.train(self.X,self.y)[0])
if __name__=='__main__':unittest.main()
