import unittest
import numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def test_initial(self):self.assertEqual(e.euler()[1][0],2.)
 def test_zero_rate(self):self.assertTrue(np.allclose(e.euler(k=0)[1],2))
 def test_invalid_n(self):self.assertRaises(ValueError,e.euler,n=0)
 def test_invalid_k(self):self.assertRaises(ValueError,e.euler,k=-1)
 def test_invalid_nan(self):self.assertRaises(ValueError,e.euler,k=np.nan)
 def test_closed_discrete(self):
  t,y,s=e.euler(n=20);self.assertAlmostEqual(y[-1],2*.9**20,places=13);self.assertAlmostEqual(s[-1],-4*.9**19,places=13)
 def test_reverse(self):self.assertAlmostEqual(e.reverse_gradient(1,2,2,20),e.euler()[2][-1],places=13)
 def test_autodiff_solver(self):
  y,g=e.autodiff_euler();self.assertAlmostEqual(y,e.euler()[1][-1],places=13);self.assertAlmostEqual(g,e.euler()[2][-1],places=13)
 def test_autodiff_shared_parameter(self):
  x=e.Value(3.);y=x*x+x;y.backward();self.assertAlmostEqual(y.value,12.);self.assertAlmostEqual(x.grad,7.)
 def test_finite_difference(self):self.assertAlmostEqual((e.euler(k=1+1e-6)[1][-1]-e.euler(k=1-1e-6)[1][-1])/2e-6,e.euler()[2][-1],places=8)
 def test_convergence(self):
  truth=e.exact(1,2,2);self.assertLess(abs(e.euler(n=100)[1][-1]-truth),abs(e.euler(n=20)[1][-1]-truth))
 def test_rk4(self):self.assertLess(abs(e.rk4(1,2,2,20)[1][-1]-e.exact(1,2,2)),1e-6)
 def test_fit(self):self.assertAlmostEqual(e.fit(e.exact(1,2,2),n=32)[0],16*(1-np.exp(-2/32)),places=4)
 def test_heat_guard(self):self.assertRaises(ValueError,e.heat_explicit,np.ones(5),1.,.1,.1,2)
 def test_heat_boundary(self):
  u=e.heat_explicit(np.ones(5),1.,.1,.001,10);self.assertEqual(u[0],0);self.assertEqual(u[-1],0)
if __name__=='__main__':unittest.main()
