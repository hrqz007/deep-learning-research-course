import unittest,json,hashlib
import numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def test_source_integrity(self):
  p=json.loads((e.ROOT/'data/provenance.json').read_text());raw=(e.ROOT/'data/Misra1a.dat').read_bytes()
  self.assertEqual(len(raw),p['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),p['sha256'])
  x,y=e.read_real();self.assertEqual(len(x),14);self.assertEqual(x[-1],760.);self.assertEqual(y[0],10.07)
 def test_noiseless_recovery(self):
  x=np.linspace(.01,5,30);theta,_=e.fit(x,e.forward(x,2.4,.55));np.testing.assert_allclose(theta,[2.4,.55],rtol=1e-7)
 def test_sensitivity(self):
  x=np.linspace(.01,2,12);theta=np.array([2.4,.55]);j=e.sensitivity(x,*theta);eps=1e-6
  for k in range(2):
   p=theta.copy();m=theta.copy();p[k]*=np.exp(eps);m[k]*=np.exp(-eps)
   np.testing.assert_allclose((e.forward(x,*p)-e.forward(x,*m))/(2*eps),j[:,k],rtol=1e-8)
 def test_certified_nist(self):
  p,v=e.read_real();t,sse=e.fit(p/1000,v/100)
  np.testing.assert_allclose(t*[100,.001],[238.94212918,.00055015643181],rtol=1e-7)
  self.assertAlmostEqual(sse*10000,.12455138894,places=8)
 def test_gain_nonidentifiability(self):
  x=np.linspace(0,5,100);np.testing.assert_array_equal(e.forward(x,2.4,.55),2*e.forward(x,1.2,.55))
 def test_split_and_results(self):
  m=json.loads((e.ROOT/'outputs/metrics.json').read_text());self.assertEqual(m['real']['heldout_high_pressure_n'],4)
  self.assertLess(m['real']['heldout_saturation_rmse'],m['real']['heldout_linear_rmse'])
  self.assertGreater(m['synthetic']['narrow']['sensitivity_condition'],m['synthetic']['wide']['sensitivity_condition']*50)

class RobustnessTests(unittest.TestCase):
 def test_invalid_dimensions(self):
  for x,y in [(np.ones((2,2)),np.ones(4)),(np.array([1.,2.]),np.ones(3)),(np.array([1.]),np.array([1.]))]:
   with self.assertRaises(ValueError):e.fit(x,y)
 def test_invalid_observations(self):
  for x,y in [(np.array([0.,np.nan]),np.ones(2)),(np.array([0.,1.]),np.array([0.,np.inf])),(np.array([-1.,2.]),np.ones(2)),(np.array([0.,0.]),np.ones(2))]:
   with self.assertRaises(ValueError):e.fit(x,y)
 def test_invalid_regularization(self):
  for lam in [-1.,np.nan,np.inf]:
   with self.assertRaises(ValueError):e.fit(np.array([0.,1.]),np.ones(2),lam=lam)
 def test_model_constraints(self):
  x=np.linspace(0,100,1001);u=e.forward(x,2.4,.55)
  self.assertEqual(u[0],0.);self.assertTrue(np.all(np.diff(u)>=0));self.assertTrue(np.all(u<=2.4));self.assertTrue(np.isfinite(u).all())
 def test_scaling_identity(self):
  p=np.array([77.6,190.8,760.]);a,b=2.3894212918,.55015643181
  np.testing.assert_allclose(100*e.forward(p/1000,a,b),e.forward(p,100*a,b/1000),rtol=1e-14)
 def test_regularized_profile_gradient(self):
  x=np.linspace(.005,.05,24);y=e.forward(x,2.4,.55);lam=.002;a0=1.2;b=.8;g=-np.expm1(-b*x);a=(g@y+lam*a0)/(g@g+lam)
  derivative=2*g@(a*g-y)+2*lam*(a-a0);self.assertAlmostEqual(derivative,0.,places=13)
 def test_saved_outputs_finite(self):
  for file,shape in [('real_predictions.csv',(14,5)),('real_bootstrap.csv',(300,2)),('fits_narrow.csv',(200,2)),('fits_wide.csv',(200,2))]:
   data=np.loadtxt(e.ROOT/'outputs'/file,delimiter=',',skiprows=1);self.assertEqual(data.shape,shape);self.assertTrue(np.isfinite(data).all())

if __name__=='__main__':unittest.main()
