import unittest,json
from pathlib import Path
import numpy as np
import experiment as e
P=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def setUp(self):self.m=json.loads((P/'outputs/metrics.json').read_text())
 def test_records(self):
  self.assertEqual(len(self.m['records']),25)
  self.assertEqual(sum(r['held_out_scale'] for r in self.m['records']),5)
 def test_accounting(self):
  for r in self.m['records']:
   self.assertEqual(r['parameters'],6*r['width']+1);self.assertEqual(r['sample_presentations'],512000)
 def test_prediction_metrics(self):
  a=np.load(P/'outputs/predictions.npz');d=np.load(P/'outputs/data.npz')
  for r in self.m['records']:
   value=((a[f"w{r['width']}_s{r['seed']}"]-d['test_y'])**2).mean();self.assertAlmostEqual(float(value),r['test_mse'])
 def test_fit_excludes_holdout(self):
  a,b=e.fit_curve(self.m['parameters'][:-1],self.m['mean_test_mse'][:-1]);self.assertAlmostEqual(b,self.m['log_fit_slope']);self.assertAlmostEqual(a,self.m['log_fit_intercept'])
 def test_relative_error(self):
  pred=self.m['predicted_mse'][-1];obs=self.m['mean_test_mse'][-1];self.assertAlmostEqual((pred-obs)/obs,self.m['heldout_relative_error'])
 def test_analytic_power(self):
  a,b=e.fit_curve(np.array([1.,2.,4.]),np.array([1.,.5,.25]));self.assertAlmostEqual(a,0);self.assertAlmostEqual(b,-1)
 def test_split_and_finite(self):
  d=np.load(P/'outputs/data.npz');self.assertFalse(np.array_equal(d['train_x'],d['test_x']))
  for r in self.m['records']:self.assertTrue(np.isfinite(r['test_mse']) and r['test_mse']>0)
if __name__=='__main__':unittest.main(verbosity=2)
