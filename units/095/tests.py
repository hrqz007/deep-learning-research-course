"""独立数值断言；运行 python tests.py。先执行 experiment.py 生成证据。"""
import unittest, json
from pathlib import Path
import numpy as np
import experiment as e
P=Path(__file__).resolve().parent
class TestExperiment(unittest.TestCase):
    def test_density_normalizes(self):
        x=np.linspace(-5,9,50000); z=(x-2)/.7
        p=np.exp(-.5*z*z)/(.7*np.sqrt(2*np.pi))
        self.assertAlmostEqual(float(np.trapezoid(p,x)),1,places=6)
    def test_constant_velocity(self):
        x=e.integrate(np.array([0.]),20,np.array([0.,1.]),np.array([[0.,2.],[0.,2.]]))
        self.assertAlmostEqual(float(x[0]),2)
    def test_velocity_endpoints(self):
        grid=np.array([0.,1.]); sigma=.7; mu=2.
        a=(-(1-grid)+grid*sigma**2)/((1-grid)**2+(grid*sigma)**2)
        b=mu-a*grid*mu
        np.testing.assert_allclose(a,[-1.,1.]);np.testing.assert_allclose(b,[2.,0.])
        fitted=np.load(P/'outputs/data.npz')['coefficients'][[0,-1]]
        np.testing.assert_allclose(fitted,np.column_stack([a,b]),atol=.2)
    def test_output(self):
        a=np.load(P/'outputs/data.npz');self.assertEqual(a['coefficients'].shape,(51,2))
        self.assertTrue(np.isfinite(a['flow_160']).all())
        self.assertLess(float(np.sqrt(np.mean((a['flow_160']-a['affine'])**2))),.03)
    def test_affine_reference(self):
        a=np.load(P/'outputs/data.npz');self.assertEqual(len(a['base']),10000)
        np.testing.assert_allclose(a['affine'],2+.7*a['base'])

if __name__=='__main__': unittest.main(verbosity=2)
