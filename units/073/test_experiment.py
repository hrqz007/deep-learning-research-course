"""独立数值恒等式和边界测试；不用可被-O删除的assert。"""
import unittest
import numpy as np
from experiment import logistic,descent_quadratic,train_logistic
from generate_data import generate

class Tests(unittest.TestCase):
    def setUp(self): self.d=generate()
    def test_gradient(self):
        w=np.array([.3,-.8]);_,g=logistic(w,self.d['x'],self.d['y']);e=1e-6
        numeric=[(logistic(w+e*np.eye(2)[j],self.d['x'],self.d['y'])[0]-logistic(w-e*np.eye(2)[j],self.d['x'],self.d['y'])[0])/(2*e) for j in range(2)]
        np.testing.assert_allclose(g,numeric,atol=1e-9)
    def test_stability(self):
        loss,g=logistic(np.array([1e4,1e4]),self.d['x'],self.d['y'])
        self.assertTrue(np.isfinite(loss));self.assertTrue(np.isfinite(g).all())
    def test_zero_loss(self):
        loss, grad = logistic(np.zeros(2),self.d['x'],self.d['y']);self.assertAlmostEqual(loss,np.log(2))
        np.testing.assert_allclose(grad,[-.55,-.6],atol=1e-14)
    def test_hard_margin(self):
        z=self.d['y'][:,None]*self.d['x'];np.testing.assert_array_less(.999999999,z@np.array([1.,.5]))
        # 所有可行w必须w0>=1,w1>=.5，因此[1,.5]确为最小范数解。
        self.assertTrue(any(np.array_equal(v,[1.,0.]) for v in z));self.assertTrue(any(np.array_equal(v,[0.,2.]) for v in z))
    def test_bound(self):
        _,f=descent_quadratic(self.d['q'],self.d['w0'],.25,40)
        self.assertTrue(np.all(f[1:]<=40/np.arange(1,41)+1e-12))
    def test_monotone(self):
        _,f=descent_quadratic(self.d['q'],self.d['w0'],.25,40);self.assertTrue(np.all(np.diff(f)<=1e-12))
    def test_overshoot(self):
        _,f=descent_quadratic(self.d['q'],self.d['w0'],.6,20);self.assertGreater(f[-1],f[0])
    def test_closed_form(self):
        w,f=descent_quadratic(self.d['q'],self.d['w0'],.25,2)
        np.testing.assert_allclose(w[1],[3.,0.]);self.assertAlmostEqual(f[1],4.5)
    def test_no_input_mutation(self):
        w=self.d['w0'].copy();descent_quadratic(self.d['q'],w,.25,3);np.testing.assert_array_equal(w,self.d['w0'])
    def test_direction(self):
        r,a=train_logistic(self.d['x'],self.d['y'],2000)
        for run in r['runs']:self.assertLess(run['final_angle_deg'],run['angle_at_10']);self.assertEqual(run['final_error'],0.)
    def test_reproducibility(self):
        np.testing.assert_array_equal(generate()['x'],generate()['x'])

if __name__=='__main__':unittest.main()
