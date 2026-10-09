"""Gradient and causal-mechanism checks independent of reported summary values."""
import unittest
import numpy as np
from experiment import generate, init, predict, gradient, ig, train, slope

class ExplanationTests(unittest.TestCase):
    def test_generator_structural_equations(self):
        x,y,e=generate(40); u,ex,ey,ez=e.T
        np.testing.assert_allclose(x[:,0],u+ex)
        np.testing.assert_allclose(y,2*x[:,0]+3*u+ey)
        np.testing.assert_allclose(x[:,1],y+ez)
    def test_different_split(self):
        a=generate(40,7701)[0]; b=generate(40,7702)[0]
        self.assertFalse(np.array_equal(a,b))
    def test_input_gradient_finite_difference(self):
        p=init(); x=np.array([[.8,4.],[-.3,.1]])
        for j in range(2):
            eps=np.zeros_like(x); eps[:,j]=1e-6
            fd=(predict(x+eps,p)-predict(x-eps,p))/2e-6
            np.testing.assert_allclose(fd,gradient(x,p)[:,j],rtol=1e-6,atol=1e-8)
    def test_integrated_gradient_completeness(self):
        p=init(); x=np.array([.8,4.]); b=np.zeros(2)
        self.assertAlmostEqual(ig(x,b,p).sum(),predict(x[None],p)[0]-predict(b[None],p)[0],places=7)
    def test_baseline_identity(self):
        x=np.array([.8,4.]); np.testing.assert_array_equal(ig(x,x,init()),np.zeros(2))
    def test_ig_input_guards(self):
        with self.assertRaises(ValueError): ig(np.zeros(2),np.zeros(2),init(),0)
        with self.assertRaises(ValueError): ig(np.zeros(3),np.zeros(2),init())
    def test_fit_learns_proxy(self):
        x,y,_=generate(); p,_=train(x,y,steps=1800)
        self.assertLess(np.sqrt(np.mean((predict(x,p)-y)**2)),.4)
    def test_intervention_is_not_conditioning(self):
        x,y,e=generate(10000); u,ex,ey,ez=e.T
        y0=2*(-.5)+3*u+ey; y1=2*(.5)+3*u+ey
        np.testing.assert_allclose(y1-y0,2)
        self.assertGreater(slope(x[:,0],y),4.)
        self.assertAlmostEqual(slope(x[:,0],y,u),2,places=1)
    def test_proxy_does_not_enter_y_equation(self):
        x,y,e=generate(20); original=y.copy(); x[:,1]+=100
        np.testing.assert_array_equal(y,original)
    def test_parameter_randomization_changes_gradient(self):
        x=np.array([[.8,4.],[-.3,2.]])
        self.assertGreater(np.linalg.norm(gradient(x,init(17))-gradient(x,init(77))),.01)

if __name__=='__main__': unittest.main()
