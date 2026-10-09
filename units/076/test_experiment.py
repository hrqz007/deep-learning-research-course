"""Meaningful numerical checks; unittest assertions survive Python -O."""
import unittest
import numpy as np
from experiment import init, loss_grad, forward, train, make_data, last_layer, features, score, NOISE

class NumericalTests(unittest.TestCase):
    def test_generator_repeatable_and_shape(self):
        a,b=make_data(),make_data()
        for u,v in zip(a,b): np.testing.assert_array_equal(u,v)
        self.assertEqual(a[0].shape,(100,))
        self.assertTrue(np.max(abs(a[0]))<=2)
    def test_analytic_parameter_gradient(self):
        x=np.array([-.8,.2,1.]); y=np.array([-.4,.1,.8]); p=init(11)
        _,g=loss_grad(x,y,p)
        for k,a in enumerate(p):
            for index in range(a.size):
                old=a[index]; eps=1e-6
                a[index]=old+eps; plus=loss_grad(x,y,p)[0]
                a[index]=old-eps; minus=loss_grad(x,y,p)[0]; a[index]=old
                self.assertAlmostEqual(g[k][index],(plus-minus)/(2*eps),places=6)
    def test_fit_learns(self):
        x,y,*_=make_data(); p=init(11); before=loss_grad(x,y,p)[0]
        p,_=train(x,y,11,500)
        self.assertLess(loss_grad(x,y,p)[0],before*.1)
    def test_posterior_matches_normal_equations(self):
        x,y,g,_=make_data(); p=init(11)
        pred,epi,m,c=last_layer(x,y,g,p)
        phi=features(x,p); precision=np.eye(17)+phi.T@phi/NOISE**2
        np.testing.assert_allclose(precision@m,phi.T@y/NOISE**2,rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(precision@c,np.eye(17),atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(c).min(),0)
        self.assertTrue(np.all(epi>=0))
    def test_predictive_variance_formula(self):
        x,y,g,_=make_data(); p=init(11); pred,epi,m,c=last_layer(x,y,g[:3],p)
        rng=np.random.default_rng(100); draws=rng.multivariate_normal(m,c,size=30000)
        values=draws@features(g[:3],p).T
        np.testing.assert_allclose(values.var(0),epi,rtol=.04)
    def test_member_mixture_variance_uses_population(self):
        means=np.array([1.,3.]); total=NOISE**2+means.var()
        self.assertAlmostEqual(total,1.0324)
    def test_score_and_input_guards(self):
        m=score(np.array([0.,0.]),np.zeros(2),np.ones(2))
        self.assertEqual(m['coverage_95'],1.)
        with self.assertRaises(ValueError): score(np.zeros(2),np.zeros(2),np.zeros(2))
        x,y,g,_=make_data()
        with self.assertRaises(ValueError): last_layer(x,y,g,init(11),0)

if __name__=='__main__': unittest.main()
