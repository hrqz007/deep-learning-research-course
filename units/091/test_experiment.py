import unittest
import numpy as np
import experiment as e
class TestDynamics(unittest.TestCase):
    def test_manual_step(self):np.testing.assert_allclose(e.true_step(np.array([1.,0.]),0),[1.,-.06])
    def test_shape(self):
        s,a=e.generate(1,3,10);self.assertEqual(s.shape,(3,11,2));self.assertEqual(a.shape,(3,10))
    def test_determinism(self):
        s,a=e.generate(1);s2,a2=e.generate(1);np.testing.assert_array_equal(s,s2)
    def test_gradient(self):
        s,a=e.generate(2,3,8);w=e.fit_one_step(s,a);s0,aa,t=e.windows(s,a,3);_,g=e.multistep_loss_grad(w,s0,aa,t)
        numeric=np.zeros_like(w);eps=1e-6
        for index in np.ndindex(w.shape):
            wp=w.copy();wm=w.copy();wp[index]+=eps;wm[index]-=eps
            numeric[index]=(e.multistep_loss_grad(wp,s0,aa,t)[0]-e.multistep_loss_grad(wm,s0,aa,t)[0])/(2*eps)
        np.testing.assert_allclose(g,numeric,atol=1e-7,rtol=1e-4)
    def test_rollout_true(self):
        s,a=e.generate(2,4,6);np.testing.assert_allclose(e.rollout(s[:,0],a),s[:,1:])
    def test_history_identity(self):
        s,a=e.generate(3,4,8)
        predicted=1.95*s[:,1:-1,0]-.95*s[:,:-2,0]+e.DT**2*(a[:,:-1]-.25*s[:,:-2,0]-.15*s[:,:-2,0]**3)
        np.testing.assert_allclose(predicted,s[:,2:,0],atol=1e-12)
    def test_candidates(self):
        self.assertEqual(e.candidate_actions(16).shape,(81,16));self.assertEqual(e.candidate_actions(1).shape,(9,1))
    def test_action_bounds(self):
        r=e.closed_loop([1.,0.],'one_step',np.array([[1.,-.06],[.15,.95],[0.,.15],[0.,0.]]),steps=5)
        self.assertLessEqual(np.abs(r['actions']).max(),1)
    def test_hidden_velocity(self):
        nxt=e.true_step(np.array([[1.,.5],[1.,-.5]]),np.zeros(2));self.assertAlmostEqual(nxt[0,0]-nxt[1,0],.15)
    def test_fit_finite(self):
        s,a=e.generate(2,4,8);w=e.fit_one_step(s,a);wm,loss=e.fit_multistep(w,s,a,steps=30)
        self.assertTrue(np.isfinite(wm).all());self.assertTrue(np.isfinite(loss).all())
if __name__=='__main__':unittest.main()
