"""检查网络Jacobian、固定核更新、参数化及真实训练。"""
import unittest
import numpy as np
from experiment import initialize,forward_jacobian,train
from generate_data import generate

class Tests(unittest.TestCase):
    def setUp(self):self.theta=initialize(4,17);self.x=np.array([-.8,.1,.7])
    def test_jacobian(self):
        _,j,_=forward_jacobian(self.theta,self.x);e=1e-6
        for k in range(len(self.theta)):
            delta=np.eye(len(self.theta))[k]*e
            g=(forward_jacobian(self.theta+delta,self.x)[0]-forward_jacobian(self.theta-delta,self.x)[0])/(2*e)
            np.testing.assert_allclose(j[:,k],g,atol=1e-9)
    def test_shapes(self):
        f,j,h=forward_jacobian(self.theta,self.x);self.assertEqual(f.shape,(3,));self.assertEqual(j.shape,(3,12));self.assertEqual(h.shape,(3,4))
    def test_gram_psd(self):
        _,j,_=forward_jacobian(self.theta,self.x);k=j@j.T;np.testing.assert_allclose(k,k.T);self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-12)
    def test_kernel_update(self):
        f,j,_=forward_jacobian(self.theta,self.x);y=np.array([1.,2.,3.]);step=-.01*j.T@(f-y)/3
        np.testing.assert_allclose(f+j@step,f-.01*(j@j.T)@(f-y)/3,atol=1e-14)
    def test_taylor_order(self):
        f,j,_=forward_jacobian(self.theta,self.x);d=np.ones_like(self.theta)
        errors=[np.linalg.norm(forward_jacobian(self.theta+s*d,self.x)[0]-f-s*j@d) for s in [.01,.005]]
        self.assertGreater(errors[0]/errors[1],3.8);self.assertLess(errors[0]/errors[1],4.2)
    def test_repeat(self):np.testing.assert_array_equal(initialize(4,17),initialize(4,17))
    def test_initial_match(self):
        row,a=train(8,1.,11,generate(),steps=0);self.assertEqual(row['relative_prediction_rmse'],0.)
    def test_training(self):
        row,a=train(8,1.,11,generate(),steps=30);self.assertLess(a['history'][-1,1],a['history'][0,1]);self.assertLess(a['history'][-1,2],a['history'][0,2])
    def test_reloaded_predict(self):
        data=generate();row,a=train(8,1.,11,data,steps=10)
        np.testing.assert_allclose(forward_jacobian(a['theta'],data['grid'])[0],a['nonlinear_grid'],atol=1e-14)
    def test_no_mutation(self):
        before=self.theta.copy();forward_jacobian(self.theta,self.x);np.testing.assert_array_equal(before,self.theta)
    def test_spectral_eta(self):
        row,a=train(8,.2,11,generate(),steps=10);self.assertAlmostEqual(row['eta']*row['initial_max_eigenvalue'],.3)

if __name__=='__main__':unittest.main()
