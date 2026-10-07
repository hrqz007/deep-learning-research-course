import unittest, tempfile
from pathlib import Path
import numpy as np
import torch
import experiment as e
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_data_reproducible(self):
        a,y=e.make_data();b,z=e.make_data();np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(y,z)
    def test_disjoint_data(self):
        a,_=e.make_data(128,630);b,_=e.make_data(128,631);self.assertFalse(np.array_equal(a,b))
    def test_bad_data(self):
        for n in [0,-1,1.5,True]:
            with self.assertRaises(ValueError):e.make_data(n)
    def test_quality_coverage(self):
        a=np.repeat(e.centers(),20,axis=0);r=e.mode_metrics(a);self.assertEqual(r['coverage'],8);self.assertEqual(r['valid_fraction'],1)
    def test_collapse(self):
        r=e.mode_metrics(np.repeat(e.centers()[:1],100,axis=0));self.assertEqual(r['coverage'],1);self.assertEqual(r['entropy'],0)
    def test_denominator_all_samples(self):
        x=np.concatenate([e.centers()[:1],np.full((199,2),100.)]);r=e.mode_metrics(x);self.assertEqual(r['coverage'],0);self.assertEqual(r['valid_fraction'],.005)
    def test_invalid_metric(self):
        for x in [[],[[float('nan'),0]],[[1,2,3]]]:
            with self.assertRaises(ValueError):e.mode_metrics(x)
    def test_distance_identity(self):
        x,_=e.make_data(100);self.assertLess(e.gaussian_feature_distance(x,x),1e-10)
    def test_distance_symmetry(self):
        x,_=e.make_data(100);y,_=e.make_data(100,22);self.assertAlmostEqual(e.gaussian_feature_distance(x,y),e.gaussian_feature_distance(y,x),places=10)
    def test_distance_translation(self):
        x,_=e.make_data(100);self.assertAlmostEqual(e.gaussian_feature_distance(x,x+[2,3]),13,places=8)
    def test_distance_scale(self):
        x,_=e.make_data(100);y,_=e.make_data(100,22);self.assertAlmostEqual(e.gaussian_feature_distance(10*x,10*y),100*e.gaussian_feature_distance(x,y),places=7)
    def test_psd_sqrt(self):
        a=np.array([[2.,1.],[1.,2.]]);b=e.psd_sqrt(a);np.testing.assert_allclose(b@b,a,atol=1e-10)
    def test_blind_feature(self):
        x=e.centers().astype(float);a=np.pi/8;r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);y=x@r.T
        self.assertLess(e.gaussian_feature_distance(x,y),1e-12);self.assertEqual(e.mode_metrics(y)['valid_fraction'],0)
    def test_nn_exact(self):
        x,_=e.make_data(30);np.testing.assert_allclose(e.nearest_distance(x,x),0)
    def test_equal_reference_size(self):
        with self.assertRaises(ValueError):e.audit([[0,0]],np.zeros((2,2)),np.zeros((3,2)))
    def test_copier_audit(self):
        x,_=e.make_data(30);y,_=e.make_data(30,1);r,_=e.audit(x,x,y);self.assertEqual(r['exact_train_fraction'],1);self.assertEqual(r['exact_holdout_fraction'],0)
    def test_vae_terms(self):
        z=torch.zeros(2,2);nll,kl=e.vae_terms(z,z,z,z);self.assertTrue(torch.all(kl==0));self.assertEqual(tuple(nll.shape),(2,))
    def test_vae_gradients(self):
        m=e.VAE();x=torch.randn(8,2);p,mu,lv=m(x,torch.randn_like(x));n,k=e.vae_terms(p,x,mu,lv);(n+k).mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_checkpoint_sampling(self):
        for family,cls in [('vae',e.VAE),('gan',e.Generator)]:
            m=cls();a,_=e.sample_model(m,family,1,20);n=cls();n.load_state_dict(m.state_dict());b,_=e.sample_model(n,family,1,20);np.testing.assert_array_equal(a,b)
    def test_real_training(self):
        x,_=e.make_data(32)
        for fam in ['vae','gan']:
            m,d,r=e.train(fam,7,x,2);self.assertEqual(r['history'][-1]['step'],2);self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
if __name__=='__main__':unittest.main()
