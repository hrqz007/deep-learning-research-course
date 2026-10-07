import unittest
import numpy as np
import torch
import experiment as e
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_data_reproducible(self):
        a=e.make_data();b=e.make_data()
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_data_shapes(self):
        d=e.make_data();self.assertEqual(d['train'].shape,(1024,64));self.assertEqual(d['test'].shape,(512,64));self.assertTrue(np.isin(d['test'],[0,1]).all())
    def test_templates(self):
        t=e.templates();self.assertEqual(t.shape,(12,64));np.testing.assert_array_equal(t.sum(1),8)
    def test_warmup_endpoints(self):
        self.assertEqual(e.beta_at(1,'warmup'),.002);self.assertEqual(e.beta_at(500,'warmup'),1);self.assertEqual(e.beta_at(999,'warmup'),1)
    def test_constant_beta(self):
        self.assertEqual(e.beta_at(1,'constant'),1);self.assertEqual(e.beta_at(200,'long_constant'),1)
    def test_invalid_beta(self):
        for a in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):e.beta_at(a,'warmup')
        with self.assertRaises(ValueError):e.beta_at(1,'other')
    def test_kl_zero(self):
        x=torch.zeros(3,64);z=torch.zeros(3,4);n,k=e.terms(x,x,z,z);self.assertTrue(torch.all(k==0));np.testing.assert_allclose(n.numpy(),64*np.log(2),rtol=1e-6)
    def test_kl_shift(self):
        x=torch.zeros(2,64);mu=torch.ones(2,4);lv=torch.zeros_like(mu);n,k=e.terms(x,x,mu,lv);np.testing.assert_allclose(k.numpy(),2)
    def test_coordinate_sum(self):
        x=torch.ones(2,64);z=torch.zeros(2,4);n,k=e.terms(torch.zeros_like(x),x,z,z);self.assertAlmostEqual(float(n[0]),64*np.log(2),places=4)
    def test_invalid_targets(self):
        with self.assertRaises(ValueError):e.terms(torch.zeros(1,64),torch.full((1,64),2.),torch.zeros(1,4),torch.zeros(1,4))
    def test_gradient(self):
        m=e.VAE();x=torch.zeros(5,64);p,mu,lv=m(x,torch.randn(5,4));n,k=e.terms(p,x,mu,lv);(n+k).mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_evaluate_reproducible(self):
        m=e.VAE();x=torch.zeros(6,64);a,aa=e.evaluate(m,x,2);b,bb=e.evaluate(m,x,2);self.assertEqual(a,b);np.testing.assert_array_equal(aa['negative_elbo'],bb['negative_elbo'])
    def test_elbo_decomposition(self):
        m=e.VAE();a,_=e.evaluate(m,torch.zeros(8,64),2);self.assertAlmostEqual(a['negative_elbo'],a['nll']+a['kl'],places=4)
    def test_template_metrics(self):
        r=e.sample_metrics(e.templates());self.assertEqual(r['coverage'],12);self.assertEqual(r['template_valid_fraction'],1)
    def test_empty_images_invalid(self):
        with self.assertRaises(ValueError):e.sample_metrics(np.zeros((0,64)))
    def test_binary_samples(self):
        a,p=e.sample_model(e.VAE(),1,20);self.assertTrue(np.isin(a,[0,1]).all());self.assertTrue(((p>0)&(p<1)).all())
    def test_checkpoint_sampling(self):
        a=e.VAE();b=e.VAE();b.load_state_dict(a.state_dict());xa,pa=e.sample_model(a,3,20);xb,pb=e.sample_model(b,3,20);np.testing.assert_array_equal(xa,xb);np.testing.assert_array_equal(pa,pb)
    def test_equal_initial_and_long_prefix(self):
        x=torch.from_numpy(e.make_data()['train']);a,ar=e.train('constant',11,x,steps=2);b,br=e.train('long_constant',11,x,steps=1)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    def test_warmup_changes_weights(self):
        x=torch.from_numpy(e.make_data()['train']);a,_=e.train('constant',11,x,steps=3);b,_=e.train('warmup',11,x,steps=3);self.assertTrue(any(not torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
    def test_invalid_train_steps(self):
        with self.assertRaises(ValueError):e.train('constant',1,torch.zeros(3,64),0)
if __name__=='__main__':unittest.main()
