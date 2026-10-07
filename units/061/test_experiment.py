"""Behavioral and mathematical checks; works unchanged under python -O."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import experiment as e
class TestGAN(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(2)
    def test_data_reproducible(self):
        a,l=e.make_data(32,8);b,m=e.make_data(32,8);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(l,m);self.assertEqual(a.shape,(32,2))
    def test_data_rejects_invalid(self):
        for n in [0,-1,1.2,True]:
            with self.assertRaises(ValueError):e.make_data(n)
    def test_metrics_reference(self):
        a,_=e.make_data();m=e.mode_metrics(a);self.assertEqual(m['coverage'],8);self.assertGreater(m['valid_fraction'],.99)
    def test_metrics_constant_control(self):
        m=e.mode_metrics(np.repeat(e.centers()[:1],100,0));self.assertEqual(m['coverage'],1);self.assertEqual(m['valid_fraction'],1);self.assertEqual(m['conditional_entropy'],0)
    def test_metrics_invalid_points_stay_in_denominator(self):
        a=np.zeros((4096,2),dtype='float32')
        for i,c in enumerate(e.centers()):a[i*41:(i+1)*41]=c
        m=e.mode_metrics(a);self.assertEqual(m['coverage'],8);self.assertAlmostEqual(m['valid_fraction'],328/4096)
    def test_metrics_rejects_invalid(self):
        for a in [np.zeros((0,2)),np.zeros((2,3)),np.full((2,2),np.nan)]:
            with self.assertRaises(ValueError):e.mode_metrics(a)
    def test_loss_scaling(self):
        logits=torch.zeros(4);v=F.softplus(-logits).mean()+F.softplus(logits).mean();self.assertAlmostEqual(float(v),2*np.log(2),places=6)
    def test_stable_extreme_logits(self):
        x=torch.tensor([-1000.,1000.],requires_grad=True);v=F.softplus(x).mean();v.backward();self.assertTrue(bool(torch.isfinite(v)));self.assertTrue(bool(torch.isfinite(x.grad).all()))
    def test_gradient_formula(self):
        s=torch.tensor([-4.,0.,3.],dtype=torch.float64,requires_grad=True);d=s.sigmoid();a=torch.log1p(-d).sum();grad=torch.autograd.grad(a,s)[0];torch.testing.assert_close(grad,-d.detach())
        s2=s.detach().clone().requires_grad_();v=F.softplus(-s2).sum();v.backward();torch.testing.assert_close(s2.grad,s2.detach().sigmoid()-1)
    def test_d_step_only_updates_d(self):
        g,d=e.Generator(),e.Discriminator();before=[p.clone() for p in g.parameters()];db=[p.clone() for p in d.parameters()];o=torch.optim.Adam(d.parameters(),lr=.01);e.discriminator_step(g,d,o,torch.randn(8,2),torch.randn(8,2))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,g.parameters())));self.assertTrue(all(p.grad is None for p in g.parameters()));self.assertTrue(any(not torch.equal(a,b) for a,b in zip(db,d.parameters())))
    def test_g_step_only_updates_g(self):
        g,d=e.Generator(),e.Discriminator();gb=[p.clone() for p in g.parameters()];db=[p.clone() for p in d.parameters()];o=torch.optim.Adam(g.parameters(),lr=.01);e.generator_step(g,d,o,torch.randn(8,2))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(db,d.parameters())));self.assertTrue(any(not torch.equal(a,b) for a,b in zip(gb,g.parameters())));self.assertTrue(all(p.requires_grad for p in d.parameters()))
    def test_optimal_discriminator(self):
        p=np.array([.5,.5]);q=np.array([1.,0.]);d=p/(p+q);v=(p*np.log(d)).sum()+np.log(1-d[0]);m=(p+q)/2;js=.5*(p*np.log(p/m)).sum()+.5*np.log(1/m[0]);self.assertAlmostEqual(v,-np.log(4)+2*js,places=12)
    def test_short_training_is_reproducible(self):
        a=e.train(4,4);b=e.train(4,4);self.assertEqual(a['final'],b['final'])
if __name__=='__main__':unittest.main()
