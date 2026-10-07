"""Formula, indexing and actual sampler tests; also execute with python -O."""
import unittest
import numpy as np
import torch
from torch import nn
import experiment as e
class Recorder(nn.Module):
    def __init__(self):super().__init__();self.seen=[]
    def forward(self,x,t):self.seen.append(int(t[0]));return torch.zeros_like(x)
class TestDDPM(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(3);self.s=e.schedule()
    def test_schedule_endpoints(self):
        s=self.s;self.assertEqual(len(s['beta']),201);self.assertEqual(float(s['beta'][0]),0);self.assertEqual(float(s['abar'][0]),1);self.assertEqual(float(s['posterior_var'][1]),0);self.assertTrue(bool((s['abar'][1:]<s['abar'][:-1]).all()));self.assertLess(float(s['abar'][-1]),.007)
    def test_schedule_invalid(self):
        for args in [(0,.1,.2),(2,0,.2),(2,.2,.1),(2,.1,1)]:
            with self.assertRaises(ValueError):e.schedule(*args)
    def test_zero_endpoint_identity(self):
        x=torch.randn(20,2);got=e.q_sample(x,torch.zeros(20,dtype=torch.long),torch.randn_like(x),self.s);torch.testing.assert_close(got,x,rtol=0,atol=0)
    def test_q_rejects_shapes(self):
        with self.assertRaises(ValueError):e.q_sample(torch.zeros(4,2),torch.ones(4,dtype=torch.long),torch.zeros(4,1),self.s)
        with self.assertRaises(ValueError):e.q_sample(torch.zeros(4,2),torch.ones(4),torch.zeros(4,2),self.s)
    def test_q_rejects_nonfinite(self):
        with self.assertRaises(ValueError):e.q_sample(torch.full((1,2),float('nan')),torch.ones(1,dtype=torch.long),torch.zeros(1,2),self.s)
    def test_invalid_times(self):
        for t in [-1,201]:
            with self.assertRaises(ValueError):e.q_sample(torch.ones(1,2),torch.tensor([t]),torch.zeros(1,2),self.s)
        with self.assertRaises(ValueError):e.reverse_mean(torch.ones(1,2),torch.tensor([0]),torch.zeros(1,2),self.s)
    def test_forward_two_step_hand_calculation(self):
        s=e.schedule(2,.1,.2);x=torch.full((1,2),2.);ep=torch.full_like(x,-1);y=e.q_sample(x,torch.tensor([2]),ep,s);target=2*np.sqrt(.72)-np.sqrt(.28);self.assertAlmostEqual(float(y[0,0]),target,places=6)
    def test_forward_closed_vs_chain_moments(self):
        n=50000;s=e.schedule(2,.1,.2);x0=torch.tensor([2.,-1.]).repeat(n,1)
        x1=s['alpha'][1].sqrt()*x0+s['beta'][1].sqrt()*torch.randn_like(x0);chain=s['alpha'][2].sqrt()*x1+s['beta'][2].sqrt()*torch.randn_like(x1)
        direct=e.q_sample(x0,torch.full((n,),2,dtype=torch.long),torch.randn_like(x0),s)
        expected=np.sqrt(.72)*np.array([2.,-1.]);np.testing.assert_allclose(chain.mean(0),expected,atol=.012);np.testing.assert_allclose(direct.mean(0),expected,atol=.012);np.testing.assert_allclose(chain.var(0),[.28,.28],atol=.008);np.testing.assert_allclose(direct.var(0),[.28,.28],atol=.008)
    def test_posterior_manual(self):
        s=e.schedule(2,.1,.2);mu=e.posterior_mean(torch.ones(1,2),torch.full((1,2),2.),torch.tensor([2]),s);expected=2*np.sqrt(.9)*.2/.28+np.sqrt(.8)*.1/.28;self.assertAlmostEqual(float(mu[0,0]),expected,places=5);self.assertAlmostEqual(float(s['posterior_var'][2]),.2*.1/.28,places=6)
    def test_two_posterior_mean_forms(self):
        x0=torch.randn(40,2);eps=torch.randn_like(x0);t=torch.randint(2,201,(40,));xt=e.q_sample(x0,t,eps,self.s);a=e.posterior_mean(xt,x0,t,self.s);b=e.reverse_mean(xt,t,eps,self.s);torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    def test_kl_mean_term_equals_weighted_epsilon(self):
        s=self.s;x0=torch.randn(30,2);eps=torch.randn_like(x0);pred=eps+.1*torch.randn_like(eps);t=torch.randint(2,201,(30,));xt=e.q_sample(x0,t,eps,s);muq=e.reverse_mean(xt,t,eps,s);mup=e.reverse_mean(xt,t,pred,s);var=s['posterior_var'][t]
        kl=((muq-mup)**2).sum(1)/(2*var);w=s['beta'][t]**2/(2*var*s['alpha'][t]*(1-s['abar'][t]));rhs=w*((eps-pred)**2).sum(1);torch.testing.assert_close(kl,rhs,rtol=3e-4,atol=2e-6)
    def test_posterior_at_one_is_x0(self):
        x0=torch.randn(10,2);t=torch.ones(10,dtype=torch.long);xt=e.q_sample(x0,t,torch.randn_like(x0),self.s);mu=e.posterior_mean(xt,x0,t,self.s);torch.testing.assert_close(mu,x0,atol=4e-4,rtol=4e-4)
    def test_embedding_is_time_dependent(self):
        m=e.EpsilonNet();em=m.embedding(torch.tensor([1,100,200]));self.assertEqual(tuple(em.shape),(3,13));self.assertFalse(torch.equal(em[0],em[1]));self.assertEqual(tuple(m(torch.randn(3,2),torch.tensor([1,100,200])).shape),(3,2))
    def test_sampling_exact_time_order(self):
        m=Recorder();s=e.schedule(5,.01,.1);a=e.sample(m,s,8,10);self.assertEqual(m.seen,[5,4,3,2,1]);self.assertEqual(a.shape,(8,2));self.assertTrue(np.isfinite(a).all())
    def test_final_step_mean_and_noise_switch(self):
        s=e.schedule(1,.1,.1);m=Recorder();seed=44;x=torch.randn(12,2,generator=torch.Generator().manual_seed(seed));expected=e.reverse_mean(x,torch.ones(12,dtype=torch.long),torch.zeros_like(x),s)
        a=e.sample(m,s,12,seed);np.testing.assert_allclose(a,expected.numpy(),rtol=1e-6);b=e.sample(m,s,12,seed,terminal_noise=True);self.assertFalse(np.array_equal(a,b))
    def test_seeded_sampler(self):
        m=e.EpsilonNet();s=e.schedule(5,.01,.1);a=e.sample(m,s,10,5);b=e.sample(m,s,10,5);np.testing.assert_array_equal(a,b)
    def test_reference_and_control_metrics(self):
        x,_=e.make_data();self.assertEqual(e.metrics(x)['coverage'],8);c=np.repeat(e.centers()[:1],100,0);self.assertEqual(e.metrics(c)['coverage'],1)
    def test_gradients_reach_model(self):
        m=e.EpsilonNet();x,_=e.make_data(16);x=torch.from_numpy(x);t=torch.randint(1,201,(16,));noise=torch.randn_like(x);loss=((m(e.q_sample(x,t,noise,self.s),t)-noise)**2).mean();loss.backward();self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
if __name__=='__main__':unittest.main()
