"""Tests use unittest assertions, which remain active under python -O."""
import unittest
from experiment import *
class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(5250);self.q=self.rng.normal(size=(2,3,2)).astype(np.float64);self.k=self.rng.normal(size=(2,4,2)).astype(np.float64);self.v=self.rng.normal(size=(2,4,3)).astype(np.float64);self.m=np.ones((2,3,4),dtype=np.bool_);self.m[:,:,3]=False
    def test_scalar_cross_attention(self):
        o,c=attention(self.q,self.k,self.v,self.m);np.testing.assert_allclose(o,scalar_attention(self.q,self.k,self.v,self.m),atol=3e-15,rtol=1e-14);np.testing.assert_allclose(c['a'].sum(-1),1,atol=1e-15);self.assertTrue(np.all(c['a'][~self.m]==0))
    def test_masked_reverse_all_paths(self):
        go=self.rng.normal(size=(2,3,3)).astype(np.float64);o,c=attention(self.q,self.k,self.v,self.m);g=reverse(c,go);q,k,v=[tensor(z,True) for z in [self.q,self.k,self.v]]
        a=torch.softmax((q@k.transpose(-1,-2)/math.sqrt(2)).masked_fill(~torch.tensor(self.m,dtype=torch.bool,device=DEVICE),-torch.inf),-1);(a@v*tensor(go)).sum().backward()
        for t,key in [(q,'gq'),(k,'gk'),(v,'gv')]:np.testing.assert_allclose(t.grad.numpy(),g[key],atol=2e-14,rtol=1e-13)
        self.assertTrue(np.all(g['gs'][~self.m]==0));self.assertTrue(np.all(g['gv'][:,3]==0));self.assertTrue(np.all(g['gk'][:,3]==0))
    def test_full_jacobian(self):
        a=np.array([.2,.3,.5],dtype=np.float64);j=np.diag(a)-np.outer(a,a);h=1e-6;z=np.log(a)
        for k in range(3):
            dz=np.zeros(3,dtype=np.float64);dz[k]=h
            def soft(x):e=np.exp(x-x.max());return e/e.sum()
            np.testing.assert_allclose((soft(z+dz)-soft(z-dz))/(2*h),j[:,k],atol=6e-11,rtol=1e-8)
        np.testing.assert_allclose(j.sum(0),0,atol=1e-16)
    def test_hand_finite_difference(self):
        r=verify_derivatives();self.assertLess(r['finite_difference_max_abs'],1e-9);self.assertLess(r['weight_autograd_max_abs'],1e-13);self.assertLess(r['input_autograd_max_abs'],1e-13)
    def test_two_synchronous_updates(self):
        states=hand_ledger();x,y,w=hand_inputs()
        for s in states:
            xt=tensor(x,True);wt=[tensor(z,True) for z in w];loss,o=torch_objective(xt,tensor(y),wt);loss.backward();np.testing.assert_allclose(o.detach().numpy(),s['prediction'],atol=2e-14);self.assertAlmostEqual(float(loss.detach()),s['loss'],places=14)
            for i in range(3):np.testing.assert_allclose(wt[i].grad.numpy(),s['weight_gradient'][i],atol=2e-14)
            w=[z-.1*t.grad.numpy() for z,t in zip(w,wt)]
        self.assertLess(states[-1]['loss'],states[0]['loss'])
    def test_sample_position_sums(self):
        x,y,w=hand_inputs();_,c,g,per,gw,paths=objective(x,y,w)
        for wi,key in enumerate(['gq','gk','gv']):
            brute=np.zeros_like(w[wi])
            for b in range(2):
                for t in range(2):brute+=np.outer(x[b,t],g[key][b,t])
            np.testing.assert_allclose(brute,gw[wi],atol=1e-16)
        self.assertEqual(sum(paths).shape,x.shape)
    def test_empty_support_rejected(self):
        self.m[0,0]=False
        with self.assertRaisesRegex(ValueError,'no allowed'):attention(self.q,self.k,self.v,self.m)
    def test_query_padding_zero(self):
        self.m[:,2]=False;valid=np.ones((2,3),dtype=np.bool_);valid[:,2]=False;o,c=attention(self.q,self.k,self.v,self.m,valid);g=reverse(c,np.ones_like(o));self.assertTrue(np.isfinite(o).all());self.assertTrue(np.all(o[:,2]==0));self.assertTrue(np.all(g['gq'][:,2]==0));self.assertTrue(np.all(c['a'][:,2]==0))
    def test_key_padding_invariance(self):
        a,_=attention(self.q,self.k,self.v,self.m);self.k[:,3]=9000;self.v[:,3]=-9000;b,_=attention(self.q,self.k,self.v,self.m);np.testing.assert_array_equal(a,b)
    def test_mask_api_semantics(self):
        r=api_probes();self.assertLess(r['sdpa_error'],1e-13);self.assertLess(r['mha_error'],1e-13);self.assertGreater(r['inverted_sdpa_max_change'],1);self.assertTrue(r['key_padding_does_not_zero_query'])
    def test_empirical_empty_library_behavior(self):self.assertTrue(all(api_probes()['all_masked'].values()),'Pinned CPU path changed: inspect rather than clean NaN')
    def test_causal_future_injection(self):
        r=leakage_probe();self.assertEqual(r['causal']['prefix_max_change'],0.);self.assertGreater(r['unmasked']['prefix_max_change'],1.)
    def test_invalid_inputs(self):
        for q in [self.q.astype(np.float32),self.q[0],np.zeros((0,3,2),dtype=np.float64),self.q*np.nan,self.q*1e10]:
            with self.assertRaises(ValueError):attention(q,self.k,self.v,self.m)
        with self.assertRaises(ValueError):attention(self.q,self.k,self.v,self.m.astype(np.float64))
        with self.assertRaises(ValueError):attention(self.q,self.k,self.v,self.m[:1])
    def test_single_key_and_shift_invariance(self):
        mask=np.zeros_like(self.m);mask[:,:,0]=True;o,c=attention(self.q,self.k,self.v,mask);np.testing.assert_allclose(o,np.broadcast_to(self.v[:,0:1],o.shape),atol=0,rtol=0);g=reverse(c,np.ones_like(o));np.testing.assert_array_equal(g['gs'],np.zeros_like(g['gs']))
        q=np.ones((1,2,2),dtype=np.float64);k=np.array([[[1,2],[3,4]]],dtype=np.float64);v=k.copy();m=np.ones((1,2,2),dtype=np.bool_);a,_=attention(q,k,v,m);b,_=attention(q,k+100,v,m);np.testing.assert_allclose(a,b,atol=2e-14)
    def test_retained_results(self):
        data=json.loads((ROOT/'data/sequences.json').read_text());self.assertEqual(data,generate_dataset());retained=json.loads((ROOT/'outputs/results.json').read_text());fresh=leak_experiment(data);self.assertEqual(len(fresh),12)
        for a,b in zip(fresh,retained['runs']):
            for key in ['coefficient','test_predictions','test_mse','train_mse']:np.testing.assert_allclose(a[key],b[key],rtol=1e-13,atol=1e-14)
        for r in fresh:
            if r['condition']=='unmasked' and r['alpha']==8:self.assertLess(r['test_mse'],1e-5)
            if r['condition']=='causal':self.assertGreater(r['test_mse'],.8)
if __name__=='__main__':unittest.main(verbosity=2)
