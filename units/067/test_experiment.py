import copy,unittest
import torch
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);self.x,self.y=e.data(12);self.m=e.model()
 def test_equal_rank_mean(self):
  full=e.gradient(self.m,self.x,self.y)[0];rows=[e.gradient(copy.deepcopy(self.m),self.x[a:a+4],self.y[a:a+4]) for a in [0,4,8]]
  actual=e.average_gradients([r[0] for r in rows],[4,4,4],'rank_mean');torch.testing.assert_close(e.flatten(full),e.flatten(actual),atol=1e-14,rtol=1e-12)
 def test_unequal_weighted(self):
  full=e.gradient(self.m,self.x,self.y)[0];rows=[e.gradient(copy.deepcopy(self.m),self.x[a:b],self.y[a:b]) for a,b in [(0,2),(2,5),(5,12)]]
  actual=e.average_gradients([r[0] for r in rows],[2,3,7]);torch.testing.assert_close(e.flatten(full),e.flatten(actual),atol=1e-14,rtol=1e-12)
 def test_unequal_rank_mean_wrong(self):
  rows=[e.gradient(copy.deepcopy(self.m),self.x[a:b],self.y[a:b]) for a,b in [(0,2),(2,5),(5,12)]];a=e.flatten(e.average_gradients([r[0] for r in rows],[2,3,7]));b=e.flatten(e.average_gradients([r[0] for r in rows],[2,3,7],'rank_mean'));self.assertGreater(float((a-b).abs().max()),.001)
 def test_sgd_step_equivalent(self):
  a=copy.deepcopy(self.m);b=copy.deepcopy(self.m);oa=torch.optim.SGD(a.parameters(),lr=.1);ob=torch.optim.SGD(b.parameters(),lr=.1)
  e.gradient(a,self.x,self.y);oa.step();e.simulate_step(b,ob,self.x,self.y,[torch.arange(2),torch.arange(2,12)])
  for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=1e-14,rtol=1e-12)
 def test_masked_denominator(self):
  mask=torch.tensor([1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],dtype=torch.float64);a=e.gradient(self.m,self.x,self.y,mask)[0];b=e.gradient(self.m,self.x[:2],self.y[:2])[0];torch.testing.assert_close(e.flatten(a),e.flatten(b))
 def test_empty_rank_contributes_zero(self):
  g,n,_=e.gradient(self.m,self.x,self.y,torch.zeros(12));self.assertEqual(n,0);self.assertEqual(float(e.flatten(g).abs().sum()),0)
 def test_invalid_mask(self):
  with self.assertRaises(ValueError):e.gradient(self.m,self.x,self.y,-torch.ones(12))
 def test_zero_global_rejected(self):
  with self.assertRaises(ValueError):e.average_gradients([[torch.tensor(1.)]],[0])
 def test_padding_duplicates(self):
  a=sum(e.sampler_indices(),[]);self.assertEqual(len(a),12);self.assertEqual(len(set(a)),10);self.assertEqual(a.count(0),2);self.assertEqual(a.count(1),2)
 def test_drop_discards(self):
  a=sum(e.sampler_indices(drop_last=True),[]);self.assertEqual(len(a),9);self.assertEqual(len(set(a)),9);self.assertNotIn(9,a)
 def test_set_epoch(self):self.assertNotEqual(e.sampler_indices(shuffle=True,epoch=0),e.sampler_indices(shuffle=True,epoch=1))
 def test_ring_world1(self):self.assertEqual(e.ring_model(100,1,.1,10)['seconds'],0)
 def test_ring_formula(self):self.assertAlmostEqual(e.ring_model(100,4,.1,10)['seconds'],15.6)
 def test_ring_invalid(self):
  with self.assertRaises(ValueError):e.ring_model(100,0,.1,10)
if __name__=='__main__':unittest.main()
