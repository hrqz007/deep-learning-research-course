"""Correctness tests stay active under python -O; timings have no speed threshold."""
import copy,unittest
import numpy as np
import torch
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_data_repeatability(self):
  a,b=e.make_data();c,d=e.make_data();torch.testing.assert_close(a,c);self.assertTrue(torch.equal(b,d))
 def test_heldout_is_different(self):self.assertFalse(torch.equal(e.make_data(8)[0],e.make_data(8,6501)[0]))
 def test_shapes(self):
  x,y=e.make_data(13);self.assertEqual(e.make_model()(x).shape,(13,2));self.assertEqual(y.dtype,torch.int64)
 def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in e.make_model().parameters()),20994)
 def test_ledger_lazy_adam(self):
  m=e.make_model();o=torch.optim.Adam(m.parameters(),foreach=False);b=e.memory_ledger(m,o)
  self.assertEqual(b['parameter_bytes'],83976);self.assertEqual(b['gradient_bytes'],0);self.assertEqual(b['optimizer_tensor_bytes'],0)
  e.step(m,o,*e.make_data(8));a=e.memory_ledger(m,o)
  self.assertEqual(a['gradient_bytes'],a['parameter_bytes']);self.assertEqual(a['optimizer_tensor_bytes'],2*a['parameter_bytes']+4*len(list(m.parameters())))
 def test_loop_equal_gradients_and_update(self):
  x,y=e.make_data(9);m=e.make_model();n=copy.deepcopy(m);a=torch.optim.SGD(m.parameters(),lr=.1);b=torch.optim.SGD(n.parameters(),lr=.1)
  e.step(m,a,x,y);e.step(n,b,x,y,'row_loop')
  for p,q in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-5);torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-5)
 def test_step_rejects_unknown(self):
  m=e.make_model()
  with self.assertRaises(ValueError):e.step(m,torch.optim.SGD(m.parameters(),lr=.1),*e.make_data(3),'bad')
 def test_timing_valid(self):
  r=e.measure('vectorized',batch=8,repeats=3,steps_per_repeat=2,warmup=1)
  self.assertEqual(len(r['seconds_per_step_blocks']),3);self.assertGreater(r['median_seconds'],0);self.assertAlmostEqual(r['samples_per_second']*r['median_seconds'],8)
 def test_timing_rejects_empty(self):
  with self.assertRaises(ValueError):e.measure('vectorized',repeats=0)
 def test_tensor_bytes(self):self.assertEqual(e.tensor_bytes(torch.zeros(3,4,dtype=torch.float64)),96)
 def test_train_loss_decreases(self):
  x,y=e.make_data(64);m=e.make_model();o=torch.optim.Adam(m.parameters(),lr=.01);a=float(e.step(m,o,x,y))
  for _ in range(20):b=float(e.step(m,o,x,y))
  self.assertLess(b,a)
if __name__=='__main__':unittest.main()
