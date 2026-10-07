import copy,unittest
import torch
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);self.x,self.y=e.data(17);self.m=e.model()
 def test_data_shape(self):self.assertEqual(self.x.shape,(17,64));self.assertEqual(self.y.shape,(17,1))
 def test_grad_finite(self):self.assertTrue(torch.isfinite(e.gradients(self.m,self.x,self.y)[1]).all())
 def test_accumulation_unequal(self):
  _,a=e.gradients(self.m,self.x,self.y);_,b=e.gradients(copy.deepcopy(self.m),self.x,self.y,sizes=[3,5,9]);torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-7)
 def test_wrong_weight_detected(self):
  _,a=e.gradients(self.m,self.x,self.y);_,b=e.gradients(copy.deepcopy(self.m),self.x,self.y,sizes=[3,5,9],wrong_equal=True);self.assertGreater(e.relative_error(a,b),.001)
 def test_checkpoint_gradient(self):
  _,a=e.gradients(self.m,self.x,self.y);_,b=e.gradients(copy.deepcopy(self.m),self.x,self.y,recompute=True);torch.testing.assert_close(a,b)
 def test_dropout_preserved(self):
  m=e.model(dropout=.4);torch.manual_seed(5);_,a=e.gradients(m,self.x,self.y);torch.manual_seed(5);_,b=e.gradients(copy.deepcopy(m),self.x,self.y,recompute=True);torch.testing.assert_close(a,b)
 def test_dropout_not_preserved_detected(self):
  m=e.model(dropout=.4);torch.manual_seed(5);_,a=e.gradients(m,self.x,self.y);torch.manual_seed(5);_,b=e.gradients(copy.deepcopy(m),self.x,self.y,recompute=True,preserve_rng=False);self.assertGreater(e.relative_error(a,b),.01)
 def test_batchnorm_counterexample(self):
  m=e.model(batchnorm=True);_,a=e.gradients(m,self.x,self.y);_,b=e.gradients(copy.deepcopy(m),self.x,self.y,sizes=[3,5,9]);self.assertGreater(e.relative_error(a,b),.01)
 def test_saved_reduced(self):
  a=e.saved_payload(self.m,self.x,self.y);b=e.saved_payload(self.m,self.x,self.y,True);self.assertLess(b['saved_nonparameter_unique_storage_bytes'],a['saved_nonparameter_unique_storage_bytes'])
 def test_bf16_finite(self):
  v,g=e.gradients(self.m,self.x,self.y,bf16=True);self.assertTrue(torch.isfinite(g).all());self.assertTrue(v>=0)
 def test_autocast_keeps_fp32_parameters(self):
  e.gradients(self.m,self.x,self.y,bf16=True);self.assertTrue(all(p.dtype==torch.float32 and p.grad.dtype==torch.float32 for p in self.m.parameters()))
 def test_cast_extremes(self):self.assertEqual(float(torch.tensor(1e-8).half()),0.);self.assertTrue(torch.isinf(torch.tensor(70000.).half()))
 def test_invalid_partition(self):
  for sizes in [[0,17],[1,2],[-1,18]]:
   with self.assertRaises(ValueError):e.gradients(self.m,self.x,self.y,sizes=sizes)
 def test_loss_scaling_before_round(self):self.assertGreater(float((torch.tensor(1e-8)*65536).half().float()/65536),0)
if __name__=='__main__':unittest.main()
