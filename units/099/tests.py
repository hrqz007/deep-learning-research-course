import unittest,json
from pathlib import Path
import numpy as np
import torch
import experiment as e
P=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_exact_random(self):
  torch.manual_seed(9)
  for n in (1,7,33):
   q,k,v=[torch.randn(n,5,dtype=torch.float64) for _ in range(3)]
   torch.testing.assert_close(e.dense(q,k,v),e.blocked(q,k,v,8),atol=1e-12,rtol=1e-12)
 def test_extreme_scores(self):
  q=torch.tensor([[1000.,0.]],dtype=torch.float64);k=torch.tensor([[1000.,0.],[-1000.,0.]],dtype=torch.float64);v=torch.tensor([[3.],[9.]],dtype=torch.float64)
  self.assertEqual(float(e.blocked(q,k,v,1)[0,0]),3.)
 def test_uniform_attention(self):
  q=torch.zeros((3,2),dtype=torch.float64);v=torch.tensor([[1.],[2.],[6.]],dtype=torch.float64)
  torch.testing.assert_close(e.blocked(q,q,v,2),torch.full((3,1),3.,dtype=torch.float64))
 def test_saved_correctness(self):
  d=np.load(P/'outputs/data.npz')
  for n in (64,128,256,512):np.testing.assert_allclose(d[f'dense_{n}'],d[f'blocked_{n}'],atol=1e-12,rtol=1e-12)
 def test_retrieval(self):
  m=json.loads((P/'outputs/metrics.json').read_text());self.assertEqual(m['local_successes'],5)
  self.assertTrue(next(r for r in m['retrieval'] if r['length']==64 and r['target_position']==32)['target_visible_to_local'])
  for r in m['retrieval']:
   self.assertEqual(r['local_success'],r['target_visible_to_local']);self.assertTrue(r['full_success']);self.assertAlmostEqual(r['full_value'],r['blocked_value'])
 def test_timing_records(self):
  m=json.loads((P/'outputs/metrics.json').read_text());self.assertEqual(len(m['timings']),72)
  for r in m['timings']:self.assertGreater(r['seconds'],0);self.assertEqual(r['score_tile_bytes'],(32*32 if r['method']=='blocked' else r['length']**2)*8)
if __name__=='__main__':unittest.main(verbosity=2)
