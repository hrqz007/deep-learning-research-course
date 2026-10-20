import unittest,json
from pathlib import Path
import numpy as np
import experiment as e
P=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_pack_roundtrip(self):
  for bits in (2,4,8):
   qmax=2**(bits-1)-1;q=np.arange(-qmax,qmax+1,dtype='int8');np.testing.assert_array_equal(q,e.unpack(e.pack(q,bits),bits,q.shape))
 def test_rounding_bound(self):
  w=np.array([-.91,-.1,0,.4,1.],dtype='float32')
  for bits in (2,4,8):q,s=e.quantize(w,bits);self.assertLessEqual(float(abs(w-q*s).max()),float(s)/2+1e-7)
 def test_zero_weight(self):
  q,s=e.quantize(np.zeros((3,3),dtype='float32'),4);self.assertTrue(np.isfinite(s));self.assertFalse(q.any())
 def test_quality_recompute(self):
  m=json.loads((P/'outputs/metrics.json').read_text());d=np.load(P/'outputs/data.npz');pred=np.load(P/'outputs/predictions.npz')
  for r in m['quality']:
   mask=np.ones(1200,bool) if r['group']=='all' else d['test_group'] if r['group']=='tail' else ~d['test_group'];acc=(pred[f"{r['method']}_s{r['seed']}"].argmax(1)[mask]==d['test_y'][mask]).mean();self.assertAlmostEqual(float(acc),r['accuracy']);self.assertEqual(int(mask.sum()),r['n'])
 def test_real_payload_bytes(self):
  m=json.loads((P/'outputs/metrics.json').read_text())
  for r in m['quality']:self.assertEqual((P/'outputs'/f"{r['method']}_s{r['seed']}.bin").stat().st_size,r['raw_parameter_payload_bytes'])
 def test_protocol(self):
  m=json.loads((P/'outputs/metrics.json').read_text());self.assertEqual(len(m['quality']),45);self.assertEqual(len(m['timings']),630)
  d=np.load(P/'outputs/data.npz');self.assertEqual(len(d['calibration_x']),400);self.assertEqual(len(d['train_x']),1800)
if __name__=='__main__':unittest.main(verbosity=2)
