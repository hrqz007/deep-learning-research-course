import unittest,json,csv,io
from pathlib import Path
import numpy as np
import experiment as e
P=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_threshold_boundary(self):
  self.assertTrue(e.eligible(.9,.8,.002));self.assertFalse(e.eligible(.899,.8,.002));self.assertFalse(e.eligible(.9,.799,.002));self.assertFalse(e.eligible(.9,.8,.0021))
 def test_quality_recompute(self):
  m=json.loads((P/'outputs/metrics.json').read_text());d=np.load(P/'outputs/data.npz');pred=np.load(P/'outputs/predictions.npz')
  for r in m['quality']:
   c=pred[r['model']].argmax(1)==d['test_y'];self.assertAlmostEqual(float(c.mean()),r['accuracy']);self.assertAlmostEqual(float(c[d['test_group']].mean()),r['tail_accuracy']);self.assertEqual(r['parameter_bytes'],(5*r['width']+2)*4)
 def test_replay_workload(self):
  d=np.load(P/'outputs/data.npz');self.assertEqual(len(d['replay_mixed_ids']),256);self.assertEqual(len(np.unique(d['replay_mixed_ids'])),256);self.assertTrue(d['test_group'][d['replay_tail_ids']].all())
 def test_summary_and_constraints(self):
  m=json.loads((P/'outputs/metrics.json').read_text());self.assertEqual(len(m['summary']),54)
  for r in m['summary']:self.assertEqual(r['eligible'],e.eligible(r['accuracy'],r['tail_accuracy'],r['batch_response_p95_seconds']))
 def test_batch_timing_accounting(self):
  rs=list(csv.DictReader(io.StringIO((P/'outputs/batch_timings.csv').read_text())));self.assertEqual(len(rs),24840);self.assertTrue(all(float(r['seconds'])>0 for r in rs))
  m=json.loads((P/'outputs/metrics.json').read_text())
  for s in m['summary']:
   sub=[r for r in rs if r['model']==s['model'] and int(r['batch'])==s['batch'] and r['workload']==s['workload']];lat=np.concatenate([np.repeat(float(r['seconds']),int(r['n'])) for r in sub]);self.assertEqual(len(lat),1280);self.assertAlmostEqual(float(np.quantile(lat,.95)),s['batch_response_p95_seconds'])
 def test_pareto(self):
  rows=[dict(config='a',parameter_bytes=10,replay_seconds_median=1,accuracy=.9),dict(config='b',parameter_bytes=20,replay_seconds_median=2,accuracy=.8),dict(config='c',parameter_bytes=30,replay_seconds_median=.5,accuracy=.95)]
  self.assertEqual(e.pareto(rows),['a','c'])
if __name__=='__main__':unittest.main(verbosity=2)
