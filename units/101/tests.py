import unittest,json,csv,io
from pathlib import Path
import numpy as np
import experiment as e
P=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_single_request(self):
  r,_=e.simulate(np.array([2.]),np.array([10]),np.array([3]),1);self.assertAlmostEqual(r[0]['finish'],2+.002+.001+3*.00095);self.assertAlmostEqual(r[0]['ttft'],.003+.00095)
 def test_no_early_events(self):
  for row in csv.DictReader(io.StringIO((P/'outputs/requests.csv').read_text())):
   self.assertGreaterEqual(float(row['start']),float(row['arrival']));self.assertGreaterEqual(float(row['first_token']),float(row['start']));self.assertGreaterEqual(float(row['finish']),float(row['first_token']))
 def test_work_conservation(self):
  a,p,o=e.workload(11,60,20)
  for cap in (1,4,8):
   rs,tr=e.simulate(a,p,o,cap);self.assertEqual(len(rs),20);self.assertEqual(len(set(r['request'] for r in rs)),20);self.assertTrue(all(t[1]<=cap for t in tr))
 def test_capacity_one_equivalence(self):
  a,p,o=e.workload(23,30,10);self.assertEqual(e.simulate(a,p,o,1,True),e.simulate(a,p,o,1,False))
 def test_parallel_boundary(self):
  r=e.parallel_model(1);self.assertEqual(r['ring_communication_seconds'],0);self.assertEqual(r['speedup'],1)
 def test_summary_recompute(self):
  m=json.loads((P/'outputs/metrics.json').read_text());self.assertEqual(len(m['summary']),48)
  all_rows=list(csv.DictReader(io.StringIO((P/'outputs/requests.csv').read_text())));self.assertEqual(len(all_rows),8640)
  for s in m['summary']:
   rs=[r for r in all_rows if int(r['seed'])==s['seed'] and int(r['rate'])==s['rate'] and r['policy']==s['policy']];self.assertEqual(len(rs),180);self.assertAlmostEqual(float(np.quantile([float(r['latency']) for r in rs],.95)),s['latency_p95'])
if __name__=='__main__':unittest.main(verbosity=2)
