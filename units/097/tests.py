"""独立数值断言；运行 python tests.py。先执行 experiment.py 生成证据。"""
import unittest, json
from pathlib import Path
import numpy as np
import experiment as e
P=Path(__file__).resolve().parent
class TestExperiment(unittest.TestCase):
    def test_records(self):
        m=json.loads((P/'outputs/metrics.json').read_text());self.assertEqual(len(m['records']),30)
        self.assertTrue(all(r['nfe']==r['steps'] and r['n']==4000 for r in m['records']))
    def test_recalculate_metric(self):
        a=np.load(P/'outputs/data.npz');m=json.loads((P/'outputs/metrics.json').read_text())
        for r in m['records']:
            x=a[f"{r['method']}_{r['steps']}_{r['seed']}"]
            self.assertAlmostEqual(float(np.sqrt((x.mean()-2)**2+(x.std()-.6)**2)),r['moment_distance'])
    def test_ode_deterministic(self):
        z=np.array([-1.,0.,1.]);a,_=e.generate(z,1,40,'ode');b,_=e.generate(z,2,40,'ode');np.testing.assert_array_equal(a,b)
    def test_exact_center(self):
        x,_=e.generate(np.zeros(5),1,40,'ode');np.testing.assert_array_equal(x,np.full(5,2.))
    def test_pairing(self):
        m=json.loads((P/'outputs/metrics.json').read_text())
        for s in m['paired_summary']:
            a={r['seed']:r['moment_distance'] for r in m['records'] if r['steps']==s['steps'] and r['method']=='sde'}
            b={r['seed']:r['moment_distance'] for r in m['records'] if r['steps']==s['steps'] and r['method']=='ode'}
            d=np.array([a[k]-b[k] for k in sorted(a)])
            self.assertAlmostEqual(float(d.mean()),s['paired_difference_mean']);self.assertAlmostEqual(float(d.std(ddof=1)/np.sqrt(5)),s['paired_difference_se'])

if __name__=='__main__': unittest.main(verbosity=2)
