"""Independent exact-arithmetic and protocol tests; standard library only."""
import csv
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import experiment as ex

HERE = Path(__file__).resolve().parent


class EvaluationTests(unittest.TestCase):
    def test_binary_exhaustive_fraction_oracle(self):
        # 1364 paired label sequences, not a call back into the production counter.
        seen = 0
        for n in range(1, 6):
            for flat in itertools.product((0, 1), repeat=2 * n):
                y, p = flat[:n], flat[n:]
                hits = sum(a * b for a, b in zip(y, p))
                actual = sum(y); predicted = sum(p)
                correct = sum(a == b for a, b in zip(y, p))
                m = ex.binary_metrics(y, p)
                expected = {'precision': Fraction(hits, predicted) if predicted else Fraction(0),
                            'recall': Fraction(hits, actual) if actual else Fraction(0),
                            'f1': Fraction(2 * hits, actual + predicted) if actual + predicted else Fraction(0),
                            'accuracy': Fraction(correct, n)}
                for key, value in expected.items():
                    self.assertAlmostEqual(m[key], float(value), places=14)
                self.assertEqual(m['tp'] + m['fp'] + m['fn'] + m['tn'], n)
                seen += 1
        self.assertEqual(seen, 1364)

    def test_multiclass_exhaustive_and_absent_labels(self):
        for n in range(1, 4):
            for flat in itertools.product((0, 1, 2), repeat=2*n):
                y, p = flat[:n], flat[n:]
                correct = sum(a == b for a, b in zip(y, p))
                exact = []
                for label in (0, 1, 2):
                    den = y.count(label) + p.count(label)
                    num = 2 * sum(a == label and b == label for a, b in zip(y, p))
                    exact.append(Fraction(num, den) if den else Fraction(0))
                m = ex.multiclass_metrics(y, p, [0, 1, 2])
                self.assertAlmostEqual(m['macro_f1'], float(sum(exact)/3), places=14)
                weighted = sum(y.count(k) * exact[k] for k in (0, 1, 2)) / n
                self.assertAlmostEqual(m['weighted_f1'], float(weighted), places=14)
                self.assertAlmostEqual(m['micro_f1'], float(Fraction(correct, n)), places=14)
        m = ex.multiclass_metrics([0,1], [0,1], [0,1,2])
        self.assertEqual(m['macro_f1'], 2/3)
        self.assertEqual(m['per_class'][2]['undefined'], ['precision','recall','f1'])
        self.assertEqual(ex.multiclass_metrics([0,1], [0,1], [0,1,2], 1)['macro_f1'], 1)

    def test_undefined_denominators(self):
        a = ex.binary_metrics([0,0],[0,0])
        self.assertEqual(a['accuracy'],1)
        self.assertEqual(a['undefined'], ['precision','recall','f1'])
        b = ex.binary_metrics([1,0],[0,0])
        self.assertEqual(b['undefined'], ['precision'])
        self.assertEqual(b['recall'],0)
        c = ex.binary_metrics([0,0],[1,0])
        self.assertEqual(c['undefined'], ['recall'])
        self.assertEqual(c['f1'],0)
        d = ex.binary_metrics([0,0],[0,0],1)
        self.assertEqual(d['f1'],1)
        self.assertEqual(d['undefined'], ['precision','recall','f1'])

    def test_threshold_hand_arithmetic_and_ties(self):
        y = [1,0,1,0,1,0,0,0]; scores = [.9,.8,.7,.6,.4,.3,.2,.1]
        threshold, rows = ex.select_threshold(y,scores)
        self.assertEqual(threshold,.4)
        self.assertEqual([r['total_cost'] for r in rows],[3,2,5,4,6])
        self.assertEqual([(r['tp'],r['fp'],r['fn'],r['tn']) for r in rows],[(3,3,0,2),(3,2,0,3),(2,2,1,3),(2,1,1,4),(1,0,2,5)])
        self.assertEqual(ex.classify([.4,.399999,0,1],.4),[1,0,0,1])
        self.assertEqual(ex.select_threshold([0,1],[.1,.9],[.3,.7])[0],.7)

    def test_regression_fraction_and_failures(self):
        m=ex.regression_metrics([2,4,6,8],[3,4,4,9])
        self.assertEqual(m['bias'],0);self.assertEqual(m['mae'],1)
        self.assertEqual(m['mse'],float(Fraction(6,4)))
        self.assertAlmostEqual(m['r2'],float(Fraction(7,10)))
        self.assertAlmostEqual(m['rmse']**2,1.5)
        self.assertIsNone(ex.regression_metrics([5,5],[5,5])['r2'])
        self.assertIsNone(ex.regression_metrics([5],[4])['r2'])
        self.assertLess(ex.regression_metrics([0,1],[10,10])['r2'],0)
        with self.assertRaises(ValueError):ex.regression_metrics([0,1e-200],[0,1e-200])
        for scale in (0.01,100):
            r=ex.regression_metrics([scale*x for x in [2,4,6,8]],[scale*x for x in [3,4,4,9]])
            self.assertAlmostEqual(r['mae'],scale)
            self.assertAlmostEqual(r['mse'],1.5*scale*scale)
            self.assertAlmostEqual(r['r2'],.7)

    def test_regression_constant_and_each_underflow_term(self):
        for value in (.1,.3,1e-50,-.1):
            for n in (2,3,11,37):
                result=ex.regression_metrics([value]*n,[value]*n)
                self.assertIsNone(result['r2']);self.assertEqual(result['mse'],0)
                self.assertEqual(result['r2_undefined_reason'],'n<2 or constant target')
        for y,p in [([0,0],[1,1e-200]),([0,0],[1,2e-162]),([0,1e-200,1],[0,0,1]),([0,1e-200],[0,1e-200])]:
            with self.assertRaises(ValueError):ex.regression_metrics(y,p)
        # Constant recognition must not bypass prediction validation.
        with self.assertRaises(ValueError):ex.regression_metrics([.1]*11,[float('nan')]*11)

    def test_failed_retry_keeps_previous_four_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';shutil.copytree(HERE/'data',data);out=root/'out'
            ex.run(out,data);before={n:(out/n).read_bytes() for n in ex.OUTPUT_NAMES}
            with (data/'validation.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            for row in rows:row['y']='0'
            with (data/'validation.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            self.assertEqual(ex.make_plan(ex.load_rows(data/'train.csv'),ex.load_rows(data/'validation.csv'),data)[0]['threshold'],.9)
            (data/'sealed_test.csv').write_text('bad,header\n1,2\n')
            for target in (out,root/'new'):
                with self.assertRaises(ValueError):ex.run(target,data)
                self.assertEqual({n:(out/n).read_bytes() for n in ex.OUTPUT_NAMES},before)
                self.assertFalse(list(root.glob('.dl021-attempt-*')))
            self.assertFalse(any((root/'new'/n).exists() for n in ex.OUTPUT_NAMES))

    def test_calibration_exact_bins_and_decision_counterexample(self):
        probabilities=[Fraction(0),Fraction(1,5),Fraction(1,4),Fraction(9,20),Fraction(1,2),Fraction(13,20),Fraction(3,4),Fraction(1)]
        y=[0,0,1,0,1,0,1,1]
        r=ex.calibration_bins(y,[float(p) for p in probabilities])
        self.assertEqual([b['n'] for b in r['bins']],[2,2,2,2])
        for b,p,f in zip(r['bins'],[Fraction(1,10),Fraction(7,20),Fraction(23,40),Fraction(7,8)],[0,Fraction(1,2),Fraction(1,2),1]):
            self.assertAlmostEqual(b['mean_probability'],float(p))
            self.assertAlmostEqual(b['positive_frequency'],float(f))
        self.assertAlmostEqual(r['ece'],float(Fraction(9,80)))
        exact=sum((p-t)**2 for p,t in zip(probabilities,y))/8
        self.assertAlmostEqual(r['brier'],float(exact))
        empty=ex.calibration_bins([0,1],[0,1])['bins'][1]
        self.assertEqual(empty['n'],0);self.assertIsNone(empty['positive_frequency'])
        self.assertEqual(ex.classify([.6]*4,.5),ex.classify([.9]*4,.5))
        self.assertAlmostEqual(ex.calibration_bins([1,0,1,0],[.6]*4)['brier'],.26)
        self.assertAlmostEqual(ex.calibration_bins([1,0,1,0],[.9]*4)['brier'],.41)

    def test_entity_time_and_group_counts(self):
        panel=ex.load_rows(HERE/'data/split_panel.csv')
        ep=ex.split_by_entity(panel,['A'],['B'],['C'])
        tp=ex.split_by_time(panel,1,2)
        self.assertTrue(ex.check_partitions(ep,True,False)['entity_disjoint'])
        self.assertFalse(ex.check_partitions(ep,True,False)['strict_time_order'])
        self.assertTrue(ex.check_partitions(tp,False,True)['strict_time_order'])
        self.assertFalse(ex.check_partitions(tp,False,True)['entity_disjoint'])
        with self.assertRaises(ValueError):ex.check_partitions(tp)
        with self.assertRaises(ValueError):ex.split_by_entity(panel,['A'],['A'],['B','C'])
        with self.assertRaises(ValueError):ex.split_by_time(panel,1,1)
        with self.assertRaises(ValueError):ex.split_by_time(panel,0,1)
        m=ex.grouped_metrics([1,0,1,0],[.8,.7,.2,.1],['A','A','B','B'],.5)
        self.assertEqual(m['A']['recall'],1)
        self.assertEqual(m['B']['undefined'],['precision'])

    def test_bad_input_rejection(self):
        calls = [lambda:ex.binary_metrics([],[]),lambda:ex.binary_metrics([1],[0,1]),
                 lambda:ex.binary_metrics([True],[1]),lambda:ex.binary_metrics([2],[1]),
                 lambda:ex.binary_metrics([1.0],[1]),lambda:ex.binary_metrics([1],[1],False),
                 lambda:ex.binary_metrics([1],[1],2),lambda:ex.count_metrics({'tp':-1,'tn':2,'fp':0,'fn':0}),
                 lambda:ex.count_metrics({'tp':0,'tn':0,'fp':0,'fn':0}),
                 lambda:ex.multiclass_metrics([1],[0],[0]),lambda:ex.multiclass_metrics([1],[0],[0,1,1]),
                 lambda:ex.multiclass_metrics([True],[0],[0,1]),lambda:ex.classify([.1],float('nan')),
                 lambda:ex.classify([float('inf')],.5),lambda:ex.classify([-.1],.5),lambda:ex.classify([1.1],.5),
                 lambda:ex.classify([True],.5),lambda:ex.classify([], .5),lambda:ex.select_threshold([0],[.5],[]),
                 lambda:ex.select_threshold([0],[.5],[.5,.5]),lambda:ex.threshold_table([0],[.5],c_fn=0,c_fp=0),
                 lambda:ex.threshold_table([0],[.5],c_fn=-1),lambda:ex.regression_metrics([float('nan')],[1]),
                 lambda:ex.regression_metrics([1e101],[1]),lambda:ex.regression_metrics([True],[1]),
                 lambda:ex.calibration_bins([0],[.5],[0,.5,.5,1]),lambda:ex.calibration_bins([0],[.5],[.1,1]),
                 lambda:ex.calibration_bins([0],[.5],[0,.9]),lambda:ex.calibration_bins([0],[1.1]),
                 lambda:ex.grouped_metrics([0],[.5],[],.5),lambda:ex.grouped_metrics([0],[.5],[''],.5)]
        for i,call in enumerate(calls):
            with self.subTest(case=i),self.assertRaises(ValueError):call()

    def test_freeze_before_first_test_read_and_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=root/'data';shutil.copytree(HERE/'data',data)
            original_loader=ex.load_rows;events=[]
            def observed(path):
                name=Path(path).name
                if name=='sealed_test.csv':
                    plans=list(root.glob('.dl021-attempt-*/frozen_plan.json'))
                    self.assertEqual(len(plans),1)
                    self.assertEqual(json.loads(plans[0].read_text())['threshold'],.4)
                events.append(name)
                return original_loader(path)
            with patch.object(ex,'load_rows',side_effect=observed):first=ex.run(root/'first',data)
            self.assertEqual(events.count('sealed_test.csv'),1)
            self.assertLess(events.index('validation.csv'),events.index('sealed_test.csv'))
            with (data/'sealed_test.csv').open(newline='', encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            for row in rows:row['y']=str(1-int(row['y']));row['sensor']=str(10-int(row['sensor']))
            with (data/'sealed_test.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            second=ex.run(root/'second',data)
            self.assertEqual((root/'first/frozen_plan.json').read_bytes(),(root/'second/frozen_plan.json').read_bytes())
            self.assertNotEqual(first['test'],second['test'])
            self.assertEqual(first['threshold'],second['threshold'])
            # Invalid development data must fail before opening the holdout.
            (data/'validation.csv').write_text('bad,header\n1,2\n')
            opened=[]
            def traced(path):opened.append(Path(path).name);return original_loader(path)
            with patch.object(ex,'load_rows',side_effect=traced),self.assertRaises(ValueError):ex.run(root/'third',data)
            self.assertNotIn('sealed_test.csv',opened)
            self.assertFalse((root/'third/frozen_plan.json').exists())

    def test_fixed_figure_inputs_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shutil.copytree(HERE/'data',root/'data');shutil.copytree(HERE/'outputs',root/'outputs')
            shutil.copy(HERE/'experiment.py',root/'experiment.py');shutil.copy(HERE/'make_figures.py',root/'make_figures.py')
            (root/'figures').mkdir();sentinel=root/'figures/keep.png';sentinel.write_bytes(b'preserve')
            with patch.object(ex,'HERE',root):ex.require_default_figure_inputs()
            custom=root/'custom';shutil.copytree(root/'data',custom)
            with (custom/'validation.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            for row in rows:row['y']='0'
            with (custom/'validation.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            ex.run(root/'outputs',custom)
            with patch.object(ex,'HERE',root),self.assertRaises(ValueError):ex.require_default_figure_inputs()
            result=subprocess.run([sys.executable,str(root/'make_figures.py')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('Fixed teaching figures',result.stderr)
            self.assertEqual(list((root/'figures').iterdir()),[sentinel]);self.assertEqual(sentinel.read_bytes(),b'preserve')
            # The entry guard ran before importing matplotlib.
            self.assertNotIn('matplotlib',result.stderr)

    def test_output_isolation_and_fresh_process(self):
        original={p:p.read_bytes() for p in (HERE/'data').glob('*.csv')}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for out in (HERE,HERE/'data',HERE/'data/subdir',HERE.parent):
                with self.assertRaises(ValueError):ex.run(out)
            dest=root/'bad';dest.mkdir();(dest/'summary.json').symlink_to(HERE/'data/train.csv')
            with self.assertRaises(ValueError):ex.run(dest)
            self.assertFalse((dest/'frozen_plan.json').exists())
            for name in ('a','b'):
                subprocess.run([sys.executable,str(HERE/'experiment.py'),'--output-dir',str(root/name)],cwd=root,check=True,capture_output=True,text=True)
            for name in ex.OUTPUT_NAMES:
                self.assertEqual((root/'a'/name).read_bytes(),(root/'b'/name).read_bytes())
                self.assertEqual((root/'a'/name).read_bytes(),(HERE/'outputs'/name).read_bytes())
        for path,content in original.items():self.assertEqual(path.read_bytes(),content)


if __name__ == '__main__':
    unittest.main(verbosity=2)
