"""Independent arithmetic, enumeration, boundaries and clean-process checks."""
from fractions import Fraction as F
from pathlib import Path
import csv
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
import unittest
import experiment as ex

ROOT=Path(__file__).resolve().parent

class SamplingTests(unittest.TestCase):
    def test_01_fixed_hand_results_and_csv(self):
        mu,e2,var=ex.finite_moments([0,2,6],[2,1,1])
        self.assertEqual((mu,e2,var),(2,10,6))
        with (ROOT/'data/group_observations.csv').open() as stream:
            values=[int(r['loss']) for r in csv.DictReader(stream)]
        self.assertEqual(values,ex.fixed_results()['fixed_eight_groups']['values'])
        self.assertEqual(ex.sample_statistics(values),(2,F(48,7),F(6,7)))
        self.assertEqual(ex.sample_statistics([x for x in values for _ in range(8)]),(2,F(128,21),F(2,21)))
        self.assertEqual(ex.sample_mean_variance(6,64),F(3,32))
        self.assertEqual(ex.sample_mean_variance(6,64,8),F(3,4))

    def test_02_150_expanded_population_oracles(self):
        rng=random.Random(101)
        for _ in range(150):
            values=[F(rng.randint(-20,20),rng.randint(1,7)) for _ in range(4)]
            counts=[rng.randint(0,6) for _ in range(4)]
            if sum(counts)==0:counts[0]=1
            expanded=[x for x,c in zip(values,counts) for _ in range(c)]
            mu,e2,var=ex.finite_moments(values,counts)
            self.assertEqual(mu,statistics.mean(expanded))
            self.assertEqual(e2,statistics.mean(x*x for x in expanded))
            self.assertEqual(var,statistics.pvariance(expanded))
            a,b=F(3,2),F(-7,3)
            tm,_,tv=ex.finite_moments([a*x+b for x in values],counts)
            self.assertEqual(tm,a*mu+b);self.assertEqual(tv,a*a*var)

    def test_03_1360_iid_sequence_oracles(self):
        atoms=[0,0,2,6];checked=0
        for n in range(2,6):
            tuples=list(itertools.product(atoms,repeat=n))
            means=[F(sum(t),n) for t in tuples]
            self.assertEqual(statistics.mean(means),2)
            self.assertEqual(statistics.pvariance(means),F(6,n))
            avg_s2=statistics.mean(statistics.variance(list(map(F,t))) for t in tuples)
            self.assertEqual(avg_s2,6)
            dist=dict(ex.exact_mean_distribution(atoms,n))
            for v in set(means):self.assertEqual(dist[v],F(means.count(v),len(means)))
            self.assertEqual(sum(dist.values()),1);checked+=len(tuples)
        self.assertEqual(checked,1360)

    def test_04_copied_groups_and_without_replacement(self):
        atoms=[0,0,2,6];checked=0
        for groups in (2,3,4):
            for copies in (1,2,3):
                sequences=list(itertools.product(atoms,repeat=groups));n=groups*copies
                expanded=[[x for x in seq for _ in range(copies)] for seq in sequences]
                means=[statistics.mean(list(map(F,row))) for row in expanded]
                self.assertEqual(statistics.pvariance(means),F(6,groups))
                expected=F(copies*(groups-1)*6,n-1)
                self.assertEqual(statistics.mean(statistics.variance(list(map(F,row))) for row in expanded),expected)
                checked+=len(expanded)
        self.assertEqual(checked,1008)
        pairs=[(atoms[i],atoms[j]) for i in range(4) for j in range(4) if i!=j]
        means=[F(x+y,2) for x,y in pairs]
        self.assertEqual(statistics.pvariance(means),2)
        self.assertEqual(statistics.mean(F(x*y) for x,y in pairs)-4,-2)

    def test_05_covariance_and_exact_n64(self):
        self.assertEqual(ex.covariance([-1,0,1],[1,0,1],[1,1,1]),0)
        rng=random.Random(32)
        for _ in range(100):
            xs=[rng.randint(-5,5) for _ in range(5)];ys=[rng.randint(-5,5) for _ in range(5)];cs=[rng.randint(1,4) for _ in xs]
            xp=[F(x) for x,c in zip(xs,cs) for _ in range(c)];yp=[F(y) for y,c in zip(ys,cs) for _ in range(c)]
            cv=statistics.mean(x*y for x,y in zip(xp,yp))-statistics.mean(xp)*statistics.mean(yp)
            self.assertEqual(ex.covariance(xs,ys,cs),cv)
            self.assertEqual(statistics.pvariance([x+y for x,y in zip(xp,yp)]),statistics.pvariance(xp)+statistics.pvariance(yp)+2*cv)
        for n in (1,4,16,64):
            law=ex.exact_mean_distribution([0,0,2,6],n)
            self.assertEqual(sum(p for _,p in law),1)
            self.assertEqual(sum(x*p for x,p in law),2)
            self.assertEqual(sum((x-2)**2*p for x,p in law),F(6,n))

    def test_06_input_rejections(self):
        bad=[lambda:ex.finite_moments([],[]),lambda:ex.finite_moments([1],[0]),
             lambda:ex.finite_moments([1,2],[1]),lambda:ex.finite_moments([1],[-1]),
             lambda:ex.finite_moments([1],[True]),lambda:ex.finite_moments([True],[1]),
             lambda:ex.finite_moments([1.0],[1]),lambda:ex.finite_moments(['nan'],[1]),
             lambda:ex.finite_moments(['inf'],[1]),lambda:ex.finite_moments(['1/0'],[1]),
             lambda:ex.finite_moments([10**13],[1]),lambda:ex.finite_moments([F(1,10**13)],[1]),
             lambda:ex.sample_statistics([]),lambda:ex.sample_statistics([1]),
             lambda:ex.sample_statistics([1,False]),lambda:ex.sample_mean_variance(-1,5),
             lambda:ex.sample_mean_variance(1,0),lambda:ex.sample_mean_variance(1,True),
             lambda:ex.sample_mean_variance(1,5,2),lambda:ex.sample_mean_variance(1,5,0),
             lambda:ex.covariance([1],[1,2],[1]),lambda:ex.exact_mean_distribution([],2),
             lambda:ex.exact_mean_distribution([0,1],0),lambda:ex.exact_mean_distribution([0,1],65),
             lambda:ex.exact_mean_distribution([0,.5],2),lambda:ex.exact_mean_distribution([False,1],2)]
        for call in bad:
            with self.assertRaises(ValueError):call()
        self.assertEqual(len(bad),26)

    def test_07_simulation_contract_and_correct_targets(self):
        cfg=json.loads((ROOT/'data/config.json').read_text());trials,summary=ex.simulate(cfg)
        self.assertEqual(len(trials),32000);self.assertEqual(len(summary),8)
        for row in summary:
            self.assertEqual(row['independent_groups'],row['n'] if row['mode']=='iid' else row['n']//8)
            self.assertAlmostEqual(row['true_se']**2,6/row['independent_groups'],places=12)
            self.assertLess(abs(row['empirical_sd']/row['true_se']-1),.04)
            self.assertLess(abs(row['observed_bias']),4*row['mcse_of_mean_of_means'])
        # These checks compare known finite simulation output, not a proof of IID in real data.
        tiny=dict(cfg,repeats=4,sample_sizes=[16]);a,b=ex.simulate(tiny);a2,b2=ex.simulate(tiny)
        self.assertEqual((a,b),(a2,b2));self.assertEqual(tiny['atoms'],[0,0,2,6])

    def test_08_config_failures_preserve_outputs(self):
        cfg=json.loads((ROOT/'data/config.json').read_text())
        cases=[dict(cfg,repeats=True),dict(cfg,copies=0),dict(cfg,copies=1),dict(cfg,sample_sizes=[15]),
               dict(cfg,sample_sizes=[8]),dict(cfg,sample_sizes=[16,16]),dict(cfg,atoms=[0,float('inf')]),
               dict(cfg,atoms=[0,True]),dict(cfg,seed=-1),dict(cfg,repeats=20000,sample_sizes=[4096]),
               dict(cfg,extra=1),dict(cfg,sample_sizes=[])]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);old=tmp/'old';old.mkdir();(old/'sentinel').write_text('safe')
            for c in cases:
                p=tmp/'bad.json';p.write_text(json.dumps(c));new=tmp/'new'
                for target in (new,old):
                    with self.assertRaises(ValueError):ex.run(target,p)
                self.assertFalse(new.exists());self.assertEqual(list(old.iterdir()),[old/'sentinel']);self.assertEqual((old/'sentinel').read_text(),'safe')
        self.assertEqual(len(cases),12)

    def test_09_fresh_other_cwd_stdlib_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-S',str(ROOT/'experiment.py'),'--output',str(Path(tmp)/'out')],cwd=tmp,capture_output=True,text=True,check=True)
            self.assertIn('Summary rows: 8',result.stdout)
            for source in (ROOT/'outputs').iterdir():self.assertEqual(source.read_bytes(),(Path(tmp)/'out'/source.name).read_bytes())

    def test_10_document_python_snippets(self):
        import os
        old=Path.cwd();env={};count=0
        try:
            os.chdir(ROOT)
            for code in re.findall(r'```python\n(.*?)```',(ROOT/'lab.md').read_text(),re.S):
                exec(compile(code,'lab.md','exec'),env);count+=1
        finally:os.chdir(old)
        self.assertEqual(count,5)

    def test_11_fixed_teaching_config_guard(self):
        import shutil
        cfg=json.loads((ROOT/'data/config.json').read_text())
        self.assertTrue(ex.require_teaching_config())
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'data').mkdir();(tmp/'outputs').mkdir();(tmp/'figures').mkdir()
            for name in ('experiment.py','make_figures.py'):shutil.copyfile(ROOT/name,tmp/name)
            (tmp/'figures/sentinel.png').write_bytes(b'old figure')
            for file_config,output_config in [(dict(cfg,copies=4),cfg),(cfg,dict(cfg,copies=4))]:
                ex.checked_config(dict(cfg,copies=4))  # Still legal for general simulation.
                (tmp/'data/config.json').write_text(json.dumps(file_config))
                (tmp/'outputs/results.json').write_text(json.dumps({'config':output_config}))
                result=subprocess.run([sys.executable,str(tmp/'make_figures.py')],cwd=tmp,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Fixed teaching figures',result.stderr)
                self.assertEqual(list((tmp/'figures').iterdir()),[tmp/'figures/sentinel.png'])
                self.assertEqual((tmp/'figures/sentinel.png').read_bytes(),b'old figure')
                with self.assertRaises(ValueError):ex.require_teaching_config(tmp/'data/config.json',tmp/'outputs/results.json')

if __name__=='__main__':unittest.main(verbosity=2)
