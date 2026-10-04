"""Exact independent checks for the finite generalization lesson."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import csv,json,math,random,re,shutil,statistics,subprocess,sys,tempfile,unittest
import experiment as ex
ROOT=Path(__file__).resolve().parent

class RiskTests(unittest.TestCase):
    def test_01_all256_rules_against80_atoms(self):
        atoms=[(x,int(k < (1 if x<4 else 9))) for x in range(8) for k in range(10)]
        best=(0,0,0,0,1,1,1,1)
        for h in product((0,1),repeat=8):
            ref=F(sum(h[x]!=y for x,y in atoms),80)
            self.assertEqual(ex.risk(h),ref)
            self.assertEqual(ref,F(1+sum(a!=b for a,b in zip(h,best)),10))
        self.assertEqual(min(ex.risk(h) for h in ex.candidates('lookup')),F(1,10))

    def test_02_hand_training_results(self):
        d=ex.load_training();self.assertEqual(d,[(0,0),(1,1),(4,1),(5,0)])
        expected={'constant':(F(1,2),F(1,2)),'threshold':(F(1,4),F(2,5)),'lookup':(0,F(1,2))}
        for kind,(train,pop) in expected.items():
            h=ex.fit(kind,d);self.assertEqual(ex.empirical_risk(h,d),train);self.assertEqual(ex.risk(h),pop)
        self.assertEqual(ex.fit('threshold',d),(0,1,1,1,1,1,1,1))
        self.assertEqual(ex.fit('lookup',d),(0,1,0,0,1,0,0,0))
        self.assertEqual(max(abs(ex.empirical_risk(h,d)-ex.risk(h)) for h in ex.candidates('threshold')),F(11,20))

    def test_03_120_datasets_lookup_erm_and_nesting(self):
        rng=random.Random(505);all_rules=list(product((0,1),repeat=8))
        for _ in range(120):
            d=[(rng.randrange(8),rng.randrange(2)) for _ in range(rng.randrange(1,18))]
            reference=min(all_rules,key=lambda h:(sum(h[x]!=y for x,y in d),h))
            self.assertEqual(ex.fit('lookup',d),reference)
            losses=[ex.empirical_risk(ex.fit(k,d),d) for k in ex.CLASSES]
            self.assertGreaterEqual(losses[0],losses[1]);self.assertGreaterEqual(losses[1],losses[2])
            for kind in ex.CLASSES:
                h=ex.fit(kind,d);self.assertEqual(ex.empirical_risk(h,d),min(ex.empirical_risk(g,d) for g in ex.candidates(kind)))
        self.assertEqual(ex.fit('lookup',[(0,0),(0,1)]),(0,)*8)

    def test_04_80_weighted_risk_oracles(self):
        rng=random.Random(42)
        for _ in range(80):
            q=[rng.randrange(11) for _ in range(8)];w=[rng.randrange(1,5) for _ in range(8)];h=tuple(rng.randrange(2) for _ in range(8))
            atoms=[(x,int(k<q[x])) for x in range(8) for _ in range(w[x]) for k in range(10)]
            self.assertEqual(ex.risk(h,q,w),F(sum(h[x]!=y for x,y in atoms),len(atoms)))
        bayes=(0,0,0,0,1,1,1,1)
        self.assertEqual(ex.risk(bayes,(9,9,9,9,1,1,1,1)),F(9,10))
        self.assertEqual(ex.risk((0,1,1,1,1,1,1,1),weights=(1,3,3,3,1,1,1,1)),F(43,70))

    def test_05_uniform_bound_40_datasets_all_actual_rules(self):
        rng=random.Random(9);cs=ex.candidates('threshold');best=min(ex.risk(h) for h in cs)
        for _ in range(40):
            d=[(rng.randrange(8),rng.randrange(2)) for _ in range(rng.randrange(1,12))]
            empirical={h:ex.empirical_risk(h,d) for h in cs};minimum=min(empirical.values())
            eps=max(abs(empirical[h]-ex.risk(h)) for h in cs)
            for h in cs:self.assertLessEqual(ex.risk(h)-F(1,10),best-F(1,10)+(empirical[h]-minimum)+2*eps)

    def test_06_selection_and_squared_bias_variance_exact(self):
        for y in (0,1):
            h=ex.fit('constant',[(0,y)])
            self.assertEqual(ex.empirical_risk(h,[(0,y)]),0);self.assertEqual(ex.risk(h),F(1,2))
        atoms=[(train_y,new_y,F((9 if train_y else 1)*(9 if new_y else 1),100)) for train_y,new_y in product((0,1),repeat=2)]
        self.assertEqual(sum(p for _,_,p in atoms),1)
        self.assertEqual(sum((t-y)**2*p for t,y,p in atoms),F(9,50))
        self.assertEqual(sum((F(1,2)-y)**2*p for t,y,p in atoms),F(1,4))
        self.assertEqual(sum((F(9,10)-y)**2*p for t,y,p in atoms),F(9,100))

    def test_07_input_rejection(self):
        cases=[lambda:ex.model([]),lambda:ex.model([0]*7),lambda:ex.model([0]*7+[True]),lambda:ex.model([0]*7+[2]),
            lambda:ex.observations([]),lambda:ex.observations([(8,0)]),lambda:ex.observations([(-1,0)]),
            lambda:ex.observations([(0,True)]),lambda:ex.observations([(0,2)]),lambda:ex.observations([(0,0,0)]),
            lambda:ex.observations([(0.,0)]),lambda:ex.observations(['01']),lambda:ex.population([1]*7),
            lambda:ex.population([True]*8),lambda:ex.population([-1]*8),lambda:ex.population([11]*8),
            lambda:ex.population(weights=[0]*8),lambda:ex.population(weights=[-1]*8),lambda:ex.population(weights=[.5]*8),
            lambda:ex.fit('bad',[(0,0)]),lambda:ex.candidates('bad'),lambda:ex.sample(random.Random(0),0)]
        for call in cases:
            with self.assertRaises(ValueError):call()
        self.assertEqual(len(cases),22)

    def test_08_failure_output_isolation(self):
        cfg=json.loads((ROOT/'data/config.json').read_text());csv_good=(ROOT/'data/hand_training.csv').read_text()
        bad_csv=['sample_id,x,x,y\n1,0,1,0\n','sample_id,sample_id,x,y\n9,1,0,0\n','sample_id,x,y\n1,0\n','sample_id,x,y\n1,0,0,extra\n','sample_id,x,y\n1,0,True\n','sample_id,x,y\n2,0,0\n','sample_id,x,y\n']
        bad_configs=[dict(cfg,seed=True),dict(cfg,repeats=0),dict(cfg,sample_sizes=[0]),dict(cfg,sample_sizes=[8,8]),dict(cfg,validation_size=0),dict(cfg,test_size=5000),dict(cfg,repeats=2000,sample_sizes=[4096]),dict(cfg,extra=1)]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=root/'old';old.mkdir();names=['results.json','trials.csv','summary.csv','environment.json']
            for n in names:shutil.copyfile(ROOT/'outputs'/n,old/n)
            before={n:(old/n).read_bytes() for n in names};cp=root/'config.json';tp=root/'train.csv'
            for bad_c,bad_t in [(cfg,b) for b in bad_csv]+[(c,csv_good) for c in bad_configs]:
                cp.write_text(json.dumps(bad_c));tp.write_text(bad_t)
                for target in (old,root/'new'):
                    with self.assertRaises(ValueError):ex.run(target,cp,tp)
                self.assertFalse((root/'new').exists());self.assertEqual(before,{n:(old/n).read_bytes() for n in names})

    def test_09_fixed_graph_guard(self):
        self.assertTrue(ex.require_teaching_inputs())
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data').mkdir();(root/'outputs').mkdir();(root/'figures').mkdir()
            for name in ('experiment.py','make_figures.py'):shutil.copyfile(ROOT/name,root/name)
            for name in ('config.json','hand_training.csv'):shutil.copyfile(ROOT/'data'/name,root/'data'/name)
            shutil.copyfile(ROOT/'outputs/results.json',root/'outputs/results.json');sentinel=root/'figures/keep.png';sentinel.write_bytes(b'safe')
            cp=root/'data/config.json';c=json.loads(cp.read_text());c['sample_sizes']=[16];cp.write_text(json.dumps(c))
            result=subprocess.run([sys.executable,str(root/'make_figures.py')],cwd=root,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('Fixed teaching figures',result.stderr);self.assertEqual(sentinel.read_bytes(),b'safe');self.assertEqual(len(list((root/'figures').iterdir())),1)

    def test_10_fresh_stdlib_other_cwd_and_all_trials(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out';subprocess.run([sys.executable,'-S',str(ROOT/'experiment.py'),'--output',str(out)],cwd=td,capture_output=True,text=True,check=True)
            for p in (ROOT/'outputs').iterdir():self.assertEqual(p.read_bytes(),(out/p.name).read_bytes())
        with (ROOT/'outputs/trials.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),9600)
        for offset in range(0,len(rows),4):
            group=rows[offset:offset+4]
            self.assertLessEqual(float(group[2]['train_risk']),float(group[1]['train_risk'])+1e-15)
            self.assertLessEqual(float(group[1]['train_risk']),float(group[0]['train_risk'])+1e-15)
            expected=min(group[:3],key=lambda r:(float(r['validation_risk']),ex.CLASSES.index(r['class'])))
            self.assertEqual(group[3]['selected_class'],expected['class'])
            for k in ('train_risk','test_risk','population_risk','validation_risk'):self.assertEqual(group[3][k],expected[k])

    def test_11_document_snippets(self):
        env={};snippets=re.findall(r'```python\n(.*?)```',(ROOT/'lab.md').read_text(),re.S)
        for block in snippets:exec(compile(block,'lab.md','exec'),env)
        self.assertEqual(len(snippets),5)

if __name__=='__main__':unittest.main(verbosity=2)
