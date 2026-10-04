"""Independent exact arithmetic, exhaustive fixtures and process failure tests."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from collections import Counter
import unittest,math,json,csv,tempfile,subprocess,sys,random,shutil,importlib.util
import experiment as e
HERE=Path(__file__).resolve().parent
class ComparisonTests(unittest.TestCase):
 def test_01_pair_covariance_200_fixtures(self):
  rng=random.Random(781)
  for _ in range(200):
   n=rng.randrange(2,30);a=[rng.randrange(101) for _ in range(n)];b=[rng.randrange(101) for _ in range(n)];d=[x-y for x,y in zip(a,b)];s=e.paired_summary(a,b)
   mu=F(sum(d),n);var=(F(sum(x*x for x in d))-F(sum(d)**2,n))/(n-1)
   self.assertEqual(s['exact_mean'],str(mu));self.assertEqual(s['exact_variance'],str(var));self.assertEqual(s['variance_identity'],str(var));self.assertAlmostEqual(s['paired_se']**2,float(var/n),places=11)
  s=e.paired_summary([x[1] for x in e.DEFAULT_ROWS],[x[2] for x in e.DEFAULT_ROWS]);self.assertEqual(s['exact_variance'],'48/7');self.assertEqual(s['mean'],2)
 def test_02_exact_convolution_exhaustive(self):
  total=0
  for n in (2,3,4):
   for vals in product((-2,0,3),repeat=n):
    direct=Counter(sum(vals[i] for i in ids) for ids in product(range(n),repeat=n));self.assertEqual(e.convolution_counts(vals),dict(sorted(direct.items())));total+=1
  self.assertEqual(total,117)
 def test_03_quantiles_and_bootstrap_moments(self):
  for vals in ([2,4,0,6,-2,4,2,0],[-2,0,3],[5,5],[-1,2,4,4]):
   result=e.exact_bootstrap(vals);n=len(vals);mu=F(sum(vals),n);v=(sum(F(x*x) for x in vals)/n-mu*mu)/n
   self.assertEqual(F(result['mean']),mu);self.assertEqual(F(result['variance']),v);self.assertEqual(result['ordered_samples'],n**n)
  exact=e.exact_bootstrap([2,4,0,6,-2,4,2,0]);self.assertEqual(exact['percentile_95'],['1/4','15/4']);self.assertEqual(exact['variance'],'3/4')
  for n in range(1,31):
   vals=[F(i//2-4) for i in range(n)]
   for q in [F(0),F(1,40),F(1,4),F(1,2),F(39,40),F(1)]:
    direct=sorted(vals)[max(0,math.ceil(q*n)-1)];self.assertEqual(e.empirical_quantile(vals,q),direct)
 def test_04_rare_coverage_independent_sequences(self):
  # Each Bernoulli sequence has a known true probability; no use of production quantiles.
  coverage=F(0);intervals={}
  for k in range(9):
   weights=[]
   for ids in product((0,1),repeat=8):
    j=sum(ids);mass=F(k,8)**j*F(8-k,8)**(8-j);weights.append((F(20*j,8),mass))
   cumulative=F(0);lo=hi=None
   for value in sorted(set(v for v,p in weights)):
    cumulative+=sum(p for v,p in weights if v==value)
    if lo is None and cumulative>=F(1,40):lo=value
    if hi is None and cumulative>=F(39,40):hi=value
   intervals[k]=(lo,hi)
  for xs in product((0,1),repeat=8):
   k=sum(xs);prob=F(1,100)**k*F(99,100)**(8-k);lo,hi=intervals[k]
   if lo<=F(1,5)<=hi:coverage+=prob
  actual,rows=e.rare_coverage();self.assertEqual(coverage,actual);self.assertEqual([r['positive_count'] for r in rows if r['covers_true_mean']],[1,2])
  self.assertEqual(sum(F(r['sample_probability']) for r in rows),1)
 def test_05_cluster_and_selection_exact(self):
  group=[-2,0,2,4];rows=[v for v in group for _ in range(8)]
  self.assertEqual(e.moments(group)[1]/4,F(5,3));self.assertEqual(e.moments(rows)[1]/32,F(5,31))
  for m in range(1,9):
   vectors=list(product((-1,1),repeat=m));expectation=F(sum(max(x) for x in vectors),2**m);self.assertEqual(expectation,1-F(2,2**m))
   # Independent two-valued retest has zero mean for every selected candidate.
   self.assertEqual(sum(F(t,2) for t in (-1,1)),0)
  self.assertAlmostEqual(float(1-F(19,20)**20),.6415140775914578)
 def test_06_bootstrap_pairing_and_fresh_rng(self):
  ds=[2,4,0,6,-2,4,2,0];n=60;seed=45;got=e.bootstrap_means(ds,n,seed);rng=random.Random(seed)
  expected=[F(sum(ds[rng.randrange(8)] for _ in range(8)),8) for _ in range(n)];self.assertEqual(got,expected)
  random.seed(456);self.assertEqual(got,e.bootstrap_means(ds,n,seed))
  a=[x[1] for x in e.DEFAULT_ROWS];b=[x[2] for x in e.DEFAULT_ROWS];ids=[3,3,0,4,1,6,2,7]
  self.assertEqual(F(sum(a[i]-b[i] for i in ids),8),F(9,4))
 def test_07_invalid_math_inputs(self):
  calls=[lambda:e.moments([]),lambda:e.moments([1]),lambda:e.moments([True,2]),lambda:e.moments([1,float('nan')]),lambda:e.moments([1,101]),lambda:e.paired_summary([1,2],[1,2,3]),lambda:e.paired_summary([-1,2],[1,2]),lambda:e.interval([1,2],float('inf')),lambda:e.interval([1,2],False),lambda:e.interval([1,2],0),lambda:e.paired_t7([1,2]),lambda:e.convolution_counts([1]*11),lambda:e.quantile_counts({},F(1,2)),lambda:e.quantile_counts({0:0},F(1,2)),lambda:e.quantile_counts({0:1},.5),lambda:e.quantile_counts({0:1},F(2)),lambda:e.quantile_counts({0:1},F(1,2),False),lambda:e.bootstrap_means([1,2],True,1),lambda:e.bootstrap_means([1,2],20001,1),lambda:e.bootstrap_means([1,2],20,-1),lambda:e.empirical_quantile([],F(1,2)),lambda:e.empirical_quantile([1],F(1,2)),lambda:e.empirical_quantile([F(1)],F(-1)),lambda:e.rare_coverage(True),lambda:e.rare_coverage(8,F(0)),lambda:e.rare_coverage(8,.1),lambda:e.null_selection(10,1),lambda:e.null_selection(20,2**32)]
  for i,call in enumerate(calls):
   with self.subTest(i=i),self.assertRaises(ValueError):call()
 def test_08_failure_output_isolation(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);good=e.DEFAULT_CONFIG.copy();good.update(bootstrap_repeats=20,selection_repeats=20);cfg=root/'config.json';cfg.write_text(json.dumps(good));old=root/'old';e.run(old,cfg);before={p.name:p.read_bytes() for p in old.iterdir()}
   malformed=['seed_id,error_A,error_B,error_B\n0,30,28,25\n','seed_id,error_A\n0,30\n','seed_id,error_A,error_B\n0,30,28,9\n','seed_id,error_A,error_B\n0,30,\n',(HERE/'data/seed_scores.csv').read_text().replace('0,30,28','1,30,28'),(HERE/'data/seed_scores.csv').read_text().replace('0,30,28','0,101,28')]
   for i,txt in enumerate(malformed):
    rows=root/'bad.csv';rows.write_text(txt)
    for out in (old,root/f'new{i}'):
     with self.assertRaises(ValueError):e.run(out,cfg,rows)
     self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
    self.assertFalse((root/f'new{i}').exists())
   for i,change in enumerate([{'seed':True},{'seed':2**32},{'bootstrap_repeats':0},{'selection_repeats':5001},{'unexpected':1}]):
    c={**good,**change};cfg.write_text(json.dumps(c))
    for out in (old,root/f'cfg{i}'):
     with self.assertRaises(ValueError):e.run(out,cfg)
     self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
    self.assertFalse((root/f'cfg{i}').exists())
   cfg.write_text('{"seed":2,"seed":3,"bootstrap_repeats":20,"selection_repeats":20}')
   with self.assertRaises(ValueError):e.run(old,cfg)
   cfg.write_text(json.dumps(good));plan=root/'plan.json';plan.write_text('{}')
   with self.assertRaises(ValueError):e.run(old,cfg,plan_path=plan)
   self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
 def test_09_outputs_and_subprocess(self):
  with tempfile.TemporaryDirectory() as folder:
   out=Path(folder)/'fresh';subprocess.run([sys.executable,'-S',str(HERE/'experiment.py'),'--output',str(out)],cwd=folder,check=True,capture_output=True)
   for n in ['results.json','bootstrap.csv','selection.csv','rare_coverage.csv','environment.json']:self.assertEqual((out/n).read_bytes(),(HERE/'outputs'/n).read_bytes())
   r=json.loads((out/'results.json').read_text());self.assertAlmostEqual(r['t7_rounded_95'][0],2-2.365*math.sqrt(6/7));self.assertGreater(r['normal_approx_95'][0],0)
   with (out/'bootstrap.csv').open() as f:rows=list(csv.DictReader(f));self.assertEqual(len(rows),4000)
   with (out/'selection.csv').open() as f:rows=list(csv.DictReader(f));self.assertEqual(len(rows),6000)
   for row in rows:self.assertTrue(0<=int(row['winner'])<int(row['candidates']));self.assertIn(int(row['selected_validation_noise']),(-1,1));self.assertIn(int(row['independent_test_noise']),(-1,1))
   for summary in r['selection_summary']:
    group=[x for x in rows if int(x['candidates'])==summary['candidates']];self.assertEqual(sum(int(x['selected_validation_noise']) for x in group)/2000,summary['mean_selected_validation']);self.assertEqual(sum(int(x['independent_test_noise']) for x in group)/2000,summary['mean_independent_test'])
 def test_10_teaching_guard(self):
  # Copy the guarded implementation and inputs, so canonical data are untouched.
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);shutil.copytree(HERE/'data',root/'data');shutil.copytree(HERE/'outputs',root/'outputs');shutil.copy(HERE/'experiment.py',root/'experiment.py')
   spec=importlib.util.spec_from_file_location('guarded_copy',root/'experiment.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.require_teaching_inputs()
   r=json.loads((root/'outputs/results.json').read_text());r['config']['seed']=42;(root/'outputs/results.json').write_text(json.dumps(r))
   with self.assertRaises(ValueError):mod.require_teaching_inputs()
   shutil.copy(HERE/'outputs/results.json',root/'outputs/results.json');(root/'data/seed_scores.csv').write_text((HERE/'data/seed_scores.csv').read_text().replace('0,30,28','0,31,28'))
   with self.assertRaises(ValueError):mod.require_teaching_inputs()
if __name__=='__main__':unittest.main(verbosity=2)
