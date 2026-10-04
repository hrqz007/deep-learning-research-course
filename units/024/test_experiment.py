"""NumPy implementation checked against Decimal, SciPy and independent derivatives."""
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import csv,json,math,random,subprocess,sys,tempfile,unittest,shutil,importlib.util
import numpy as np
from scipy.special import softmax,log_softmax,expit,logsumexp
import experiment as e
HERE=Path(__file__).resolve().parent
class ClassificationTests(unittest.TestCase):
 def test_01_decimal_multiclass_240_vectors(self):
  rng=random.Random(172);cases=[]
  for _ in range(240):
   k=rng.randrange(2,7);z=[rng.randrange(-100,101)/4 for _ in range(k)];y=rng.randrange(k);cases.append((z,y))
  cases += [([1000,1001,-1000],2),([37,0,0],0),([0,0,0],0)]
  with localcontext() as ctx:
   ctx.prec=100
   for z,y in cases:
    dz=[D(str(v)) for v in z];weights=[v.exp() for v in dz];s=sum(weights);p=[v/s for v in weights];loss=s.ln()-dz[y]
    l,g,got=e.multiclass_loss_gradient([z],[y]);np.testing.assert_allclose(got[0],[float(v) for v in p],rtol=3e-14,atol=1e-300);self.assertTrue(math.isclose(l,float(loss),rel_tol=3e-14,abs_tol=1e-300))
    ref=[float(v-(i==y)) for i,v in enumerate(p)];np.testing.assert_allclose(g[0],ref,rtol=3e-14,atol=1e-300)
 def test_02_scipy_300_batches(self):
  rng=np.random.default_rng(118)
  for _ in range(300):
   n=int(rng.integers(1,10));k=int(rng.integers(2,8));z=rng.normal(0,30,(n,k));y=rng.integers(k,size=n);loss,g,p=e.multiclass_loss_gradient(z,y)
   np.testing.assert_allclose(p,softmax(z,axis=1),rtol=2e-14,atol=1e-15);np.testing.assert_allclose(e.log_softmax(z),log_softmax(z,axis=1),rtol=2e-14,atol=2e-14)
   self.assertAlmostEqual(loss,float(-np.mean(log_softmax(z,axis=1)[np.arange(n),y])),places=12);np.testing.assert_allclose(np.sum(g,axis=1),0,atol=1e-16)
   shifts=rng.normal(0,100,(n,1));np.testing.assert_allclose(e.log_softmax(z+shifts),e.log_softmax(z),atol=8e-14,rtol=1e-13)
 def test_03_binary_decimal_and_scipy(self):
  for z in [-1000,-40,-37,-2,0,2,37,40,1000]:
   np.testing.assert_allclose(e.sigmoid([z]),expit([z]),atol=1e-300,rtol=2e-14)
   for y in [0,1]:
    loss,g=e.binary_loss_gradient([z],[y])
    with localcontext() as ctx:
     ctx.prec=100;dz=D(z);signed=-dz if y else dz;ref=(1+signed.exp()).ln();prob=1/(1+(-dz).exp());rg=prob-y
     # 100-digit Decimal also rounds log(1+exp(-1000)) to zero; both float results are zero.
     self.assertTrue(math.isclose(loss,float(ref),rel_tol=3e-14,abs_tol=1e-300));self.assertTrue(math.isclose(g[0],float(rg),rel_tol=3e-14,abs_tol=1e-300))
  self.assertLess(e.binary_loss_gradient([37],[1])[1][0],0);self.assertEqual(float(expit(37)-1),0)
 def test_04_logit_and_weight_finite_differences(self):
  rng=np.random.default_rng(883);maxerr=0.
  for _ in range(60):
   z=rng.normal(size=(3,3));y=rng.integers(3,size=3);loss,g,p=e.multiclass_loss_gradient(z,y);h=1e-5
   for idx in np.ndindex(z.shape):
    plus=z.copy();minus=z.copy();plus[idx]+=h;minus[idx]-=h;num=(e.multiclass_loss_gradient(plus,y)[0]-e.multiclass_loss_gradient(minus,y)[0])/(2*h);maxerr=max(maxerr,abs(num-g[idx]))
   x=np.column_stack([rng.normal(size=(6,2)),np.ones(6)]);w=rng.normal(size=(3,3));yy=np.array([0,1,2,0,1,2]);loss,gw,p=e.objective(x,yy,w,.03)
   for idx in np.ndindex(w.shape):
    plus=w.copy();minus=w.copy();plus[idx]+=h;minus[idx]-=h;num=(e.objective(x,yy,plus,.03)[0]-e.objective(x,yy,minus,.03)[0])/(2*h);maxerr=max(maxerr,abs(num-gw[idx]))
  self.assertLess(maxerr,2e-9)
 def test_05_hand_initial_gradient_and_first_step(self):
  rows=e.read_data(HERE/'data/classification.csv');tr=[r for r in rows if r[1]=='train'];x=np.array([r[2] for r in tr]);y=np.array([r[3] for r in tr]);loss,g,p=e.objective(x,y,np.zeros((3,3)),0)
  expected=np.array([[float(F(5,9)),float(F(-5,9)),0],[float(F(5,27)),float(F(5,27)),float(F(-10,27))],[0,0,0]])
  np.testing.assert_allclose(g,expected,atol=1e-16);self.assertAlmostEqual(loss,math.log(3));w,h=e.fit(x,y,1,.3,0);np.testing.assert_allclose(w,-.3*expected,atol=1e-16)
  np.testing.assert_allclose(e.multiclass_loss_gradient([[math.log(2),0,0]],[0])[1],[[-.5,.25,.25]],atol=1e-16)
 def test_06_reduction_and_wrong_double_softmax(self):
  z=np.array([[.2,1.,-.4],[2.,-.3,1.1]]);y=np.array([2,0]);m,g,p=e.multiclass_loss_gradient(z,y);s,gs,_=e.multiclass_loss_gradient(z,y,'sum');self.assertAlmostEqual(s,2*m);np.testing.assert_allclose(gs,2*g)
  # Independent chain-rule Jacobian for the wrong objective CE(softmax(z),y).
  loss,outer,_=e.multiclass_loss_gradient(p,y);chain=np.array([(np.diag(v)-np.outer(v,v))@o for v,o in zip(p,outer)])
  for idx in np.ndindex(z.shape):
   plus=z.copy();minus=z.copy();plus[idx]+=1e-5;minus[idx]-=1e-5
   num=(e.multiclass_loss_gradient(softmax(plus,axis=1),y)[0]-e.multiclass_loss_gradient(softmax(minus,axis=1),y)[0])/(2e-5);self.assertAlmostEqual(num,chain[idx],places=9)
  self.assertGreater(np.linalg.norm(chain-g),.1)
  direct=e.multiclass_loss_gradient([[8,0,0]],[0])[0];wrong=e.multiclass_loss_gradient(softmax([[8,0,0]],axis=1),[0])[0];self.assertLess(direct,.001);self.assertGreater(wrong,.55)
 def test_07_training_and_holdout_isolation(self):
  rows=e.read_data(HERE/'data/classification.csv');tr=[r for r in rows if r[1]=='train'];x=np.array([r[2] for r in tr]);y=np.array([r[3] for r in tr]);w,h=e.fit(x,y,400,.25,.03)
  self.assertTrue(all(b['objective']<=a['objective']+1e-14 for a,b in zip(h,h[1:])));self.assertLess(h[-1]['gradient_norm'],.002)
  binary=e.binary_path(400,.25);self.assertTrue(all(b['weight']>a['weight'] and b['mean_bce']<a['mean_bce'] for a,b in zip(binary,binary[1:])))
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);changed=root/'changed.csv'
   with (HERE/'data/classification.csv').open(newline='') as f:rr=list(csv.DictReader(f))
   for row in rr:
    if row['split']=='test':row['y']=str((int(row['y'])+1)%3)
   with changed.open('w',newline='') as f:
    ww=csv.DictWriter(f,fieldnames=list(rr[0]));ww.writeheader();ww.writerows(rr)
   a=e.run(root/'a');b=e.run(root/'b',data_path=changed);self.assertEqual(a['weights'],b['weights']);self.assertEqual((root/'a/training.csv').read_bytes(),(root/'b/training.csv').read_bytes());self.assertNotEqual(a['metrics']['test'],b['metrics']['test'])
 def test_08_bad_math(self):
  calls=[lambda:e.sigmoid([]),lambda:e.sigmoid([[1]]),lambda:e.sigmoid([True,1]),lambda:e.sigmoid([float('nan')]),lambda:e.sigmoid([1e7]),lambda:e.log_softmax([[0]]),lambda:e.log_softmax([0,1]),lambda:e.log_softmax([[float('inf'),0]]),lambda:e.multiclass_loss_gradient([[0,0]],[1.0]),lambda:e.multiclass_loss_gradient([[0,0]],[True]),lambda:e.multiclass_loss_gradient([[0,0],[1,1]],[False,1]),lambda:e.multiclass_loss_gradient([[0,0]],[2]),lambda:e.multiclass_loss_gradient([[0,0]],[-1]),lambda:e.binary_loss_gradient([0],[2]),lambda:e.binary_loss_gradient([0],[1],'none'),lambda:e.multiclass_loss_gradient([[0,0]],[0],'none'),lambda:e.objective([[1,2,0]],[0],np.zeros((3,3))),lambda:e.objective([[1,2,1]],[0],np.zeros((2,3))),lambda:e.objective([[1,2,1]],[0],np.zeros((3,3)),True),lambda:e.binary_path(True,.1),lambda:e.binary_path(3,float('nan')),lambda:e.binary_path(3,10)]
  for i,call in enumerate(calls):
   with self.subTest(case=i),self.assertRaises(ValueError):call()
 def test_09_csv_config_failures_preserve_outputs(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);old=root/'old';e.run(old);before={p.name:p.read_bytes() for p in old.iterdir()};original=(HERE/'data/classification.csv').read_text()
   bad=[original.replace('x2,y','x2,y,y'),original.replace('tr0,train,-2,0,0','tr0,train,-2,0,'),original.replace('tr0,train,-2,0,0','tr0,train,-2,0,0,1'),original.replace('tr1,train','tr0,train'),original.replace('tr0,train,-2,0,0','tr0,train,nan,0,0'),original.replace(',2\n',',1\n')]
   for i,content in enumerate(bad):
    data=root/'bad.csv';data.write_text(content)
    for out in (old,root/f'new{i}'):
     with self.assertRaises(ValueError):e.run(out,data_path=data)
     self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
    self.assertFalse((root/f'new{i}').exists())
   for i,c in enumerate([{**e.DEFAULT,'iterations':True},{**e.DEFAULT,'iterations':2001},{**e.DEFAULT,'learning_rate':float('inf')},{**e.DEFAULT,'l2':-1},{**e.DEFAULT,'unexpected':0}]):
    cfg=root/'bad.json';cfg.write_text(json.dumps(c))
    for out in (old,root/f'cfg{i}'):
     with self.assertRaises(ValueError):e.run(out,config_path=cfg)
     self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
   sym=root/'symlink';sym.mkdir();(sym/'results.json').symlink_to(old/'results.json')
   with self.assertRaises(ValueError):e.run(sym)
   self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
 def test_10_fresh_process_outputs(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'fresh';subprocess.run([sys.executable,str(HERE/'experiment.py'),'--output',str(out)],cwd=tmp,check=True,capture_output=True)
   for name in ['results.json','training.csv','binary_path.csv','predictions.csv','environment.json']:self.assertEqual((out/name).read_bytes(),(HERE/'outputs'/name).read_bytes())
 def test_11_nonzero_conversion_and_json_csv(self):
  tiny=np.longdouble('1e-400')
  if tiny!=0:
   with self.assertRaises(ValueError):e.real_array([tiny],1,'tiny',100)
  self.assertEqual(e.parsed_float('0e-400'),0.)
  with self.assertRaises(ValueError):e.parsed_float('1e-400')
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);old=root/'old';e.run(old);before={p.name:p.read_bytes() for p in old.iterdir()}
   data=root/'tiny.csv';data.write_text((HERE/'data/classification.csv').read_text().replace('tr0,train,-2,0,0','tr0,train,1e-400,0,0'))
   cfg=root/'tiny.json';cfg.write_text(json.dumps(e.DEFAULT).replace('0.03','1e-400'))
   for target in (old,root/'new'):
    with self.assertRaises(ValueError):e.run(target,data_path=data)
    with self.assertRaises(ValueError):e.run(target,config_path=cfg)
    self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
   self.assertFalse((root/'new').exists())
 def test_12_fixed_teaching_guard(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for directory in ['data','outputs']:shutil.copytree(HERE/directory,root/directory)
   for filename in ['experiment.py','make_figures.py']:shutil.copy(HERE/filename,root/filename)
   spec=importlib.util.spec_from_file_location('temp_guard',root/'experiment.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.require_teaching_inputs()
   for relative in module.TEACHING_INPUT_HASHES:
    path=root/relative;before=path.read_bytes();path.write_bytes(before+b' ')
    with self.assertRaises(ValueError):module.require_teaching_inputs()
    process=subprocess.run([sys.executable,str(root/'make_figures.py')],capture_output=True,text=True)
    self.assertNotEqual(process.returncode,0);self.assertIn('fixed illustrations',process.stderr);self.assertFalse((root/'figures').exists());path.write_bytes(before)
   module.require_teaching_inputs()
if __name__=='__main__':unittest.main(verbosity=2)
