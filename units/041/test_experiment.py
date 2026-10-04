"""Independent numerical references and explicit failure tests; valid under -O."""
import copy,gzip,hashlib,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import experiment as e
ERRORS={}
class Test041(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.result=json.loads(gzip.decompress((e.ROOT/'outputs/results.json.gz').read_bytes()))
  cls.data={k:e.parse(v,k) for k,v in e.check_fixtures(e.ROOT/'data').items()}
 def close(self,a,b,tol=1e-11):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
 def test_hand_rational_and_local(self):
  h=e.hand_trace();self.close(h['h'],[.5,-.5]);self.close(h['prediction'],[1.1,-.9]);self.close(h['loss_each'],[.005,.125]);self.close(h['risk'],.065)
  self.close(h['parameter_contributions'],[[.075,0,.075,.025,.05],[0,-.375,-.375,.125,-.25]])
  self.close(h['gradient'],[.075,-.375,-.3,.15,-.2]);self.close(h['local']['dh_dz'],[.75,.75])
  self.close(h['theta_next_grouped'],[math.atanh(.5)-.0075,-math.atanh(.5)+.0375,.03,1.97,.14])
  self.close(e.hand_trace(h['theta_next_frozen'])['prediction'],[1.125,-.845]);self.close(e.hand_trace(h['theta_next_frozen'])['risk'],.0534125)
 def test_all_hand_paths_autograd_and_finite_difference(self):
  worst=0
  for theta in [e.hand_trace()['theta'],e.hand_trace()['theta_next_grouped'],e.hand_trace()['theta_next_frozen']]:
   h=e.hand_trace(theta);t=torch.tensor(theta,dtype=torch.float64,requires_grad=True);x=torch.eye(2,dtype=torch.float64,requires_grad=True);y=torch.tensor([1.,-.4],dtype=torch.float64)
   pred=t[3]*torch.tanh(x@t[:2]+t[2])+t[4];loss=.5*(pred-y)**2
   for i in range(2):self.close(torch.autograd.grad(loss[i]/2,t,retain_graph=True)[0].numpy(),h['parameter_contributions'][i])
   self.close(torch.autograd.grad(loss.mean(),x)[0].numpy(),h['input_gradient'])
   for j in range(5):
    plus=np.array(theta);minus=np.array(theta);plus[j]+=1e-6;minus[j]-=1e-6
    # Scalar math reference, independent of production forward/hand_trace.
    def scalar(th):return sum(.25*(th[3]*math.tanh(th[i]+th[2])+th[4]-[1,-.4][i])**2 for i in range(2))
    fd=(scalar(plus)-scalar(minus))/(2e-6);worst=max(worst,abs(fd-h['gradient'][j]));self.assertLess(abs(fd-h['gradient'][j]),1e-9)
  ERRORS['hand_fd_max_abs']=worst
 def test_freeze_detach_and_eval(self):
  d=e.freeze_demo();self.assertIsNotNone(d['frozen']['x_grad']);self.assertIsNone(d['frozen']['backbone_grad'])
  self.assertIsNone(d['detach']['x_grad']);self.assertIsNone(d['no_grad']['x_grad']);self.assertIsNotNone(d['eval_only']['backbone_grad'])
  for mode in d:self.close(d[mode]['head_grad'],d['frozen']['head_grad'])
 def test_frozen_batchnorm_buffers(self):
  m=torch.nn.BatchNorm1d(2,dtype=torch.float64);m.requires_grad_(False);x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
  before=m.running_mean.clone();m.train();m(x);self.assertFalse(torch.equal(before,m.running_mean));before=m.running_mean.clone();m.eval();m(x);self.assertTrue(torch.equal(before,m.running_mean))
 def test_independent_numpy_all_training(self):
  worst_theta=0.;worst_mse=0.;updates=0
  traces=[];s=self.data['source.csv'];tr=self.data['target_train.csv'];va=self.data['target_validation.csv']
  for r in self.result['source_runs']:traces.append((r['history'],s['x'],s['y']['related'],.1,.1,None))
  for g in self.result['target_groups']:
   for r in g['runs']:
    for c in r['candidates']:
     if c['history']:traces.append((c['history'],tr['x'][:g['n']],tr['y'][g['task']][:g['n']],c['backbone_lr'],c['setting'],(va['x'],va['y'][g['task']])))
  for rows,x,y,lr1,lr2,val in traces:
   t=np.array(rows[0]['theta']);lr=np.array([lr1]*3+[lr2]*2)
   for i,row in enumerate(rows):
    z=x[:,0]*t[0]+x[:,1]*t[1]+t[2];h=np.tanh(z);p=h*t[3]+t[4];q=(p-y)/len(y);dz=q*t[3]*(1-h*h)
    worst_theta=max(worst_theta,float(np.max(np.abs(t-row['theta']))));mse=float(np.mean((p-y)**2));worst_mse=max(worst_mse,abs(mse-row['train_mse']))
    if val is not None:
     xx,yy=val;vp=t[3]*np.tanh(xx[:,0]*t[0]+xx[:,1]*t[1]+t[2])+t[4];self.assertLess(abs(float(np.mean((vp-yy)**2))-row['validation_mse']),1e-10)
    if i<len(rows)-1:t-=lr*np.array([np.sum(dz*x[:,0]),np.sum(dz*x[:,1]),sum(dz),sum(q*h),sum(q)]);updates+=1
  self.assertLess(worst_theta,1e-10);self.assertLess(worst_mse,1e-10);self.assertEqual(updates,24840)
  ERRORS.update(training_theta_max_abs=worst_theta,training_mse_max_abs=worst_mse,independently_reconstructed_updates=updates)
 def test_ridge_stationarity_and_probe_fixed(self):
  tr=self.data['target_train.csv'];worst=0.
  for g in self.result['target_groups']:
   for r in g['runs']:
    if r['method'] not in ('pretrained_probe','random_probe','rbf_ridge'):continue
    for c in r['candidates']:
     x=tr['x'][:g['n']];y=tr['y'][g['task']][:g['n']]
     if r['method']=='rbf_ridge':
      grid=np.array([(a,b) for a in np.linspace(-2,2,5) for b in np.linspace(-2,2,5)]);phi=np.column_stack([np.exp(-np.sum((x[:,None,:]-grid[None,:,:])**2,2)/2),np.ones(len(x))]);coef=np.array(c['coefficients'])
     else:
      t=np.array(c['theta']);self.close(t[:3],c['initial'][:3]);phi=np.column_stack([np.tanh(x[:,0]*t[0]+x[:,1]*t[1]+t[2]),np.ones(len(x))]);coef=t[3:]
     reg=coef*c['setting'];reg[-1]=0;grad=phi.T@(phi@coef-y)/len(y)+reg;worst=max(worst,float(np.max(np.abs(grad))))
  self.assertLess(worst,1e-12);ERRORS['ridge_stationarity_max_abs']=worst
 def test_selection_predictions_metrics_and_boundary(self):
  metric_index={(r['seed'],r['task'],r['n'],r['method']):r for r in self.result['metrics']};n=0
  for g in self.result['target_groups']:
   for r in g['runs']:
    values=[sum((p-y)**2 for p,y in zip(c['validation_prediction'],self.data['target_validation.csv']['y'][g['task']]))/128 for c in r['candidates']]
    self.assertEqual(r['selected'],min(range(3),key=lambda i:(values[i],i)))
    for split,d in r['predictions'].items():
     data=self.data[{'train':'target_train.csv','validation':'target_validation.csv','test':'target_test.csv'}[split]];xx=data['x'][:len(d['prediction'])];c=r['candidates'][r['selected']]
     if r['method']=='rbf_ridge':
      grid=np.array([(a,b) for a in np.linspace(-2,2,5) for b in np.linspace(-2,2,5)]);phi=np.column_stack([np.exp(-np.sum((xx[:,None,:]-grid[None,:,:])**2,2)/2),np.ones(len(xx))]);expected=phi@np.array(c['coefficients'])
     else:
      t=np.array(c['theta']);expected=t[3]*np.tanh(xx[:,0]*t[0]+xx[:,1]*t[1]+t[2])+t[4]
     self.close(expected,d['prediction']);self.close(data['y'][g['task']][:len(xx)],d['y'])
     mse=sum((p-y)**2 for p,y in zip(d['prediction'],d['y']))/len(d['y']);self.close(mse,d['mse']);self.close(mse,metric_index[(g['seed'],g['task'],g['n'],r['method'])][split+'_mse']);n+=1
  self.assertEqual(n,252);self.assertEqual(self.result['events'][-2:],['all_validation_selections_frozen','test_labels_parsed_after_selection'])
  # Live call order: perturb source training to fail and ensure test parsing never begins.
  original=e.train;parse=e.parse;seen=[]
  def fail(*args,**kwargs):raise RuntimeError('training deliberately stopped')
  def record(raw,name):seen.append(name);return parse(raw,name)
  try:
   e.train=fail;e.parse=record
   with self.assertRaisesRegex(RuntimeError,'deliberately'):e.run()
  finally:e.train=original;e.parse=parse
  self.assertNotIn('target_test',seen)
 def test_negative_transfer_is_paired_and_bounded(self):
  ix={(r['seed'],r['task'],r['n'],r['method']):r['test_mse'] for r in self.result['metrics']}
  for seed in e.SEEDS:
   for n in e.LABELS:
    scratch=ix[(seed,'orthogonal',n,'scratch')];self.assertGreater(ix[(seed,'orthogonal',n,'finetune_small')]-scratch,.4);self.assertLess(ix[(seed,'orthogonal',n,'finetune_equal')],.02)
 def test_input_validation_no_bare_assert(self):
  for v in [True,0,-1,1.5,'160',float('nan'),2001]:
   with self.assertRaises(ValueError):e.integer(v,'steps')
  for bad in [[True,0,0,1,0],['.1',0,0,1,0],[float('nan'),0,0,1,0],[float('inf'),0,0,1,0],[1e7,0,0,1,0],[1,2]]:
   with self.assertRaises(ValueError):e.forward(bad,[[1,2]])
  with self.assertRaises(ValueError):e.forward([1,1,0,1,0],[])
 def test_fixture_and_serialization_failure_no_output(self):
  import shutil
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d=root/'data';shutil.copytree(e.ROOT/'data',d);out=root/'out'
   (d/'target_test.csv').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'hash mismatch'):e.run(out,d)
   self.assertFalse(out.exists());(d/'target_test.csv').unlink()
   with self.assertRaisesRegex(ValueError,'missing fixture'):e.run(out,d)
   self.assertFalse(out.exists())
   with self.assertRaises(ValueError):e.serialize({'a':float('nan')})
   with self.assertRaises(ValueError):e.commit(out,{'x':'not bytes'})
   self.assertFalse(out.exists())
 def test_packed_integrity(self):
  s=json.loads((e.ROOT/'outputs/summary.json').read_text())['packed_result'];b=(e.ROOT/'outputs/results.json.gz').read_bytes();raw=gzip.decompress(b)
  self.assertEqual(e.digest(b),s['packed_sha256']);self.assertEqual(e.digest(raw),s['raw_sha256']);self.assertEqual(e.pack(raw),b);self.assertLess(len(b)*4/3,16*1024**2)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Test041));print(json.dumps(ERRORS,sort_keys=True))
 raise SystemExit(0 if result.wasSuccessful() else 1)
