"""Independent scalar/Decimal references; run: python -m unittest -v test_experiment."""
import copy,hashlib,json,math,tempfile,unittest
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import experiment as e

def scalar_network(theta,x,y):
 t=list(map(float,theta));grad=[0.]*9;loss=0.;allrows=[]
 for row,target in zip(x,y):
  u,v=map(float,row);z=[u*t[0]+v*t[2]+t[4],u*t[1]+v*t[3]+t[5]];h=[max(zj,0.) for zj in z];p=h[0]*t[6]+h[1]*t[7]+t[8];r=p-float(target[0]);loss+=r*r/(2*len(x));dp=r/len(x)
  dz=[dp*t[j+6]*(z[j]>0) for j in range(2)];local=[u*dz[0],u*dz[1],v*dz[0],v*dz[1],*dz,dp*h[0],dp*h[1],dp]
  grad=[a+b for a,b in zip(grad,local)];allrows.append((z,h,p,r,local))
 return loss,np.array(grad),allrows

def scalar_fit(train,val,cfg,family,lr,wd,seed):
 q=list(cfg['initial_theta']);mom=[0.]*9;sq=[0.]*9;order=e.generate_order(seed,cfg['epochs'],len(train['X']));bs=cfg['batch_size'];history=[]
 for k in range(cfg['steps']):
  ids=order[k*bs:(k+1)*bs];_,g,_=scalar_network(q,train['X'][ids],train['y'][ids]);new=[]
  for j in range(9):
   lam=wd if j in (0,1,2,3,6,7) else 0.;d=float(g[j])+(lam*q[j] if family!='adamw' else 0.)
   if family=='sgd_momentum':
    mom[j]=cfg['momentum']*mom[j]+d;new.append(q[j]-lr*mom[j])
   else:
    b1,b2=cfg['betas'];mom[j]=b1*mom[j]+(1-b1)*d;sq[j]=b2*sq[j]+(1-b2)*d*d
    mean=mom[j]/(1-b1**(k+1));second=sq[j]/(1-b2**(k+1));update=lr*mean/(math.sqrt(second)+cfg['eps'])
    new.append(q[j]*(1-lr*lam if family=='adamw' else 1)-update)
  q=new;history.append(q)
 return np.array(q),scalar_network(q,train['X'],train['y'])[0],scalar_network(q,val['X'],val['y'])[0],history

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cls.cfg,cls.hand,cls.data=e.load_inputs()
 def test_01_full_chain_scalar_and_torch(self):
  rng=np.random.default_rng(341);checked=0
  for k in range(100):
   theta=rng.uniform(-.9,.9,9);x=rng.uniform(-1,1,(3,2));y=rng.uniform(-1,1,(3,1));a=e.manual(theta,x,y);L,g,rows=scalar_network(theta,x,y)
   self.assertAlmostEqual(L,a['loss'],places=14);np.testing.assert_allclose(g,a['gradient'],atol=2e-15,rtol=2e-14)
   p=torch.tensor(theta,requires_grad=True);xt=torch.tensor(x,requires_grad=True);lt=e.tensor_loss(p,xt,torch.tensor(y));lt.backward()
   np.testing.assert_allclose(p.grad.numpy(),g,atol=2e-15,rtol=2e-14);np.testing.assert_allclose(xt.grad.numpy(),a['chain']['dL_dX'],atol=2e-15,rtol=2e-14);checked+=1
  self.assertEqual(checked,100)
 def test_02_two_step_trace_finite_difference(self):
  trace=e.hand_trace(self.hand)
  for s in trace['steps']:
   f=s['forward_backward'];theta=np.array(f['theta']);g=np.array(f['gradient']);h=1e-6
   for j in range(9):
    plus=theta.copy();minus=theta.copy();plus[j]+=h;minus[j]-=h
    fd=(scalar_network(plus,f['X'],f['y'])[0]-scalar_network(minus,f['X'],f['y'])[0])/(2*h)
    self.assertAlmostEqual(fd,g[j],delta=3e-11)
   np.testing.assert_allclose(np.sum(f['contributions'],axis=0),g,atol=1e-15)
 def test_03_decimal_adam_sequences(self):
  rng=np.random.default_rng(342)
  with localcontext() as ctx:
   ctx.prec=75
   for scenario in range(90):
    b1=Decimal(str([.0,.5,.8,.9,.99][scenario%5]));b2=Decimal(str([.0,.5,.9,.99,.999][scenario%5]));lr=Decimal('.02');eps=Decimal('.001');wd=Decimal('.1');decoupled=scenario%2==0
    t=[Decimal('0.3'),Decimal('-0.4')];m=[Decimal(0)]*2;v=[Decimal(0)]*2;nt=np.array([.3,-.4]);nm=np.zeros(2);nv=np.zeros(2)
    for k in range(1,8):
     gs=[Decimal(int(z))/Decimal(10) for z in rng.integers(-9,10,2)];expected=[]
     for j in range(2):
      d=gs[j] if decoupled else gs[j]+wd*t[j];m[j]=b1*m[j]+(1-b1)*d;v[j]=b2*v[j]+(1-b2)*d*d
      mh=m[j]/(1-b1**k);vh=v[j]/(1-b2**k);expected.append(t[j]*(1-lr*wd if decoupled else 1)-lr*mh/(vh.sqrt()+eps))
     r=e.adam_step(nt,[float(z) for z in gs],nm,nv,k,float(lr),(float(b1),float(b2)),float(eps),float(wd),decoupled)
     np.testing.assert_allclose(r['theta_after'],list(map(float,expected)),atol=3e-14,rtol=3e-14)
     np.testing.assert_allclose(r['m'],list(map(float,m)),atol=2e-15,rtol=2e-14);np.testing.assert_allclose(r['v'],list(map(float,v)),atol=2e-15,rtol=2e-14)
     t=expected;nt,nm,nv=r['theta_after'],r['m'],r['v']
 def test_04_exact_bias_correction_weights(self):
  for beta in (Fraction(0),Fraction(1,2),Fraction(4,5),Fraction(9,10)):
   for t in range(1,31):
    weights=[(1-beta)*beta**(t-k) for k in range(1,t+1)]
    self.assertEqual(sum(weights),1-beta**t);self.assertEqual(sum(w/(1-beta**t) for w in weights),1)
    self.assertEqual(sum(w*Fraction(3,7) for w in weights)/(1-beta**t),Fraction(3,7))
 def test_05_torch_decay_and_epsilon_semantics(self):
  m=e.mechanisms();self.assertNotEqual(m['decay_comparison']['adam_l2'][-1]['theta_after'].tolist(),m['decay_comparison']['adamw'][-1]['theta_after'].tolist())
  self.assertEqual(m['none_vs_zero']['none']['step_after'],1);self.assertEqual(m['none_vs_zero']['zero']['step_after'],2)
  self.assertEqual(m['none_vs_zero']['none']['after'],m['none_vs_zero']['none']['before']);self.assertLess(m['none_vs_zero']['zero']['after'],m['none_vs_zero']['zero']['before'])
  self.assertGreater(m['delayed_gradient']['normalized_magnitude'],1)
  for row in m['pure_decay']:self.assertAlmostEqual(row['product'],row['path'][-1],places=14)
 def test_06_adam_l2_equals_explicit_penalty(self):
  t=torch.tensor(self.hand['theta'],dtype=torch.float64);p=torch.nn.Parameter(t.clone());q=torch.nn.Parameter(t.clone());o1=torch.optim.Adam([p],lr=.01,weight_decay=.2,foreach=False,fused=False);o2=torch.optim.Adam([q],lr=.01,weight_decay=0.,foreach=False,fused=False);x=torch.tensor(self.hand['X'],dtype=torch.float64);y=torch.tensor(self.hand['y'],dtype=torch.float64)
  for _ in range(8):
   o1.zero_grad(set_to_none=True);o2.zero_grad(set_to_none=True);e.tensor_loss(p,x,y).backward();(e.tensor_loss(q,x,y)+.1*q.square().sum()).backward();o1.step();o2.step();torch.testing.assert_close(p,q,rtol=1e-14,atol=1e-14)
 def test_07_plain_sgd_equivalence_not_momentum(self):
  for a in range(-7,8):
   for b in range(-4,5):
    t=Fraction(a,3);g=Fraction(b,5);lr=Fraction(1,10);wd=Fraction(1,5)
    self.assertEqual(t-lr*(g+wd*t),(1-lr*wd)*t-lr*g)
  # Coupled decay goes into momentum after the first update.
  t1=(1-.1*.2)*1-.1*.3;coupled=t1-.1*(.8*(.3+.2*1)+.4+.2*t1);decoupled=(1-.1*.2)*t1-.1*(.8*.3+.4)
  self.assertGreater(abs(coupled-decoupled),1e-4)
 def test_08_all_grid_runs_independent(self):
  selection,rows,cache=e.search(self.data['train'],self.data['validation'],self.cfg)
  for row in rows:
   q,tr,va,history=scalar_fit(self.data['train'],self.data['validation'],self.cfg,row['family'],row['lr'],row['weight_decay'],row['seed']);actual=cache[(row['family'],row['grid_index'],row['seed'])]['final']
   np.testing.assert_allclose(q,actual['theta'],atol=3e-12,rtol=3e-12);self.assertAlmostEqual(tr,actual['train_half_mse'],delta=3e-13);self.assertAlmostEqual(va,row['validation_half_mse'],delta=3e-13)
   self.assertEqual(row['optimizer_steps'],96);self.assertEqual(row['training_examples'],768)
  self.assertEqual(len(rows),81);self.assertEqual({r['order_sha256'] for r in rows if r['seed']==340},{rows[0]['order_sha256']})
  for family in e.FAMILIES:self.assertEqual(sum(r['optimizer_steps'] for r in rows if r['family']==family),2592)
  self.assertEqual(selection['selected'][1]['lr'],.01);self.assertEqual(selection['selected'][1]['weight_decay'],0.)
 def test_09_selected_each_update_reference(self):
  histories=json.loads((e.ROOT/'outputs/selected-trajectories.json').read_text())
  for run in histories:
   q,tr,va,reference=scalar_fit(self.data['train'],self.data['validation'],self.cfg,run['family'],run['lr'],run['weight_decay'],run['seed'])
   for actual,expected in zip(run['history'][1:],reference):np.testing.assert_allclose(actual['theta'],expected,atol=3e-12,rtol=3e-12)
 def test_10_test_not_used_in_search(self):
  # Fit/search APIs receive only two splits. Test labels enter only frozen evaluation.
  import inspect
  self.assertEqual(list(inspect.signature(e.search).parameters),['train','validation','cfg'])
  selected=json.loads((e.ROOT/'outputs/selection.json').read_text());report=json.loads((e.ROOT/'outputs/test-report.json').read_text())
  self.assertEqual(report['selection_sha256'],hashlib.sha256(e.canonical(selected)).hexdigest());self.assertFalse(selected['test_seen'])
  histories=json.loads((e.ROOT/'outputs/selected-trajectories.json').read_text())
  for row in report['rows']:
   run=next(r for r in histories if r['family']==row['family'] and r['seed']==row['seed']);L,_,_=scalar_network(run['final']['theta'],self.data['test']['X'],self.data['test']['y']);self.assertAlmostEqual(L,row['test_half_mse'],places=14)
 def test_11_biases_excluded_from_decay(self):
  for family in e.FAMILIES:
   p=e.init_parameters(self.cfg['initial_theta']);o=e.make_optimizer(p,family,.1,.2,self.cfg);before=[q.detach().clone() for q in p]
   for q in p:q.grad=torch.zeros_like(q)
   o.step();self.assertTrue(torch.equal(p[1],before[1]));self.assertTrue(torch.equal(p[3],before[3]));self.assertFalse(torch.equal(p[0],before[0]))
  for mask in ([1]*9,[True,1,1,1,0,0,1,1,0],None):
   bad=copy.deepcopy(self.cfg);bad['decay_mask']=mask
   with self.assertRaises(ValueError):e.make_optimizer(e.init_parameters(self.cfg['initial_theta']),'adamw',.1,.2,bad)
 def test_12_resume_full_state(self):
  r=e.state_demo(self.hand);self.assertEqual(r['full_theta'],r['resumed_theta']);self.assertGreater(r['weights_only_max_gap'],.01)
 def test_13_guard_rejects_before_calculation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name in e.INPUT_HASHES:
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((e.ROOT/name).read_bytes())
   for name in e.INPUT_HASHES:
    p=root/name;old=p.read_bytes();p.write_bytes(old+b' ')
    with self.assertRaisesRegex(ValueError,'fixed teaching input'):e.load_inputs(root)
    p.write_bytes(old)
 def test_14_bad_domain_rejected(self):
  for g in ([True],[float('nan')],[101.],['1']):
   with self.assertRaises(ValueError):e.adam_step([1.],g,[0.],[0.],1)
  for key,value in [('lr',True),('lr',0.),('lr',float('inf')),('eps',0.),('eps',float('nan')),('weight_decay',-.1),('betas',(1.,.9)),('betas',(.9,True)),('decoupled',1),('mask',[.5])]:
   with self.assertRaises(ValueError):e.adam_step([1.],[.1],[0.],[0.],1,**{key:value})
  for s in (True,0,-1,1.5):
   with self.assertRaises(ValueError):e.adam_step([1.],[.1],[0.],[0.],s)
  with self.assertRaises(ValueError):e.adam_step([1.],[.1],[0.],[-.1],1)
  for theta,x,y in [([0.]*8,[[1,2]],[[1]]),([0.]*9,[[1,2]],[1]),([0.]*9,[[True,2]],[[1]])]:
   with self.assertRaises(ValueError):e.manual(theta,x,y)
 def test_15_serialization_disallows_nonfinite(self):
  with self.assertRaises(ValueError):e.canonical({'x':float('nan')})
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'kept.json';p.write_bytes(b'keep');
   with patch('sys.argv',['experiment.py','--output',d]),patch.object(e,'run',side_effect=ArithmeticError('injected')):
    with self.assertRaises(ArithmeticError):e.main()
   self.assertEqual(list(Path(d).iterdir()),[p]);self.assertEqual(p.read_bytes(),b'keep')
if __name__=='__main__':unittest.main(verbosity=2)
