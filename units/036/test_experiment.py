"""Independent scalar Decimal and NumPy update references, plus execution contracts."""
import copy,decimal,hashlib,json,math,tempfile,unittest
from decimal import Decimal as D
from pathlib import Path
import numpy as np
import torch
import experiment as e
C=decimal.Context(prec=75)
def scalar_loss(theta,X,y,eps):
 with decimal.localcontext(C):
  t=[D(str(v)) for v in theta];x=[[D(str(v)) for v in row] for row in X];target=[D(str(v)) for v in y];ep=D(str(eps));n=len(x)
  z=[[sum(x[i][k]*t[k*2+j] for k in range(2))+t[4+j] for j in range(2)] for i in range(n)]
  mu=[sum(row[j] for row in z)/n for j in range(2)];v=[sum((row[j]-mu[j])**2 for row in z)/n for j in range(2)];s=[(v[j]+ep).sqrt() for j in range(2)]
  q=[[(z[i][j]-mu[j])/s[j] for j in range(2)] for i in range(n)];p=[sum((q[i][j]*t[6+j]+t[8+j])*t[10+j] for j in range(2))+t[12] for i in range(n)]
  return sum((p[i]-target[i])**2 for i in range(n))/(2*n)
def decimal_grad(theta,X,y,eps,inputs=False):
 with decimal.localcontext(C):
  h=D('1e-25');g=[];t=[D(str(v)) for v in theta];x=[[D(str(v)) for v in row] for row in X]
  for k in range(6 if inputs else 13):
   tp=t.copy();tm=t.copy();xp=copy.deepcopy(x);xm=copy.deepcopy(x)
   if inputs:xp[k//2][k%2]+=h;xm[k//2][k%2]-=h
   else:tp[k]+=h;tm[k]-=h
   g.append(float((scalar_loss(tp,xp,y,eps)-scalar_loss(tm,xm,y,eps))/(2*h)))
  return np.array(g).reshape(3,2) if inputs else np.array(g)
def shadow_train(data,B,seed):
 X=np.array(data['train']['X']);y=np.array(data['train']['y']);vx=np.array(data['validation']['X']);vy=np.array(data['validation']['y']);W=np.array([[.6,-.4,.3,-.5],[-.2,.5,.7,-.3]]);b=np.array([.1,-.2,.05,.2]);gamma=np.ones(4);beta=np.zeros(4);u=np.array([.4,-.3,.2,.1]);c=.05;rm=np.zeros(4);rv=np.ones(4);rng=np.random.default_rng(seed);history=[]
 for epoch in range(8):
  ids=rng.permutation(64)
  for start in range(0,64,B):
   a=X[ids[start:start+B]];target=y[ids[start:start+B]];z=a@W+b;mu=z.mean(0);cent=z-mu;var=np.mean(cent**2,axis=0);s=np.sqrt(var+.001);q=cent/s;h=q*gamma+beta;v=np.tanh(h);p=v@u+c;r=(p-target)/B
   du=v.T@r;dc=r.sum();dh=r[:,None]*u*(1-v*v);dg=(dh*q).sum(0);dbeta=dh.sum(0);dq=dh*gamma;dz=(dq-dq.mean(0)-q*(dq*q).mean(0))/s;dw=a.T@dz;db=dz.sum(0);rm=.9*rm+.1*mu;rv=.9*rv+.1*var*B/(B-1)
   W-=.03*dw;b-=.03*db;gamma-=.03*dg;beta-=.03*dbeta;u-=.03*du;c-=.03*dc
  h=((vx@W+b-rm)/np.sqrt(rv+.001))*gamma+beta;pred=np.tanh(h)@u+c;history.append(float(np.mean((pred-vy)**2)/2))
 return {'first.weight':W.T,'first.bias':b,'norm.weight':gamma,'norm.bias':beta,'norm.running_mean':rm,'norm.running_var':rv,'norm.num_batches_tracked':8*64//B,'last.weight':u[None,:],'last.bias':[c]},history
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.cfg,cls.hand,cls.data=e.load_inputs()
 def close(self,a,b,tol=2e-12):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
 def test_01_hand_two_steps(self):
  trace=e.hand_trace(self.hand)
  for step in trace['steps']:
   t=step['theta_before'];f=step['forward_backward'];ref=e.torch_hand(t,self.hand['X'],self.hand['y'],self.hand['eps']);self.close(f['gradient'],ref['gradient']);self.close(f['chain']['dX'],ref['dX']);self.close(f['P'],ref['P']);self.close(np.sum(f['contributions'],axis=0),f['gradient']);self.close(np.array(f['chain']['dZ_direct'])+f['chain']['dZ_variance_path']+f['chain']['dZ_mean_path'],f['chain']['dZ']);self.close(step['theta_after'],np.array(t)-.1*np.array(f['gradient']));self.close(f['gradient'],decimal_grad(t,self.hand['X'],self.hand['y'],self.hand['eps']));self.close(f['chain']['dX'],decimal_grad(t,self.hand['X'],self.hand['y'],self.hand['eps'],True))
 def test_02_fraction_first_step(self):
  from fractions import Fraction as F
  x=[[-1,0],[0,1],[1,2]];q=[-1,0,1];u=[F(3,5),F(-2,5)];gamma=[F(1),F(1,2)];beta=[F(1,4),F(-1,10)];p=[sum((q[i]*gamma[j]+beta[j])*u[j] for j in range(2))+F(1,20) for i in range(3)];r=[(p[i]-[F(1,5),F(-1,10),F(4,5)][i])/3 for i in range(3)];J=[[F(int(i==k))-F(1,3)-F(q[i]*q[k],3) for k in range(3)] for i in range(3)];rows=[]
  for i in range(3):
   dz=[[r[i]*u[j]*gamma[j]*J[i][k] for j in range(2)] for k in range(3)];rows.append([sum(x[k][a]*dz[k][j] for k in range(3)) for a in range(2) for j in range(2)]+[sum(dz[k][j] for k in range(3)) for j in range(2)]+[r[i]*u[j]*q[i] for j in range(2)]+[r[i]*u[j] for j in range(2)]+[r[i]*(q[i]*gamma[j]+beta[j]) for j in range(2)]+[r[i]])
  f=e.hand_backward(self.hand['theta'],self.hand['X'],self.hand['y'],self.hand['eps']);self.close(f['contributions'],[[float(v) for v in row] for row in rows]);self.close(f['loss'],float(sum((3*v)**2 for v in r)/6))
 def test_03_random_decimal_and_input_gradients(self):
  rng=np.random.default_rng(3699)
  for _ in range(32):
   t=rng.normal(size=13);x=rng.normal(size=(3,2));y=rng.normal(size=3);ep=float(rng.uniform(.02,.7));f=e.hand_backward(t,x,y,ep);self.close(f['gradient'],decimal_grad(t,x,y,ep),5e-11);self.close(f['chain']['dX'],decimal_grad(t,x,y,ep,True),5e-11);self.close(f['gradient'],e.torch_hand(t,x,y,ep)['gradient'],5e-11)
 def test_04_every_loss_path(self):
  f=e.hand_backward(self.hand['theta'],self.hand['X'],self.hand['y'],self.hand['eps']);t=torch.tensor(self.hand['theta'],dtype=e.DT,requires_grad=True);x=torch.tensor(self.hand['X'],dtype=e.DT,requires_grad=True);z=x@t[:4].reshape(2,2)+t[4:6];z.retain_grad();h=torch.nn.functional.batch_norm(z,None,None,t[6:8],t[8:10],True,0.,1/3);p=h@t[10:12]+t[12]
  for i in range(3):
   loss=(p[i]-self.hand['y'][i])**2/6;g,dz=torch.autograd.grad(loss,(t,z),retain_graph=True);self.close(g.detach().numpy(),f['contributions'][i]);self.close(dz.detach().numpy(),f['chain']['loss_path_input_gradients'][i])
 def test_05_jacobian_invariants(self):
  rng=np.random.default_rng(42)
  for _ in range(80):
   t=rng.normal(size=13);x=rng.normal(size=(3,2));y=rng.normal(size=3);eps=.03;f=e.hand_backward(t,x,y,eps)
   for j,J in enumerate(f['local']['jacobian_q_by_z']):
    self.close(J@np.ones(3),np.zeros(3));self.close(J@f['Q'][:,j],eps/f['S'][0,j]**3*f['Q'][:,j]);self.close(J,J.T)
   self.close(f['gradient'][4:6],[0,0]);self.close(f['chain']['dZ'].sum(0),[0,0])
 def test_06_axes_and_shapes(self):
  self.close(e.axes_demo()['torch_max_error'],[0,0,0]);rng=np.random.default_rng(5)
  for N,T,D in [(1,4,2),(2,3,4),(4,1,3)]:
   x=rng.normal(size=(N,T,D));a=e.norm_forward(x,(0,1),.1);b=e.norm_forward(x,(2,),.1);self.close(a['q'].mean((0,1)),np.zeros(D));self.close(b['q'].mean(2),np.zeros((N,T)));self.close((a['q']**2).mean((0,1)),a['var'].reshape(-1)/(a['var'].reshape(-1)+.1))
 def test_07_modes(self):
  r=e.modes_and_buffers()
  for row in r['mode_matrix']:
   self.assertEqual(row['batches'],int(row['train']));self.assertEqual(row['output_requires_grad'],row['grad_enabled']);self.close(row['running_mean'],[.2,0] if row['train'] else [0,0]);self.close(row['running_var'],[1,1])
  self.assertTrue(r['singleton_BN_error']);self.close(r['LN_D1_output'],[[0],[0]]);self.close(r['cumulative_unequal']['running_mean'],[6]);self.close(r['cumulative_unequal']['pooled_mean'],[23/3]);self.close(r['cumulative_unequal']['running_var'],[5/3]);self.assertFalse(r['state_dict_does_not_restore_training_flag']['destination_training'])
 def test_08_microbatch_and_peers(self):
  self.assertGreater(e.microbatch_example(self.hand)['max_gradient_gap'],.01);rows,d=e.batch_sensitivity(self.cfg);self.assertEqual(len(rows),800);p=d['peer_change'];self.close(p['LN_before'],p['LN_after']);self.assertGreater(np.max(abs(np.array(p['BN_before'])-p['BN_after'])),.1)
  for r in d['running_shift']:self.assertLess(r['torch_mean_error'],1e-12);self.assertLess(r['torch_var_error'],1e-12)
 def test_09_shadow_all_480_updates(self):
  rows,details=e.training_demo(self.data,self.cfg)
  for detail in details:
   state,hist=shadow_train(self.data,detail['batch_size'],detail['shuffle_seed'])
   for k,v in state.items():self.close(v,detail['final_state'][k],2e-11)
   self.close(hist,[h['validation_half_mse'] for h in detail['history']],2e-11)
  for r in rows:self.assertEqual(r['examples'],512);self.assertEqual(r['num_batches_after_wrong']-r['num_batches_before'],8);self.assertLess(r['eval_partition_max_gap'],1e-12);self.assertGreater(r['running_mean_max_change'],.05)
 def test_10_fixed_input_guards(self):
  for name,sha in e.INPUT_HASHES.items():self.assertEqual(hashlib.sha256((e.ROOT/name).read_bytes()).hexdigest(),sha)
  for eps in [0,-1,float('nan'),float('inf'),True]:
   with self.assertRaises(ValueError):e.norm_forward([[1,2]],(0,),eps)
  for axes in [(),(0,0),(-1,),[0],(2,)]:
   with self.assertRaises(ValueError):e.norm_forward([[1,2]],axes,.01)
  with self.assertRaises(ValueError):e.hand_forward([0]*12,self.hand['X'],self.hand['y'],.1)
  with self.assertRaises(ValueError):e.norm_forward([[float('nan')]],(0,),.1)
 def test_11_write_validation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'out';p.mkdir();(p/'old').write_bytes(b'safe')
   for data in [{'../bad':b'x'},{'ok':'notbytes'}]:
    with self.assertRaises(ValueError):e.write_payloads(data,p)
    self.assertEqual((p/'old').read_bytes(),b'safe');self.assertEqual(list(p.iterdir()),[p/'old'])
 def test_12_training_bias_invariance(self):
  t=np.array(self.hand['theta']);a=e.hand_forward(t,self.hand['X'],self.hand['y'],1/3);t[4:6]+=[4,-9];b=e.hand_forward(t,self.hand['X'],self.hand['y'],1/3);self.close(a['P'],b['P'])
 def test_13_untracked_eval(self):
  r=e.modes_and_buffers()['untracked_eval'];self.assertTrue(r['running_mean_is_none']);self.close(r['train_output'],r['eval_output'])
 def test_14_LN_numeric_and_jacobian(self):
  x=torch.tensor([[1.,3.],[2.,2.]],dtype=e.DT,requires_grad=True);gamma=torch.tensor([2.,1.],dtype=e.DT);beta=torch.tensor([.1,-.2],dtype=e.DT);out=torch.nn.functional.layer_norm(x,(2,),gamma,beta,1.);self.close(out.detach(),[[-2/math.sqrt(2)+.1,1/math.sqrt(2)-.2],[.1,-.2]])
  grad=torch.autograd.grad(out[1,0],x)[0];self.close(grad,[[0,0],[1,-1]])
 def test_15_eval_input_gradient(self):
  h=self.hand;t=np.array(h['theta']);r=e.torch_hand(t,h['X'],h['y'],h['eps'],False,[[.2,.3],[1.5,.7]]);W,b,gamma,beta,u,c=e.unpack(t);X=np.array(h['X']);s=np.sqrt(np.array([1.5,.7])+h['eps']);p=((X@W+b-[.2,.3])/s*gamma+beta)@u+c;dx=((p-np.array(h['y']))[:,None]/3*u*gamma/s)@W.T;self.close(r['dX'],dx)
if __name__=='__main__':unittest.main(verbosity=2)
