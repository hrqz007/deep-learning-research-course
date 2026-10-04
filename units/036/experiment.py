"""CPU float64 normalization teaching experiments; no network or dataset download."""
from pathlib import Path
import argparse, copy, csv, hashlib, io, json, math, os, tempfile
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
DT=torch.float64
ORDER=['W11','W12','W21','W22','b1','b2','gamma1','gamma2','beta1','beta2','u1','u2','c']
INPUT_HASHES={'data/config.json': '0182da5e95f6cf094ca7b7c9c8dd0214160bb9032034e824051a1e5177dc5d74', 'data/hand-example.json': 'c2e88058701be66ea72f1df58355950fc0e24fc7f40287db2de14f78c42d529e', 'data/samples.json': '700c21b235136c86c6ca4f7f32ed87976028fd9334f2de169d02a375c6d163db'}
def canonical(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def clean(x):
 if isinstance(x,(np.ndarray,torch.Tensor)):return clean(x.detach().cpu().numpy().tolist() if isinstance(x,torch.Tensor) else x.tolist())
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return x.item()
 return x
def finite(x,name):
 a=np.asarray(x,dtype=float)
 if not np.isfinite(a).all():raise ValueError(name+' must be finite')
 return a
def norm_forward(z,axes,eps,gamma=1.,beta=0.):
 z=finite(z,'z')
 if isinstance(eps,bool) or not isinstance(eps,(int,float)) or not math.isfinite(eps) or eps<=0:raise ValueError('eps must be positive finite')
 if not isinstance(axes,tuple) or not axes or len(set(axes))!=len(axes) or any(type(a)!=int or a<0 or a>=z.ndim for a in axes):raise ValueError('unique nonnegative axes required')
 if any(z.shape[a]==0 for a in axes):raise ValueError('empty normalization group')
 mu=z.mean(axis=axes,keepdims=True);center=z-mu;var=(center**2).mean(axis=axes,keepdims=True);s=np.sqrt(var+eps);q=center/s
 return {'mu':mu,'center':center,'var':var,'s':s,'q':q,'h':q*gamma+beta}
def unpack(theta):
 t=finite(theta,'theta')
 if t.shape!=(13,):raise ValueError('theta must have13 coordinates')
 return t[:4].reshape(2,2),t[4:6],t[6:8],t[8:10],t[10:12],t[12]
def hand_forward(theta,X,y,eps):
 X=finite(X,'X');y=finite(y,'y')
 if X.shape!=(3,2) or y.shape!=(3,):raise ValueError('hand example requires X3x2,y3')
 W,b,gamma,beta,u,c=unpack(theta);z=X@W+b;n=norm_forward(z,(0,),eps,gamma,beta);p=n['h']@u+c;e=p-y
 return dict(X=X,y=y,Z=z,**{k.upper():v for k,v in n.items()},P=p,residual=e,sample_half_squared_error=e**2/2,loss=float(np.mean(e**2)/2))
def hand_backward(theta,X,y,eps):
 f=hand_forward(theta,X,y,eps);W,b,gamma,beta,u,c=unpack(theta);X=f['X'];q=f['Q'];s=f['S'][0];center=f['CENTER'];dP=f['residual']/3;dH=dP[:,None]*u;dQ=dH*gamma
 J=np.stack([(np.eye(3)-np.ones((3,3))/3-np.outer(q[:,j],q[:,j])/3)/s[j] for j in range(2)])
 # loss-path x input-row x feature. Each loss also changes the other rows.
 paths=np.stack([J[j].T*dQ[:,j] for j in range(2)],axis=-1).transpose(1,0,2)
 dZ=paths.sum(axis=0)
 dvar=np.sum(dQ*center*(-.5)*(f['VAR']+eps)**(-1.5),axis=0)
 dmu=np.sum(-dQ/f['S'],axis=0)+dvar*np.mean(-2*center,axis=0)
 direct=dQ/f['S'];via_var=2*center*dvar/3;via_mu=np.broadcast_to(dmu/3,(3,2))
 contributions=[]
 for i in range(3):
  dz=paths[i];row=np.r_[(X.T@dz).ravel(),dz.sum(0),dH[i]*q[i],dH[i],dP[i]*f['H'][i],dP[i]];contributions.append(row)
 contributions=np.array(contributions);g=contributions.sum(0)
 f.update(local={'d_loss_d_p':dP,'d_p_d_h':u,'d_h_d_q':gamma,'d_h_d_gamma':q,'d_h_d_beta':np.ones_like(q),'d_mu_d_z':np.ones(3)/3,'d_var_d_z':2*center/3,'d_q_d_var':center*(-.5)*(f['VAR']+eps)**(-1.5),'d_z_d_W':X,'d_z_d_b':1,'jacobian_q_by_z':J},chain={'dH':dH,'dQ':dQ,'dvar':dvar,'dmu':dmu,'dZ_direct':direct,'dZ_variance_path':via_var,'dZ_mean_path':via_mu,'loss_path_input_gradients':paths,'dZ':dZ,'dX':dZ@W.T},contributions=contributions,gradient=g)
 return f
def torch_hand(theta,X,y,eps,training=True,running=None):
 t=torch.tensor(theta,dtype=DT,requires_grad=True);x=torch.tensor(X,dtype=DT,requires_grad=True);z=x@t[:4].reshape(2,2)+t[4:6]
 h=torch.nn.functional.batch_norm(z,None if running is None else torch.tensor(running[0],dtype=DT),None if running is None else torch.tensor(running[1],dtype=DT),t[6:8],t[8:10],training=training,momentum=.2,eps=eps)
 p=h@t[10:12]+t[12];loss=((p-torch.tensor(y,dtype=DT))**2).mean()/2
 g,dx=torch.autograd.grad(loss,(t,x));return clean({'P':p,'loss':loss,'gradient':g,'dX':dx})
def hand_trace(hand):
 theta=np.array(hand['theta'],float);steps=[];rm=np.zeros(2);rv=np.ones(2)
 for step in range(2):
  f=hand_backward(theta,hand['X'],hand['y'],hand['eps']);previous={'mean':rm.copy(),'variance':rv.copy()};rm=.8*rm+.2*f['MU'][0];rv=.8*rv+.2*f['VAR'][0]*3/2
  new=theta-hand['lr']*f['gradient'];steps.append({'step':step+1,'theta_before':theta.copy(),'forward_backward':f,'delta':new-theta,'theta_after':new.copy(),'running_before':previous,'running_after':{'mean':rm.copy(),'variance':rv.copy(),'num_batches_tracked':step+1},'next_forward':hand_forward(new,hand['X'],hand['y'],hand['eps'])});theta=new
 eval_result=torch_hand(theta,hand['X'],hand['y'],hand['eps'],False,[rm,rv])
 return clean({'parameter_order':ORDER,'steps':steps,'eval_after_two_steps':eval_result,'note':'running buffers were updated with pre-update activations; diagnostic next_forward is a pure formula and does not update buffers'})
def axes_demo():
 x=np.array([[[0,2],[1,4],[2,6]],[[4,-2],[5,0],[6,2]]],float)
 bn=norm_forward(x,(0,1),.1);ln=norm_forward(x,(2,),.1);whole=norm_forward(x,(1,2),.1)
 layer=nn.BatchNorm1d(2,eps=.1,affine=False,track_running_stats=False).to(DT);a=layer(torch.tensor(x,dtype=DT).transpose(1,2)).transpose(1,2)
 b=nn.LayerNorm(2,eps=.1,elementwise_affine=False).to(DT)(torch.tensor(x,dtype=DT));c=nn.LayerNorm((3,2),eps=.1,elementwise_affine=False).to(DT)(torch.tensor(x,dtype=DT))
 return clean({'shape_N_T_D':list(x.shape),'X':x,'BN_axes_N_T':bn,'LN_axes_D':ln,'LN_axes_T_D':whole,'torch_max_error':[np.max(abs(a.detach().numpy()-bn['h'])),np.max(abs(b.numpy()-ln['h'])),np.max(abs(c.numpy()-whole['h']))]})
def modes_and_buffers():
 z=torch.tensor([[0.,-1.],[1.,0.],[2.,1.]],dtype=DT);rows=[]
 for train in [True,False]:
  for grad in [True,False]:
   layer=nn.BatchNorm1d(2,eps=1/3,momentum=.2).to(DT);layer.train(train);x=z.clone().requires_grad_(True)
   with torch.set_grad_enabled(grad):h=layer(x)
   rows.append({'train':train,'grad_enabled':grad,'output_requires_grad':h.requires_grad,'running_mean':clean(layer.running_mean),'running_var':clean(layer.running_var),'batches':int(layer.num_batches_tracked),'output':clean(h)})
 frozen=nn.BatchNorm1d(2,eps=1/3,momentum=.2).to(DT)
 for p in frozen.parameters():p.requires_grad_(False)
 frozen(z)
 # Unequal-size batches make the cumulative average of batch means != pooled mean.
 cum=nn.BatchNorm1d(1,eps=.01,momentum=None,affine=False).to(DT);groups=[torch.tensor([[0.],[2.]],dtype=DT),torch.tensor([[10.],[10.],[12.],[12.]],dtype=DT)]
 for g in groups:cum(g)
 pooled=torch.cat(groups)
 one=nn.BatchNorm1d(2).to(DT);error=''
 try:one(torch.ones((1,2),dtype=DT))
 except ValueError as exc:error=str(exc).split('\n')[0]
 temporal=one(torch.tensor([[[1.,2.],[3.,4.]]],dtype=DT))
 singleton_ln=nn.LayerNorm(1).to(DT)(torch.tensor([[3.],[8.]],dtype=DT))
 state=copy.deepcopy(frozen.state_dict());clone=nn.BatchNorm1d(2,eps=1/3,momentum=.2).to(DT);clone.eval();clone.load_state_dict(state)
 # state_dict carries buffers, not the module.training Python flag.
 untracked=nn.BatchNorm1d(2,eps=1/3,affine=False,track_running_stats=False).to(DT);u_train=untracked(z);untracked.eval();u_eval=untracked(z)
 return clean({'untracked_eval':{'running_mean_is_none':untracked.running_mean is None,'train_output':u_train,'eval_output':u_eval},'mode_matrix':rows,'frozen_affine_training':{'running_mean':frozen.running_mean,'batches':int(frozen.num_batches_tracked)},'cumulative_unequal':{'batch_sizes':[2,4],'running_mean':cum.running_mean,'running_var':cum.running_var,'pooled_mean':pooled.mean(0),'pooled_unbiased_variance':pooled.var(0,unbiased=True)},'singleton_BN_error':error,'BN_N1_L2_output':temporal,'LN_D1_output':singleton_ln,'state_dict_keys':list(state),'state_dict_does_not_restore_training_flag':{'source_training':frozen.training,'destination_training':clone.training}})
def batch_sensitivity(cfg):
 rng=np.random.default_rng(cfg['sampling_seed']);rows=[]
 for B in cfg['batch_sizes']:
  for rep in range(cfg['repeats']):
   # Fixed anchor and B-1 iid peers: not a fully iid batch mean experiment.
   peers=rng.normal(size=(B-1,2));x=np.vstack([[.5,-.5],peers]);a=norm_forward(x,(0,),cfg['eps'])
   rows.append({'batch_size':B,'repeat':rep,'anchor_channel0':float(a['q'][0,0]),'mean_channel0':float(a['mu'][0,0]),'variance_channel0':float(a['var'][0,0])})
 anchor=np.array([[.5,-.5]]);x1=np.vstack([anchor,[[0,1],[1,0]]]);x2=np.vstack([anchor,[[5,6],[6,5]]]);ln1=norm_forward(x1,(1,),cfg['eps']);ln2=norm_forward(x2,(1,),cfg['eps']);bn1=norm_forward(x1,(0,),cfg['eps']);bn2=norm_forward(x2,(0,),cfg['eps'])
 shifts=[];rm=np.zeros(2);rv=np.ones(2);layer=nn.BatchNorm1d(2,eps=cfg['eps'],momentum=.1,affine=False).to(DT)
 for step in range(40):
  mean=0 if step<20 else 3;x=rng.normal(mean,1,size=(32,2));mu=x.mean(0);v=x.var(0,ddof=1);rm=.9*rm+.1*mu;rv=.9*rv+.1*v;layer(torch.tensor(x,dtype=DT))
  shifts.append({'step':step+1,'source_mean':mean,'batch_mean':float(mu[0]),'running_mean':float(rm[0]),'running_var':float(rv[0]),'torch_mean_error':float(np.max(abs(layer.running_mean.numpy()-rm))),'torch_var_error':float(np.max(abs(layer.running_var.numpy()-rv)))})
 return rows,clean({'peer_change':{'anchor':anchor[0],'BN_before':bn1['q'][0],'BN_after':bn2['q'][0],'LN_before':ln1['q'][0],'LN_after':ln2['q'][0]},'running_shift':shifts,'sampling_design':'same fixed anchor plus independent random peers;200 replicates for each specified batch size;descriptive finite experiment'})
class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.first=nn.Linear(2,4,dtype=DT);self.norm=nn.BatchNorm1d(4,eps=1e-3,momentum=.1,dtype=DT);self.last=nn.Linear(4,1,dtype=DT)
  with torch.no_grad():
   self.first.weight.copy_(torch.tensor([[.6,-.2],[-.4,.5],[.3,.7],[-.5,-.3]],dtype=DT));self.first.bias.copy_(torch.tensor([.1,-.2,.05,.2],dtype=DT));self.last.weight.copy_(torch.tensor([[.4,-.3,.2,.1]],dtype=DT));self.last.bias.fill_(.05)
 def forward(self,x):return self.last(torch.tanh(self.norm(self.first(x)))).squeeze(-1)
def model_snapshot(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}
def state_equal(a,b):return all(torch.equal(a[k],b[k]) for k in a)
def training_demo(data,cfg):
 X=torch.tensor(data['train']['X'],dtype=DT);y=torch.tensor(data['train']['y'],dtype=DT);vx=torch.tensor(data['validation']['X'],dtype=DT);vy=torch.tensor(data['validation']['y'],dtype=DT);rows=[];details=[]
 for B in [4,16]:
  for seed in cfg['shuffle_seeds']:
   m=Tiny();opt=torch.optim.SGD(m.parameters(),lr=.03,foreach=False);rng=np.random.default_rng(seed);history=[];steps=0
   for epoch in range(8):
    perm=rng.permutation(len(X));m.train()
    for start in range(0,len(X),B):
     idx=perm[start:start+B];opt.zero_grad(set_to_none=True);loss=((m(X[idx])-y[idx])**2).mean()/2;loss.backward();opt.step();steps+=1
    before=model_snapshot(m);m.eval()
    with torch.no_grad():pred=m(vx);vl=float(((pred-vy)**2).mean()/2)
    if not state_equal(before,model_snapshot(m)):raise ArithmeticError('eval mutated state')
    history.append({'epoch':epoch+1,'steps':steps,'validation_half_mse':vl})
   m.eval();before=model_snapshot(m)
   with torch.no_grad():one=m(vx);split=torch.cat([m(vx[i:i+4]) for i in range(0,len(vx),4)])
   correct=float(((one-vy)**2).mean()/2)
   bad=copy.deepcopy(m);bad.train()
   with torch.no_grad():wrong=torch.cat([bad(vx[i:i+4]) for i in range(0,len(vx),4)])
   wrong_loss=float(((wrong-vy)**2).mean()/2)
   # Re-evaluate in eval after accidental validation updates; captures contamination.
   bad.eval()
   with torch.no_grad():after=bad(vx)
   rows.append({'batch_size':B,'shuffle_seed':seed,'optimizer_steps':steps,'examples':8*len(X),'eval_half_mse':correct,'wrong_train_no_grad_half_mse':wrong_loss,'eval_after_contamination_half_mse':float(((after-vy)**2).mean()/2),'eval_partition_max_gap':float(torch.max(abs(one-split))),'wrong_prediction_max_gap':float(torch.max(abs(one-wrong))),'num_batches_before':int(before['norm.num_batches_tracked']),'num_batches_after_wrong':int(bad.norm.num_batches_tracked),'running_mean_max_change':float(torch.max(abs(before['norm.running_mean']-bad.norm.running_mean)))})
   details.append({'batch_size':B,'shuffle_seed':seed,'history':history,'final_state':clean(before),'correct_eval_predictions':clean(one),'wrong_train_predictions':clean(wrong)})
 return rows,details

def microbatch_example(hand):
 # Four rows so both two-row microbatches are valid BN groups.
 X=torch.tensor([[-1.,0.],[0.,1.],[1.,2.],[2.,-1.]],dtype=DT);y=torch.tensor([.2,-.1,.8,.4],dtype=DT);theta=torch.tensor(hand['theta'],dtype=DT,requires_grad=True)
 def loss(t,x,target):
  z=x@t[:4].reshape(2,2)+t[4:6];h=torch.nn.functional.batch_norm(z,None,None,t[6:8],t[8:10],True,0.,hand['eps']);p=h@t[10:12]+t[12];return ((p-target)**2).mean()/2
 full=loss(theta,X,y);gf=torch.autograd.grad(full,theta)[0];small=[loss(theta,X[i:i+2],y[i:i+2]) for i in [0,2]];avg=sum(small)/2;gs=torch.autograd.grad(avg,theta)[0]
 return clean({'full_loss':full,'weighted_microbatch_loss':avg,'full_gradient':gf,'weighted_microbatch_gradient':gs,'max_gradient_gap':torch.max(abs(gf-gs)),'note':'correct weighting cannot undo different normalization groups'})
def csv_bytes(rows):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return f.getvalue().encode()
def load_inputs():
 for name,sha in INPUT_HASHES.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Fixed teaching fixture changed: '+name)
 cfg=json.loads((ROOT/'data/config.json').read_text());hand=json.loads((ROOT/'data/hand-example.json').read_text());data=json.loads((ROOT/'data/samples.json').read_text());return cfg,hand,data
def run():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 cfg,hand,data=load_inputs();srows,sensitivity=batch_sensitivity(cfg);trows,details=training_demo(data,cfg)
 payload={'hand-trace.json':canonical(hand_trace(hand)),'axes.json':canonical(axes_demo()),'state-modes.json':canonical(modes_and_buffers()),'batch-sampling.csv':csv_bytes(srows),'batch-sensitivity.json':canonical(sensitivity),'training.csv':csv_bytes(trows),'training-detail.json':canonical(details),'microbatch.json':canonical(microbatch_example(hand))}
 # Serialization of every result precedes writing any final output.
 payload['summary.json']=canonical({'unit':'036','runtime':{'torch':torch.__version__,'numpy':np.__version__,'dtype':'float64','device':'cpu','threads':1},'hand_losses':[hand_trace(hand)['steps'][0]['forward_backward']['loss']]+[s['next_forward']['loss'] for s in hand_trace(hand)['steps']],'training_runs':len(trows),'training_examples_per_run':512,'optimizer_steps_by_batch_size':{'4':128,'16':32},'limits':['synthetic fixed dataset and initialization','three shuffle seeds only','equal examples do not mean equal optimizer steps or equal running-stat updates','no held-out test ranking or tuning','finite CPU checks;no GPU/cross-version guarantee']})
 return payload
def write_payloads(payload,out):
 out=Path(out)
 if out.resolve()==ROOT/'data':raise ValueError('Cannot replace teaching data')
 for name,b in payload.items():
  if Path(name).name!=name or not isinstance(b,bytes):raise ValueError('Expected flat byte payloads')
 out.mkdir(parents=True,exist_ok=True)
 for name,b in payload.items():
  fd,tmp=tempfile.mkstemp(prefix='.'+name,dir=out)
  try:
   with os.fdopen(fd,'wb') as f:f.write(b)
   os.replace(tmp,out/name)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args();result=run();write_payloads(result,a.output);print('wrote',len(result),'deterministic outputs')
