"""DL034: auditable Adam/AdamW updates and predeclared equal-budget search.

Offline synthetic CPU-only experiment. No cross-unit import, downloads or pickle.
The fixed driver checks teaching input bytes before calculating or writing.
"""
from pathlib import Path
import argparse,copy,csv,hashlib,io,json,math,os,platform,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
INPUT_HASHES={'data/samples.csv':'482391854b087c4d063921bae01af4ed2f11c83f19bb2b19c3a36d7a8874671c','data/hand-example.json':'ab398e1c13115642372e5994c6787ad07a07058cae720a472f592308a0b50c89','data/config.json':'44a32083abe6111dd15cb923e1af6c68b93bed4adbf44dcfe2ca779cf186f8d0'}
NAMES=['W11','W12','W21','W22','b1','b2','v1','v2','c']
FAMILIES=('sgd_momentum','adam','adamw')

def require(ok,message):
 if not ok:raise ValueError(message)

def scalar(x,name,lo,hi):
 require(type(x) in (int,float) and math.isfinite(x) and lo<=x<=hi,name+' must be a finite non-boolean scalar in the teaching domain')
 return float(x)

def integer(x,name,lo,hi):
 require(type(x) is int and lo<=x<=hi,name+' must be an integer in the teaching domain')
 return x

def array(x,name,shape=None):
 def has_bool(a):return isinstance(a,(bool,np.bool_)) or (isinstance(a,(list,tuple)) and any(has_bool(v) for v in a))
 require(not has_bool(x),name+' contains boolean')
 a=np.asarray(x);require(a.dtype.kind in 'fiu' and a.size>0,name+' must contain real numbers')
 a=np.array(a,dtype=np.float64,copy=True)
 require(np.isfinite(a).all() and np.max(np.abs(a))<=100,name+' must be finite and bounded by100')
 if shape is not None:require(a.shape==shape,name+' wrong shape')
 return a

def validate(theta,x,y):
 t=array(theta,'theta',(9,));x=array(x,'X');require(x.ndim==2 and x.shape[1]==2 and 1<=len(x)<=256,'X shape must be B by2')
 return t,x,array(y,'target',(len(x),1))

def load_inputs(root=ROOT):
 root=Path(root)
 for name,digest in INPUT_HASHES.items():require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'fixed teaching input changed: '+name)
 cfg=json.loads((root/'data/config.json').read_text());hand=json.loads((root/'data/hand-example.json').read_text())
 with (root/'data/samples.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 require([int(r['id']) for r in rows]==list(range(80)),'unexpected sample IDs')
 data={}
 for name,count in [('train',48),('validation',16),('test',16)]:
  subset=[r for r in rows if r['split']==name];require(len(subset)==count,'wrong split')
  x=np.array([[float(r['x1']),float(r['x2'])] for r in subset]);y=np.array([[float(r['y'])] for r in subset]);validate(cfg['initial_theta'],x,y)
  data[name]={'X':x,'y':y,'ids':[int(r['id']) for r in subset]}
 return cfg,hand,data

def manual(theta,x,y):
 t,x,y=validate(theta,x,y);b=len(x);w=t[:4].reshape(2,2);v=t[6:8]
 z=x@w+t[4:6];h=np.maximum(z,0);p=h@v[:,None]+t[8];r=p-y
 dp=r/b;mask=(z>0).astype(float);dh=dp*v;dz=dh*mask
 contributions=np.concatenate([np.einsum('ni,nj->nij',x,dz).reshape(b,4),dz,dp*h,dp],axis=1)
 gradient=contributions.sum(0)
 result={'theta':t,'X':x,'y':y,'Z':z,'H':h,'P':p,'residual':r,'sample_half_squared_error':r[:,0]**2/2,'loss':float(np.mean(r*r)/2),
  'shapes':{'X':list(x.shape),'W':[2,2],'b':[2],'Z':list(z.shape),'H':list(h.shape),'v':[2,1],'c':[],'P':list(p.shape),'contributions':list(contributions.shape)},
  'local':{'dL_dsample_loss':[1/b]*b,'dsample_loss_dP':r,'dP_dH':np.tile(v,(b,1)),'dP_dv':h,'dP_dc':[1.]*b,'dH_dZ':mask,'dZ_dW':x,'dZ_db':np.ones_like(z)},
  'chain':{'dL_dP':dp,'dL_dH':dh,'dL_dZ':dz,'dL_dX':dz@w.T},'contributions':contributions,'gradient':gradient}
 require(math.isfinite(result['loss']) and np.isfinite(gradient).all(),'nonfinite forward/backward')
 return result

def tensor_loss(theta,x,y):
 z=x@theta[:4].reshape(2,2)+theta[4:6];h=torch.relu(z);p=h@theta[6:8,None]+theta[8]
 require(p.shape==y.shape,'no target broadcasting')
 return .5*(p-y).square().mean()

def adam_step(theta,gradient,m,v,step,lr=.02,betas=(.8,.9),eps=.001,weight_decay=0.,decoupled=False,mask=None):
 theta=array(theta,'theta');require(theta.ndim==1,'theta must be vector');shape=theta.shape
 g=array(gradient,'gradient',shape);m=array(m,'m',shape);v=array(v,'v',shape);require(np.all(v>=0),'v must be nonnegative')
 integer(step,'step',1,100000);lr=scalar(lr,'lr',1e-10,1);eps=scalar(eps,'eps',1e-15,1);wd=scalar(weight_decay,'weight_decay',0,1)
 require(type(decoupled) is bool,'decoupled must be boolean');require(isinstance(betas,(list,tuple)) and len(betas)==2,'two betas required')
 b1,b2=[scalar(z,'beta',0,.99999) for z in betas]
 mask=np.ones(shape) if mask is None else array(mask,'decay mask',shape);require(np.all((mask==0)|(mask==1)),'decay mask must be0/1')
 effective=g if decoupled else g+wd*mask*theta
 newm=b1*m+(1-b1)*effective;newv=b2*v+(1-b2)*effective**2
 mh=newm/(1-b1**step);vh=newv/(1-b2**step);den=np.sqrt(vh)+eps;coefficient=lr/den
 decay=-lr*wd*mask*theta if decoupled else np.zeros(shape);adaptive=-coefficient*mh
 updated=theta+decay+adaptive
 require(np.isfinite(updated).all() and np.max(np.abs(updated))<=100,'Adam update left teaching domain')
 return {'theta_before':theta,'data_gradient':g,'effective_gradient':effective,'m_before':m,'v_before':v,'m':newm,'v':newv,'m_hat':mh,'v_hat':vh,'denominator':den,'coefficient_on_m_hat':coefficient,'adaptive_delta':adaptive,'decay_delta':decay,'theta_after':updated,'step':step}

def hand_trace(hand):
 t,x,y=validate(hand['theta'],hand['X'],hand['y']);m=np.zeros(9);v=np.zeros(9)
 p=torch.nn.Parameter(torch.tensor(t,dtype=torch.float64));opt=torch.optim.Adam([p],lr=hand['lr'],betas=tuple(hand['betas']),eps=hand['eps'],weight_decay=0,foreach=False,fused=False)
 tx=torch.tensor(x);ty=torch.tensor(y);steps=[]
 for k in range(1,hand['steps']+1):
  forward=manual(t,x,y);opt.zero_grad(set_to_none=True);loss=tensor_loss(p,tx,ty);loss.backward()
  require(np.allclose(p.grad.numpy(),forward['gradient'],rtol=1e-13,atol=1e-14),'hand gradient differs from autograd')
  update=adam_step(t,forward['gradient'],m,v,k,hand['lr'],hand['betas'],hand['eps']);opt.step()
  require(np.allclose(p.detach().numpy(),update['theta_after'],rtol=1e-13,atol=1e-14),'hand update differs from torch')
  t=update['theta_after'];m=update['m'];v=update['v'];after=manual(t,x,y)
  steps.append({'forward_backward':forward,'adam':update,'next_forward':{key:after[key] for key in ('Z','H','P','residual','sample_half_squared_error','loss')},'torch_theta':p.detach().tolist()})
 return {'parameter_order':NAMES,'config':hand,'steps':steps,'relu_at_zero':0,'all_parameters_receive_gradient_each_step':True}

def mechanisms():
 prescribed=[[.1,2.],[-.2,1.]];paths={}
 for decoupled in (False,True):
  t=np.array([1.,-2.]);m=np.zeros(2);v=np.zeros(2);rows=[]
  cls=torch.optim.AdamW if decoupled else torch.optim.Adam;p=torch.nn.Parameter(torch.tensor(t));opt=cls([p],lr=.1,betas=(.5,.5),eps=.01,weight_decay=.2,foreach=False,fused=False)
  for step,g in enumerate(prescribed,1):
   r=adam_step(t,g,m,v,step,.1,(.5,.5),.01,.2,decoupled);p.grad=torch.tensor(g,dtype=torch.float64);opt.step()
   require(np.allclose(p.detach().numpy(),r['theta_after'],rtol=1e-14,atol=1e-14),'decay comparison mismatch')
   t,m,v=r['theta_after'],r['m'],r['v'];rows.append(r)
  paths['adamw' if decoupled else 'adam_l2']=rows
 # Same preconditioner thought experiment, not a complete Adam trajectory.
 D=np.array([1.,10.]);theta=np.array([1.,-2.]);g=np.array([.1,2.]);lr=.1;wd=.2
 preconditioned={'theta':theta,'gradient':g,'D_diagonal':D,'l2_step':theta-lr*D*(g+wd*theta),'decoupled_step':theta-lr*D*g-lr*wd*theta,'explicit_scope':'one fixed diagonal preconditioner, not Adam with changed moments'}
 epsilon=[]
 for g in (0.,1e-12,1e-8,1e-4,1.):
  for eps in (1e-8,.001):
   epsilon.append({'gradient':g,'eps_outside_sqrt':eps,'normalized_update':0. if g==0 else g/(abs(g)+eps),'wrong_inside_sqrt':0. if g==0 else g/math.sqrt(g*g+eps)})
 skip={}
 for name,grad in [('none',None),('zero',0.)]:
  p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.1,betas=(.5,.5),eps=.01,weight_decay=.2,foreach=False,fused=False)
  p.grad=torch.tensor([1.],dtype=torch.float64);opt.step();before=p.detach().item();s0=copy.deepcopy(opt.state[p]);p.grad=None if grad is None else torch.zeros_like(p);opt.step();s=opt.state[p]
  skip[name]={'before':before,'after':p.detach().item(),'step_before':int(s0['step'].item()),'step_after':int(s['step'].item()),'m_before':s0['exp_avg'].item(),'m_after':s['exp_avg'].item(),'v_before':s0['exp_avg_sq'].item(),'v_after':s['exp_avg_sq'].item()}
 decay=[]
 for name,rates in [('constant',[.1]*20),('lower_lr',[.01]*20),('half_then_half',[.1]*10+[.01]*10)]:
  p=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64));opt=torch.optim.AdamW([p],lr=rates[0],weight_decay=.2,foreach=False,fused=False);path=[2.];product=2.
  for rate in rates:
   opt.param_groups[0]['lr']=rate;p.grad=torch.zeros_like(p);opt.step();product*=1-rate*.2;path.append(p.item())
  require(abs(p.item()-product)<1e-14,'decay product mismatch');decay.append({'case':name,'rates':rates,'path':path,'product':product})
 # A delayed nonzero gradient can give an update magnitude > lr.
 r=adam_step([0.],[1.],[0.],[0.],1000,.001,(.9,.999),1e-8)
 return {'prescribed_gradients':prescribed,'decay_comparison':paths,'preconditioned':preconditioned,'epsilon':epsilon,'none_vs_zero':skip,'pure_decay':decay,'delayed_gradient':{'step':1000,'normalized_magnitude':float(-r['adaptive_delta'][0]/.001),'note':'999 supplied zero gradients,then1; moments remain zero until step1000'}}

def init_parameters(theta):
 theta=array(theta,'theta',(9,))
 return [torch.nn.Parameter(torch.tensor(theta[a:b],dtype=torch.float64)) for a,b in [(0,4),(4,6),(6,8),(8,9)]]

def flat_parameters(parts):return torch.cat(parts)

def make_optimizer(parts,family,lr,wd,cfg):
 require(family in FAMILIES,'unknown optimizer family');lr=scalar(lr,'lr',1e-10,1);wd=scalar(wd,'weight_decay',0,1)
 require(type(cfg) is dict,'optimizer config must be dictionary')
 mask=cfg.get('decay_mask');require(type(mask) is list and all(type(z) is int for z in mask) and mask==[1,1,1,1,0,0,1,1,0],'this teaching model requires declared weight-only decay mask; create new parameter groups for a different policy')
 scalar(cfg.get('momentum'),'momentum',0,.99)
 betas=cfg.get('betas');require(isinstance(betas,(list,tuple)) and len(betas)==2,'two betas required')
 for value in betas:scalar(value,'beta',0,.99999)
 scalar(cfg.get('eps'),'eps',1e-15,1)
 groups=[{'params':[parts[0],parts[2]],'weight_decay':wd},{'params':[parts[1],parts[3]],'weight_decay':0.}]
 if family=='sgd_momentum':return torch.optim.SGD(groups,lr=lr,momentum=cfg['momentum'],foreach=False)
 cls=torch.optim.AdamW if family=='adamw' else torch.optim.Adam
 return cls(groups,lr=lr,betas=tuple(cfg['betas']),eps=cfg['eps'],foreach=False,fused=False)

def generate_order(seed,epochs,n):
 integer(seed,'seed',0,2**31-1);integer(epochs,'epochs',1,50);integer(n,'n',1,256)
 rng=np.random.default_rng(seed);return np.concatenate([rng.permutation(n) for _ in range(epochs)]).tolist()

def fit(train,validation,cfg,family,lr,wd,seed,record=False):
 # Neither test inputs nor test labels are parameters to this function.
 t,x,y=validate(cfg['initial_theta'],train['X'],train['y']);_,vx,vy=validate(t,validation['X'],validation['y'])
 bs=integer(cfg['batch_size'],'batch_size',1,len(x));steps=integer(cfg['steps'],'steps',1,10000);integer(cfg['epochs'],'epochs',1,50)
 require(len(x)%bs==0 and steps*bs==len(x)*cfg['epochs'],'budget mismatch');require(type(record) is bool,'record must be boolean')
 parts=init_parameters(t);opt=make_optimizer(parts,family,lr,wd,cfg);order=generate_order(seed,cfg['epochs'],len(x))
 tx=torch.tensor(x);ty=torch.tensor(y);history=[]
 def snapshot(k):
  q=flat_parameters(parts).detach().numpy();return {'step':k,'examples':k*bs,'theta':q.tolist(),'train_half_mse':manual(q,x,y)['loss'],'validation_half_mse':manual(q,vx,vy)['loss']}
 if record:history.append(snapshot(0))
 for k in range(steps):
  ids=order[k*bs:(k+1)*bs];opt.zero_grad(set_to_none=True);loss=tensor_loss(flat_parameters(parts),tx[ids],ty[ids]);require(torch.isfinite(loss).item(),'nonfinite batch loss');loss.backward()
  require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parts),'nonfinite or absent training gradient');opt.step()
  q=flat_parameters(parts).detach();require(torch.isfinite(q).all().item() and q.abs().max().item()<=100,'training parameters left teaching domain')
  if record:history.append(snapshot(k+1))
 final=snapshot(steps)
 return {'family':family,'lr':lr,'weight_decay':wd,'seed':seed,'final':final,'history':history,'order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),'optimizer_steps':steps,'training_examples':steps*bs}

def search(train,validation,cfg):
 rows=[];cache={};means=[]
 for family in FAMILIES:
  index=0
  for lr in cfg['learning_rates'][family]:
   for wd in cfg['decays']:
    losses=[]
    for seed in cfg['seeds']:
     result=fit(train,validation,cfg,family,lr,wd,seed);cache[(family,index,seed)]=result;losses.append(result['final']['validation_half_mse'])
     rows.append({'family':family,'grid_index':index,'lr':lr,'weight_decay':wd,'seed':seed,'train_half_mse':result['final']['train_half_mse'],'validation_half_mse':losses[-1],'optimizer_steps':result['optimizer_steps'],'training_examples':result['training_examples'],'order_sha256':result['order_sha256']})
    means.append({'family':family,'grid_index':index,'lr':lr,'weight_decay':wd,'mean_validation_half_mse':float(np.mean(losses)),'min_seed_validation':min(losses),'max_seed_validation':max(losses)});index+=1
 selected=[min([r for r in means if r['family']==family],key=lambda r:(r['mean_validation_half_mse'],r['grid_index'])) for family in FAMILIES]
 return {'all_candidates':means,'selected':selected,'criterion':cfg['selection'],'test_seen':False},rows,cache

def evaluate_frozen(selection,cache,test,cfg):
 # Called only after all three family choices are already fixed.
 rows=[]
 for choice in selection['selected']:
  for seed in cfg['seeds']:
   run=cache[(choice['family'],choice['grid_index'],seed)];loss=manual(run['final']['theta'],test['X'],test['y'])['loss']
   rows.append({'family':choice['family'],'grid_index':choice['grid_index'],'lr':choice['lr'],'weight_decay':choice['weight_decay'],'seed':seed,'test_half_mse':loss})
 return {'selection_sha256':hashlib.sha256(canonical(selection)).hexdigest(),'rows':rows,'family_means':{f:float(np.mean([r['test_half_mse'] for r in rows if r['family']==f])) for f in FAMILIES},'interpretation':'descriptive on one held-out synthetic split; training seeds are not independent sampled datasets'}

def state_demo(hand):
 x=torch.tensor(hand['X'],dtype=torch.float64);y=torch.tensor(hand['y'],dtype=torch.float64);initial=np.array(hand['theta'])
 def start(theta):
  p=torch.nn.Parameter(torch.tensor(theta,dtype=torch.float64));o=torch.optim.AdamW([p],lr=.02,betas=(.8,.9),eps=.001,weight_decay=.1,foreach=False,fused=False);return p,o
 def update(p,o,n):
  for _ in range(n):o.zero_grad(set_to_none=True);tensor_loss(p,x,y).backward();o.step()
 full,of=start(initial);update(full,of,2);saved_theta=full.detach().clone();saved_opt=copy.deepcopy(of.state_dict());update(full,of,3)
 resumed,orr=start(saved_theta.numpy());orr.load_state_dict(saved_opt);update(resumed,orr,3)
 reset,ore=start(saved_theta.numpy());update(reset,ore,3)
 require(torch.equal(full,resumed),'full Adam state resume mismatch')
 return {'cut_step':2,'total_steps':5,'full_theta':full.detach().tolist(),'resumed_theta':resumed.detach().tolist(),'weights_only_theta':reset.detach().tolist(),'exact_parameter_equality':True,'weights_only_max_gap':float((full-reset).abs().max().detach()),'saved_state_keys':sorted(next(iter(saved_opt['state'].values())).keys()),'scope':'in-memory deep-copy restore, same fixed batch; not disk or cross-version resume'}

def clean(obj):
 if isinstance(obj,np.ndarray):return obj.tolist()
 if isinstance(obj,np.generic):return obj.item()
 if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
 return obj

def canonical(obj):return (json.dumps(clean(obj),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def run(root=ROOT):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cfg,hand,data=load_inputs(root)
 trace=hand_trace(hand);mech=mechanisms();selection,rows,cache=search(data['train'],data['validation'],cfg)
 test=evaluate_frozen(selection,cache,data['test'],cfg);histories=[]
 for choice in selection['selected']:
  for seed in cfg['seeds']:
   r=fit(data['train'],data['validation'],cfg,choice['family'],choice['lr'],choice['weight_decay'],seed,record=True)
   require(r['final']==cache[(choice['family'],choice['grid_index'],seed)]['final'],'recording altered selected run');histories.append(r)
 state=state_demo(hand)
 summary={'unit':'034','runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':1},'hand_initial_loss':trace['steps'][0]['forward_backward']['loss'],'hand_after_two_steps_loss':trace['steps'][-1]['next_forward']['loss'],'candidate_configurations_per_family':9,'training_seeds':cfg['seeds'],'search_runs':len(rows),'steps_per_run':96,'examples_per_run':768,'search_steps_per_family':2592,'search_examples_per_family':20736,'additional_diagnostic_replays':9,'test_evaluations_after_selection':9,'selected':selection['selected'],'test_means':test['family_means'],'no_time_or_compute_equivalence_claim':True}
 outputs={'hand-trace.json':trace,'mechanisms.json':mech,'selection.json':selection,'selected-trajectories.json':histories,'test-report.json':test,'state-check.json':state,'summary.json':summary}
 buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 # Every calculation and serialization precedes all writes.
 payloads={name:canonical(value) for name,value in outputs.items()};payloads['search-grid.csv']=buf.getvalue().encode()
 return payloads

def write_payloads(payloads,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 # Atomic per final file; not a multi-file transaction if OS fails mid-publication.
 for name,content in payloads.items():
  fd,tmp=tempfile.mkstemp(prefix='.'+name+'.',dir=out)
  try:
   with os.fdopen(fd,'wb') as f:f.write(content)
   os.replace(tmp,out/name)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs');args=parser.parse_args();payloads=run();write_payloads(payloads,args.output);print(json.dumps({'status':'passed','outputs':sorted(payloads),'bytes':sum(map(len,payloads.values()))}))
if __name__=='__main__':main()
