"""041: paired synthetic transfer experiments. No downloads; CPU float64."""
from pathlib import Path
import argparse,csv,gzip,hashlib,io,json,math,os,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
FIXTURES={'source.csv':'8e378d24b4ae4fd84a02b7c4158d2e80ab75ce7b94e603e2fef679af285fe957','target_test.csv':'3fcd9ccd5089ad80135a611f270e2cee2c45706e45e089bc4899d16a230f2b54','target_train.csv':'9655201602ecbdb461ef62e6db8af22aa194cb76720595a38c20ad4ba4f1cfd6','target_validation.csv':'33c22da3ad65f6a3266dade462db9c9e8334826aa3975511170cdd5346bb0681'}
SEEDS=(11,23,37); LABELS=(16,64); TASKS=('related','orthogonal')
METHODS=('scratch','random_probe','pretrained_probe','finetune_small','finetune_equal','lpft','rbf_ridge')
RATES=(.03,.1,.3); PENALTIES=(.0001,.01,.1); SOURCE_STEPS=600; TARGET_STEPS=160
PARAMETERS=('a1','a2','c','v','b')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def require(condition,message):
 if not condition:raise ValueError(message)
def integer(v,name,lo=1,hi=2000):
 require(type(v) is int and lo<=v<=hi,f'{name} must be integer in [{lo},{hi}]');return v
def array(v,shape=None):
 def nobool(w):
  if isinstance(w,(list,tuple)):return all(nobool(z) for z in w)
  return not isinstance(w,(bool,np.bool_,str,bytes))
 require(nobool(v),'boolean/string is not numeric input');a=np.asarray(v)
 require(a.dtype.kind in 'ifu','real numeric input required');a=a.astype(np.float64)
 require(np.isfinite(a).all() and (np.abs(a)<=1e6).all(),'input must be finite with absolute value <= 1e6')
 if shape is not None:require(a.shape==shape,f'expected shape {shape}, got {a.shape}')
 return a

def forward(theta,x):
 t=array(theta,(5,));x=array(x);require(x.ndim==2 and x.shape[1]==2 and len(x)>0,'x must be nonempty N by 2')
 return t[3]*np.tanh(x@t[:2]+t[2])+t[4]
def gradient(theta,x,y):
 t=array(theta,(5,));x=array(x);y=array(y,(len(x),));pred=forward(t,x);h=np.tanh(x@t[:2]+t[2]);q=(pred-y)/len(x);dz=q*t[3]*(1-h*h)
 return np.array([(dz*x[:,0]).sum(),(dz*x[:,1]).sum(),dz.sum(),(q*h).sum(),q.sum()])
def hand_trace(theta=None):
 t=array([math.atanh(.5),-math.atanh(.5),0,2,.1] if theta is None else theta,(5,));x=np.eye(2);y=np.array([1.,-.4]);z=x@t[:2]+t[2];h=np.tanh(z);p=t[3]*h+t[4];err=p-y;q=err/2;dz=q*t[3]*(1-h*h)
 contributions=np.column_stack((dz*x[:,0],dz*x[:,1],dz,q*h,q));g=contributions.sum(0)
 return {'theta':t.tolist(),'x':x.tolist(),'y':y.tolist(),'z':z.tolist(),'h':h.tolist(),'prediction':p.tolist(),'error':err.tolist(),'loss_each':(.5*err**2).tolist(),'risk':float(np.mean(.5*err**2)),
 'local':{'dL_dli':[.5,.5],'dli_dprediction':err.tolist(),'dp_dv':h.tolist(),'dp_db':[1,1],'dp_dh':[float(t[3])]*2,'dh_dz':(1-h*h).tolist(),'dz_da1':x[:,0].tolist(),'dz_da2':x[:,1].tolist(),'dz_dc':[1,1],'dz_dx1':[float(t[0])]*2,'dz_dx2':[float(t[1])]*2},
 'q':q.tolist(),'dL_dz':dz.tolist(),'parameter_contributions':contributions.tolist(),'gradient':g.tolist(),'input_gradient':(dz[:,None]*t[None,:2]).tolist(),
 'theta_next_grouped':(t-np.array([.1,.1,.1,.2,.2])*g).tolist(),'theta_next_frozen':(t-np.array([0,0,0,.2,.2])*g).tolist()}

def freeze_demo():
 result={}
 for mode in ('frozen','detach','no_grad','eval_only'):
  x=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64,requires_grad=True)
  a=torch.nn.Parameter(torch.tensor([.55,-.55,0.],dtype=torch.float64));v=torch.nn.Parameter(torch.tensor([2.,.1],dtype=torch.float64))
  module=torch.nn.Module();module.register_parameter('backbone',a);module.register_parameter('head',v)
  if mode=='eval_only':module.eval()
  if mode!='eval_only':a.requires_grad_(False)
  if mode=='no_grad':
   with torch.no_grad():h=torch.tanh(x@a[:2]+a[2])
  else:h=torch.tanh(x@a[:2]+a[2])
  if mode=='detach':h=h.detach()
  loss=.5*((h*v[0]+v[1]-torch.tensor([1.,-.4],dtype=torch.float64))**2).mean();loss.backward()
  result[mode]={'x_grad':None if x.grad is None else x.grad.tolist(),'backbone_grad':None if a.grad is None else a.grad.tolist(),'head_grad':v.grad.tolist()}
 return result

def check_fixtures(data):
 data=Path(data);raw={}
 for name,sha in FIXTURES.items():
  p=data/name;require(p.is_file(),f'missing fixture {name}');raw[name]=p.read_bytes();require(digest(raw[name])==sha,f'fixture hash mismatch {name}')
 return raw

def parse(raw,name):
 rows=list(csv.DictReader(io.StringIO(raw.decode())));required=['id','x1','x2','related','orthogonal'];require(list(rows[0])==required,'fixture header mismatch')
 x=array([[float(r['x1']),float(r['x2'])] for r in rows]);ys={k:array([float(r[k]) for r in rows]) for k in TASKS}
 return {'ids':[r['id'] for r in rows],'x':x,'y':ys}

def initialize(seed):
 rng=np.random.default_rng(seed);return np.array([*rng.normal(0,.3,2),0,1,0],dtype=float)

def train(initial,x,y,steps,backbone_lr,head_lr,validation=None):
 integer(steps,'steps');t=array(initial,(5,));x=array(x);y=array(y,(len(x),))
 require(type(backbone_lr) in (float,int) and 0<=backbone_lr<=1 and math.isfinite(backbone_lr),'bad backbone learning rate')
 require(type(head_lr) in (float,int) and 0<head_lr<=1 and math.isfinite(head_lr),'bad head learning rate')
 a=torch.nn.Parameter(torch.tensor(t[:3],dtype=torch.float64));b=torch.nn.Parameter(torch.tensor(t[3:],dtype=torch.float64));xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(y,dtype=torch.float64)
 opt=torch.optim.SGD([{'params':[a],'lr':backbone_lr},{'params':[b],'lr':head_lr}]);history=[]
 for step in range(steps+1):
  pred=torch.tanh(xt@a[:2]+a[2])*b[0]+b[1];loss=.5*((pred-yt)**2).mean();theta=np.r_[a.detach().numpy(),b.detach().numpy()]
  row={'step':step,'theta':theta.tolist(),'train_mse':float(2*loss.detach()),'validation_mse':None if validation is None else float(np.mean((forward(theta,validation[0])-validation[1])**2))}
  require(np.isfinite(theta).all() and math.isfinite(row['train_mse']),'nonfinite training result');history.append(row)
  if step<steps:opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return theta,history

def ridge_features(x,kind,theta=None):
 if kind=='probe':return np.column_stack([np.tanh(x@theta[:2]+theta[2]),np.ones(len(x))])
 grid=np.array([(a,b) for a in np.linspace(-2,2,5) for b in np.linspace(-2,2,5)])
 return np.column_stack([np.exp(-((x[:,None]-grid[None,:])**2).sum(2)/2),np.ones(len(x))])
def ridge_fit(phi,y,penalty):
 n=len(y);reg=np.eye(phi.shape[1])*penalty;reg[-1,-1]=0
 return np.linalg.solve(phi.T@phi/n+reg,phi.T@y/n)

def serialize(obj):return (json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pack(raw):
 b=io.BytesIO()
 with gzip.GzipFile(filename='',mode='wb',fileobj=b,compresslevel=9,mtime=0) as z:z.write(raw)
 return b.getvalue()
def commit(output,files):
 output=Path(output);require(not output.exists() or output.is_dir(),'output must be directory')
 require(all(type(v) is bytes for v in files.values()),'all outputs must be serialized before writing')
 output.mkdir(parents=True,exist_ok=True)
 for name,raw in files.items():
  fd,tmp=tempfile.mkstemp(prefix='.'+name,dir=output)
  try:
   with os.fdopen(fd,'wb') as f:f.write(raw)
   os.replace(tmp,output/name)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)

def run(output=None,data=None,target_steps=TARGET_STEPS):
 integer(target_steps,'target_steps');torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 raw=check_fixtures(ROOT/'data' if data is None else data)
 source=parse(raw['source.csv'],'source');tr=parse(raw['target_train.csv'],'target_train');va=parse(raw['target_validation.csv'],'target_validation')
 events=['all_fixture_bytes_verified','source_train_target_train_validation_labels_parsed'];sources=[];groups=[]
 for seed in SEEDS:
  init=initialize(seed);pre,hist=train(init,source['x'],source['y']['related'],SOURCE_STEPS,.1,.1)
  sources.append({'seed':seed,'initial':init.tolist(),'theta':pre.tolist(),'history':hist,'source_sample_presentations':SOURCE_STEPS*len(source['x'])})
  for task in TASKS:
   for n in LABELS:
    x=tr['x'][:n];y=tr['y'][task][:n];vx=va['x'];vy=va['y'][task];runs=[]
    for method in METHODS:
     candidates=[]
     for idx,value in enumerate(PENALTIES if method.endswith('probe') or method=='rbf_ridge' else RATES):
      start=init.copy() if method in ('scratch','random_probe') else np.r_[pre[:3],init[3:]]
      if method in ('random_probe','pretrained_probe','rbf_ridge'):
       kind='rbf' if method=='rbf_ridge' else 'probe';ph= ridge_features(x,kind,start);coef=ridge_fit(ph,y,value);vp=ridge_features(vx,kind,start)@coef
       theta=None if kind=='rbf' else np.r_[start[:3],coef];history=[];lr=None
      else:
       if method=='lpft':start[3:]=ridge_fit(ridge_features(x,'probe',start),y,.0001)
       lr=value*(.1 if method in ('finetune_small','lpft') else 1)
       theta,history=train(start,x,y,target_steps,lr,value,(vx,vy));vp=forward(theta,vx);coef=None
      candidates.append({'candidate':idx,'setting':value,'backbone_lr':lr,'initial':start.tolist(),'theta':None if theta is None else theta.tolist(),'coefficients':None if coef is None else coef.tolist(),'history':history,'validation_prediction':vp.tolist(),'validation_mse':float(np.mean((vp-vy)**2))})
     selected=min(range(3),key=lambda k:(candidates[k]['validation_mse'],k));runs.append({'method':method,'selected':selected,'candidates':candidates})
    groups.append({'seed':seed,'task':task,'n':n,'runs':runs})
 events+=['all_candidate_models_fitted','all_validation_selections_frozen']
 # Deliberate information boundary: test labels are parsed only after every choice.
 te=parse(raw['target_test.csv'],'target_test');events.append('test_labels_parsed_after_selection');metrics=[]
 for group in groups:
  for run_ in group['runs']:
   c=run_['candidates'][run_['selected']];predictions={}
   for split,d in [('train',tr),('validation',va),('test',te)]:
    x=d['x'][:group['n']] if split=='train' else d['x'];y=d['y'][group['task']][:len(x)]
    p=ridge_features(x,'rbf')@np.array(c['coefficients']) if run_['method']=='rbf_ridge' else forward(c['theta'],x)
    predictions[split]={'ids':d['ids'][:len(x)],'prediction':p.tolist(),'y':y.tolist(),'mse':float(np.mean((p-y)**2))}
   run_['predictions']=predictions
   metrics.append({k:group[k] for k in ('seed','task','n')}|{'method':run_['method'],'setting':c['setting'],'train_mse':predictions['train']['mse'],'validation_mse':predictions['validation']['mse'],'test_mse':predictions['test']['mse']})
 summary=[]
 for task in TASKS:
  for n in LABELS:
   scratch={r['seed']:r['test_mse'] for r in metrics if r['task']==task and r['n']==n and r['method']=='scratch'}
   for method in METHODS:
    rows=[r for r in metrics if r['task']==task and r['n']==n and r['method']==method];diff=[r['test_mse']-scratch[r['seed']] for r in rows]
    summary.append({'task':task,'n':n,'method':method,'mean_test_mse':float(np.mean([r['test_mse'] for r in rows])),'min_test_mse':min(r['test_mse'] for r in rows),'max_test_mse':max(r['test_mse'] for r in rows),'paired_delta_vs_scratch':diff,'mean_delta_vs_scratch':float(np.mean(diff))})
 protocol={'source_steps':SOURCE_STEPS,'source_n':512,'target_steps':target_steps,'target_label_counts':list(LABELS),'validation_labels_per_task':128,'seeds':list(SEEDS),'methods':list(METHODS),'rates':list(RATES),'penalties':list(PENALTIES),'target_nn_candidates':144,'target_nn_updates':144*target_steps,'target_nn_sample_presentations':4*len(SEEDS)*len(TASKS)*3*sum(LABELS)*target_steps,'ridge_solves':108,'lpft_initial_solves':36,'source_updates':3*SOURCE_STEPS,'source_sample_presentations':3*SOURCE_STEPS*512,'per_nn_method_per_task_n_fit_count':3,'claim':'Target neural arms share update/sample budgets. Ridge arms use closed-form solves; neither wall-clock nor end-to-end pretraining budget is matched.'}
 result={'schema':'unit041-v1','protocol':protocol,'events':events,'fixture_sha256':FIXTURES,'source_runs':sources,'target_groups':groups,'metrics':metrics,'summary':summary}
 calc={'first':hand_trace(),'grouped_next':hand_trace(hand_trace()['theta_next_grouped']),'frozen_next':hand_trace(hand_trace()['theta_next_frozen']),'freeze_demo':freeze_demo()}
 result_raw=serialize(result);packed=pack(result_raw);small={'schema':'unit041-summary-v1','protocol':protocol,'summary':summary,'packed_result':{'filename':'results.json.gz','raw_sha256':digest(result_raw),'raw_bytes':len(result_raw),'packed_sha256':digest(packed),'packed_bytes':len(packed),'gzip':{'level':9,'mtime':0,'filename':''}},'metric_rows':len(metrics),'target_groups':len(groups),'source_history_rows':sum(len(v['history']) for v in sources),'target_history_rows':sum(len(c['history']) for g in groups for r in g['runs'] for c in r['candidates']),'test_prediction_rows':sum(len(r['predictions']['test']['prediction']) for g in groups for r in g['runs'])}
 files={'results.json.gz':packed,'summary.json':serialize(small),'calculations.json':serialize(calc)}
 # All computations including serialization finish before the first output write.
 if output is not None:commit(output,files)
 return {'sha256':{k:digest(v) for k,v in files.items()},'bytes':{k:len(v) for k,v in files.items()},'summary':small}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--steps',type=int,default=TARGET_STEPS);a=p.parse_args();print(json.dumps(run(a.output,a.data,a.steps),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
