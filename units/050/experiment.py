"""DL050: explicit float64 CPU RNN, full BPTT / detach / global clipping.
Run from any directory. Never downloads data. All runs follow protocol.json.
"""
from pathlib import Path
import argparse,json,hashlib,platform,time
import numpy as np
import torch
import torch.nn.functional as F
from reference import validate,fixture,full_bptt,finite_difference,scalar_fixture_reference,NAMES
ROOT=Path(__file__).resolve().parent
DT=torch.float64; DEV=torch.device('cpu')
torch.set_num_threads(1)

def tensor(a,grad=False): return torch.tensor(a,dtype=DT,device=DEV,requires_grad=grad)
def params_from_numpy(p): return {k:tensor(v,True) for k,v in p.items()}
def numpy_params(p): return {k:v.detach().cpu().numpy().copy() for k,v in p.items()}

def check_inputs(p,x,lengths,truncate=None):
    if not isinstance(x,torch.Tensor) or x.dtype!=DT or x.device!=DEV:
        raise ValueError('x must be a float64 CPU tensor')
    if not isinstance(lengths,torch.Tensor) or lengths.dtype!=torch.int64 or lengths.device!=DEV: raise ValueError('lengths must be int64 CPU')
    for v in p.values():
        if not isinstance(v,torch.Tensor) or v.dtype!=DT or v.device!=DEV: raise ValueError('parameters must be float64 CPU tensors')
    validate(numpy_params(p),x.detach().numpy(),lengths.numpy())
    if truncate is not None and (isinstance(truncate,bool) or not isinstance(truncate,int) or not 1<=truncate<=x.shape[1]):
        raise ValueError('truncate must be None or integer 1..T')

def _forward(p,x,lengths,truncate=None):
    n,t,_=x.shape
    h=torch.zeros((n,p['b'].numel()),dtype=DT,device=DEV); states=[]
    for k in range(t):
        # Boundaries are BEFORE t=K,2K,... using zero-based indexing.
        if truncate is not None and k>0 and k%truncate==0: h=h.detach()
        candidate=torch.tanh(x[:,k]@p['U'].T+h@p['W'].T+p['b'])
        active=(k<lengths).unsqueeze(1)
        h=torch.where(active,candidate,h)
        states.append(h)
    states=torch.stack(states,dim=1)
    return states@p['v']+p['c'],states

def forward(p,x,lengths,truncate=None):
    check_inputs(p,x,lengths,truncate)
    return _forward(p,x,lengths,truncate)

def token_loss(logits,y,lengths):
    if not isinstance(logits,torch.Tensor) or logits.dtype!=DT or logits.device!=DEV or logits.ndim!=2 or not torch.isfinite(logits).all(): raise ValueError('logits must be finite float64 CPU [N,T]')
    n,t=logits.shape
    if not (1<=n<=4096 and 1<=t<=128): raise ValueError('logit shape outside supported domain')
    if not isinstance(lengths,torch.Tensor) or lengths.dtype!=torch.int64 or lengths.device!=DEV or lengths.shape!=(n,) or not torch.all((lengths>=1)&(lengths<=t)): raise ValueError('invalid lengths for token loss')
    if not isinstance(y,torch.Tensor): raise ValueError('y must be a Tensor')
    if logits.shape!=y.shape or y.dtype!=DT or y.device!=DEV or not torch.isfinite(y).all() or not torch.all((y==0)|(y==1)): raise ValueError('binary float64 targets must match logits')
    mask=torch.arange(logits.shape[1],dtype=torch.int64,device=DEV)[None,:]<lengths[:,None]
    return (F.binary_cross_entropy_with_logits(logits,y,reduction='none')*mask).sum()/mask.sum()

def init_params(seed,input_size=3,hidden_size=12,gain=1.1):
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32: raise ValueError('seed outside uint32 range')
    if not (isinstance(input_size,int) and isinstance(hidden_size,int) and 1<=input_size<=64 and 1<=hidden_size<=64): raise ValueError('dimensions must be 1..64')
    if not np.isfinite(gain) or not 0<gain<=2: raise ValueError('gain must be in (0,2]')
    rng=np.random.default_rng(seed);q,r=np.linalg.qr(rng.normal(size=(hidden_size,hidden_size)))
    q=q*np.where(np.diag(r)>=0,1.,-1.)[None,:]
    p={'U':rng.normal(0,1/np.sqrt(input_size),size=(hidden_size,input_size)), 'W':gain*q,'b':np.zeros(hidden_size),'v':rng.normal(0,1/np.sqrt(hidden_size),size=hidden_size),'c':np.array(0.)}
    return params_from_numpy({k:np.asarray(v,dtype=np.float64) for k,v in p.items()})

def step(p,x,y,lengths,lr,truncate=None,clip=None):
    check_inputs(p,x,lengths,truncate)
    if not np.isfinite(lr) or not 0<lr<=3: raise ValueError('lr must be in (0,3]')
    if clip is not None and (not np.isfinite(clip) or not 0<clip<=100): raise ValueError('clip must be in (0,100]')
    if y.dtype!=DT or y.device!=DEV or y.shape!=(x.shape[0],) or not torch.all((y==0)|(y==1)): raise ValueError('last-step target must be float64 binary [N]')
    for q in p.values():q.grad=None
    z,_=_forward(p,x,lengths,truncate)
    last=z[torch.arange(x.shape[0],dtype=torch.int64,device=DEV),lengths-1]
    loss=F.binary_cross_entropy_with_logits(last,y)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward()
    norm=torch.linalg.vector_norm(torch.cat([p[k].grad.reshape(-1) for k in NAMES]))
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
    pre_norm=float(norm);clipped=False
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(list(p.values()),clip,error_if_nonfinite=True,foreach=False)
        clipped=pre_norm>clip
    post_norm=float(torch.linalg.vector_norm(torch.cat([p[k].grad.reshape(-1) for k in NAMES])))
    # Every gradient comes from the same old parameter vector; update only now.
    with torch.no_grad():
        for q in p.values():q.sub_(lr*q.grad)
    return {'loss_before_update':float(loss.detach()),'grad_norm_before':pre_norm,'grad_norm_after':post_norm,'clipped':clipped}

def evaluate(p,x,y):
    lengths=torch.full((x.shape[0],),x.shape[1],dtype=torch.int64,device=DEV)
    with torch.no_grad():
        z,_=forward(p,x,lengths);last=z[:,-1]
        return {'bce':float(F.binary_cross_entropy_with_logits(last,y)),'accuracy':float(((last>=0)==(y>=.5)).to(DT).mean())}

def input_probe(p,x,truncate=None):
    xx=x[:1].detach().clone().requires_grad_(True)
    lengths=torch.full((1,),xx.shape[1],dtype=torch.int64,device=DEV)
    z,_=forward(p,xx,lengths,truncate);g=torch.autograd.grad(z[0,-1],xx)[0]
    return torch.linalg.vector_norm(g[0],dim=1).detach().numpy().tolist()

def hand_result():
    p,x,l,y=fixture();r=full_bptt(p,x,l,y);tp=params_from_numpy(p)
    z,h=forward(tp,tensor(x),torch.tensor(l,dtype=torch.int64,device=DEV));loss=token_loss(z,tensor(y),torch.tensor(l,dtype=torch.int64,device=DEV));loss.backward()
    ag={k:tp[k].grad.numpy().copy() for k in NAMES};fd=finite_difference(p,x,l,y)
    new={k:np.asarray(p[k]-.2*r['grads'][k],dtype=np.float64) for k in NAMES};next_r=full_bptt(new,x,l,y)
    def serial(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,dict):return {k:serial(q) for k,q in v.items()}
        return v
    return serial({'initial_parameters':p,'x':x,'lengths':l,'y':y,'reference':r,'autograd':ag,'finite_difference':fd,'scalar_loss':scalar_fixture_reference(p,x,l,y)[0],'learning_rate':.2,'updated_parameters':new,'next_forward':next_r})

def run_all(outdir=None):
    start=time.perf_counter(); out=Path(outdir) if outdir else ROOT/'outputs';out.mkdir(parents=True,exist_ok=True)
    prot=json.loads((ROOT/'protocol.json').read_text())
    if not isinstance(prot['steps'],int) or isinstance(prot['steps'],bool) or not 1<=prot['steps']<=1000: raise ValueError('steps must be 1..1000')
    if len(prot['delays'])*len(prot['model_seeds'])*len(prot['conditions'])>100: raise ValueError('at most 100 runs supported')
    data=np.load(ROOT/'data/delayed_sign.npz',allow_pickle=False)
    result={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'runs':[],'interpretation_limit':prot['interpretation_limit']}
    arrays={}
    for delay in prot['delays']:
        splits={s:(tensor(data[f'd{delay}_{s}_x']),tensor(data[f'd{delay}_{s}_y'])) for s in prot['counts']}
        train_x,train_y=splits['train'];lengths=torch.full((len(train_y),),delay+1,dtype=torch.int64,device=DEV)
        for seed in prot['model_seeds']:
            for cond in prot['conditions']:
                trunc=cond['truncate']
                # K >= T has no boundary and is exactly full BPTT.
                trunc=min(trunc,delay+1) if trunc is not None else None
                p=init_params(seed,prot['input_size'],prot['hidden_size'],prot['recurrent_orthogonal_gain'])
                rid=f'd{delay}_s{seed}_{cond["name"]}'
                rec={'id':rid,'delay':delay,'seed':seed,'condition':cond['name'],'truncate_effective':trunc,'clip':cond['clip'],'status':'complete','history':[],'initial':{s:evaluate(p,*q) for s,q in splits.items() if s!='test'},'probe_initial':input_probe(p,train_x)}
                try:
                    for k in range(prot['steps']):
                        metrics=step(p,train_x,train_y,lengths,prot['learning_rate'],trunc,cond['clip']);metrics['step']=k+1;rec['history'].append(metrics)
                    rec['final']={s:evaluate(p,*q) for s,q in splits.items()}
                    rec['probe_final_full']=input_probe(p,train_x)
                    rec['probe_final_training_graph']=input_probe(p,train_x,trunc)
                    rec['clip_fraction']=float(np.mean([v['clipped'] for v in rec['history']]))
                    for key,a in numpy_params(p).items():arrays[rid+'_'+key]=a
                except (FloatingPointError,ValueError) as e:
                    rec['status']='failed';rec['error']=str(e)
                result['runs'].append(rec)
                print(rid,rec['status'],rec.get('final',{}).get('test',{}),flush=True)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    (out/'hand_calculation.json').write_text(json.dumps(hand_result(),ensure_ascii=False,indent=2,allow_nan=False))
    np.savez_compressed(out/'trained_parameters.npz',**arrays)
    env={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','torch_num_threads':torch.get_num_threads(),'seconds':time.perf_counter()-start,'protocol_sha256':result['protocol_sha256'],'command':'python experiment.py --out '+str(out),'platform':platform.platform()}
    (out/'environment.json').write_text(json.dumps(env,indent=2));return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--hand-only',action='store_true');args=parser.parse_args()
    if args.hand_only: print(json.dumps(hand_result(),indent=2,allow_nan=False))
    else:run_all(args.out)
if __name__=='__main__':main()
