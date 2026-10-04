"""035: finite CPU experiments about initialization, moments and gradient transport.
Run from any working directory; no download, no GPU, no hidden notebook state.
"""
from pathlib import Path
import argparse, csv, gzip, io, json, math, platform
from fractions import Fraction as F
import numpy as np
import torch
BASE = Path(__file__).resolve().parent
ACTS = ('linear', 'tanh', 'relu', 'sigmoid')
SCALES = (0.5, 1.0, math.sqrt(2), 2.0)
SEEDS = (35, 36, 37)
DTYPE = torch.float64

def integer(value, name, low, high):
    if isinstance(value, (bool,np.bool_)) or not isinstance(value, (int,np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return int(value)

def positive(value, name, high=10):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value) or not 0 < value <= high:
        raise ValueError(f'{name} must be finite and in (0, {high}]')
    return float(value)

def validate_batch(x,y):
    # Reject bool before coercion; check values, not merely dtype after conversion.
    for value in (x,y):
        a=np.asarray(value,dtype=object)
        if any(isinstance(v,(bool,np.bool_)) for v in a.flat): raise ValueError('boolean data are not numeric measurements')
        if any(not isinstance(v,(int,float,np.integer,np.floating)) for v in a.flat): raise ValueError('real numeric data required')
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<1 or y.shape!=(x.shape[0],1): raise ValueError('expected X[n,d] and y[n,1], n>=2')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('data must be finite')
    if max(np.abs(x).max(),np.abs(y).max())>100: raise ValueError('teaching data magnitude exceeds 100')
    return x,y

def load_data(path=None):
    path=Path(path) if path else BASE/'data/synthetic_regression.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f); rows=list(reader)
        if reader.fieldnames != ['id']+[f'x{i}' for i in range(16)]+['y']: raise ValueError('unexpected CSV schema')
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('duplicate sample ID')
    return validate_batch([[float(r[f'x{i}']) for i in range(16)] for r in rows],[[float(r['y'])] for r in rows])

def activation(z, name):
    if name not in ACTS: raise ValueError('unknown activation')
    return {'linear':lambda t:t, 'tanh':torch.tanh,'relu':torch.relu,'sigmoid':torch.sigmoid}[name](z)

class Network:
    """PyTorch layout: each W[out,in]; use h @ W.T + b."""
    def __init__(self, in_dim=16, width=64, depth=12, act='relu', alpha=math.sqrt(2), seed=35):
        in_dim=integer(in_dim,'in_dim',1,128); width=integer(width,'width',1,256); depth=integer(depth,'depth',1,30)
        alpha=positive(alpha,'alpha',4); seed=integer(seed,'seed',0,2**31-1)
        if act not in ACTS: raise ValueError('unknown activation')
        self.act=act; self.params=[]; gen=torch.Generator().manual_seed(seed)
        for fan_in in [in_dim]+[width]*(depth-1):
            w=(torch.randn(width,fan_in,generator=gen,dtype=DTYPE)*alpha/math.sqrt(fan_in)).requires_grad_()
            b=torch.zeros(width,dtype=DTYPE,requires_grad=True); self.params.extend((w,b))
        self.params.extend(((torch.randn(1,width,generator=gen,dtype=DTYPE)/math.sqrt(width)).requires_grad_(),torch.zeros(1,dtype=DTYPE,requires_grad=True)))
    def forward(self,x):
        zs=[]; hs=[]; h=x
        for w,b in zip(self.params[:-2:2],self.params[1:-2:2]):
            z=h@w.T+b; h=activation(z,self.act); zs.append(z);hs.append(h)
        return h@self.params[-2].T+self.params[-1],zs,hs

def moments(a):
    a=np.asarray(a,dtype=np.float64).ravel()
    if not a.size or not np.isfinite(a).all(): raise ValueError('moment input must be nonempty and finite')
    mean=float(a.mean()); return {'mean':mean,'variance':float(np.mean((a-mean)**2)),'second':float(np.mean(a*a)),
        'q01':float(np.quantile(a,.01)),'q50':float(np.quantile(a,.5)),'q99':float(np.quantile(a,.99)),
        'zero_fraction':float(np.mean(a==0))}

def snapshot(net,x,y,probe):
    pred,zs,hs=net.forward(x); loss=.5*(pred-y).square().mean()
    task_grads=torch.autograd.grad(loss,zs,retain_graph=True)
    # Q is a synthetic scalar, not the regression loss. Terminal gradient is probe.
    q=(hs[-1]*probe).sum()
    probe_grads=torch.autograd.grad(q,zs,retain_graph=True)
    rows=[]; arrays={}
    for i,(z,h,gt,gp) in enumerate(zip(zs,hs,task_grads,probe_grads),1):
        z,h,gt,gp=[v.detach().numpy() for v in (z,h,gt,gp)]
        d={'linear':lambda:np.ones_like(z),'relu':lambda:(z>0).astype(float),'tanh':lambda:1-h*h,'sigmoid':lambda:h*(1-h)}[net.act]()
        row={'layer':i, 'derivative_second':float(np.mean(d*d)), 'small_derivative_fraction':float(np.mean(np.abs(d)<.01)),
            'inactive_fraction':float(np.mean(z<=0)) if net.act=='relu' else 0.,
            'mean_unit_data_variance':float(np.var(h,axis=0).mean()), 'variance_unit_data_means':float(np.var(h.mean(axis=0)))}
        for name,a in [('z',z),('h',h),('task_grad',gt),('probe_grad',gp)]:
            row.update({name+'_'+k:v for k,v in moments(a).items()})
            if i in (1,6,12): arrays[f'{name}_{i}']=a.ravel().tolist()
        rows.append(row)
    return float(loss.detach()),rows,arrays

def run_case(x,y,act,alpha,seed,steps=100,width=64,depth=12,lr=.03):
    net=Network(x.shape[1],width,depth,act,alpha,seed)
    gen=torch.Generator().manual_seed(9000+seed)
    probe=torch.randn(x.shape[0],width,generator=gen,dtype=DTYPE)/math.sqrt(x.shape[0]*width)
    loss,initial,arrays=snapshot(net,x,y,probe)
    curve=[]; status='completed'; completed=0
    for step in range(steps+1):
        pred,_,_=net.forward(x); loss=.5*(pred-y).square().mean(); value=float(loss.detach())
        if not math.isfinite(value) or value>1e10:
            status='stopped_loss_limit'; break
        curve.append({'step':step,'loss':value})
        if step==steps: break
        grads=torch.autograd.grad(loss,net.params)
        candidates=[p.detach()-lr*g for p,g in zip(net.params,grads)]
        if not all(torch.isfinite(c).all() for c in candidates): status='stopped_nonfinite_update';break
        with torch.no_grad():
            for p,c in zip(net.params,candidates):p.copy_(c)
        completed=step+1
    # Last usable loss is the curve's final entry; do not silently treat a stop as convergence.
    summary={'activation':act,'alpha':alpha,'seed':seed,'status':status,'updates_applied':completed,
             'initial_loss':curve[0]['loss'] if curve else loss.item(),'last_recorded_loss':curve[-1]['loss'] if curve else None}
    return summary,initial,curve,arrays

def hand_reference():
    """Independent exact-rational scalar loops; every parameter receives a derivative."""
    x=[F(1),F(-1)]; y=[F(1,2),F(-1,2)]
    w1=[[F(1)],[F(-1)]]; b1=[F(1,5),F(1,10)]
    w2=[[F(1,2),F(-1,4)],[F(1,4),F(1,2)]];b2=[F(1,10),F(1,5)]
    u=[F(1),F(-1,2)];c=F(1,10)
    def forward(w1,b1,w2,b2,u,c):
        out=[]
        for xi,yi in zip(x,y):
            z1=[w1[j][0]*xi+b1[j] for j in range(2)];h1=[max(F(0),v) for v in z1]
            z2=[sum(w2[j][k]*h1[k] for k in range(2))+b2[j] for j in range(2)];h2=[max(F(0),v) for v in z2]
            pred=sum(u[j]*h2[j] for j in range(2))+c
            out.append({'x':xi,'y':yi,'z1':z1,'h1':h1,'z2':z2,'h2':h2,'pred':pred,'residual':pred-yi})
        return out
    old=forward(w1,b1,w2,b2,u,c)
    gw1=[[F(0)],[F(0)]];gb1=[F(0),F(0)];gw2=[[F(0),F(0)],[F(0),F(0)]];gb2=[F(0),F(0)];gu=[F(0),F(0)];gc=F(0)
    for r in old:
        e=r['residual']/2;dh2=[e*v for v in u];dz2=[dh2[j]*int(r['z2'][j]>0) for j in range(2)]
        dh1=[sum(dz2[j]*w2[j][k] for j in range(2)) for k in range(2)];dz1=[dh1[k]*int(r['z1'][k]>0) for k in range(2)]
        r.update(e=e,dh2=dh2,dz2=dz2,dh1=dh1,dz1=dz1)
        for j in range(2):
            gu[j]+=e*r['h2'][j];gb2[j]+=dz2[j];gb1[j]+=dz1[j];gw1[j][0]+=dz1[j]*r['x']
            for k in range(2):gw2[j][k]+=dz2[j]*r['h1'][k]
        gc+=e
    params=[*sum(w1,[]),*b1,*sum(w2,[]),*b2,*u,c]
    grads=[*sum(gw1,[]),*gb1,*sum(gw2,[]),*gb2,*gu,gc]
    new=[p-F(1,10)*g for p,g in zip(params,grads)]
    after=forward([[new[0]],[new[1]]],new[2:4],[new[4:6],new[6:8]],new[8:10],new[10:12],new[12])
    return {'before':old,'after':after,'theta':params,'gradient':grads,'updated':new,
       'loss':sum(r['residual']**2 for r in old)/4,'new_loss':sum(r['residual']**2 for r in after)/4}

def hand_torch(theta=None):
    if theta is None:theta=[float(v) for v in hand_reference()['theta']]
    theta=torch.tensor(theta,dtype=DTYPE,requires_grad=True)
    x=torch.tensor([[1.],[-1.]],dtype=DTYPE);y=torch.tensor([[.5],[-.5]],dtype=DTYPE)
    z1=x@theta[:2].reshape(2,1).T+theta[2:4];h1=torch.relu(z1)
    z2=h1@theta[4:8].reshape(2,2).T+theta[8:10];h2=torch.relu(z2)
    pred=h2@theta[10:12,None]+theta[12];loss=.5*(pred-y).square().mean()
    grad=torch.autograd.grad(loss,theta)[0]
    return float(loss.detach()),grad.detach().numpy()

def extra_mechanisms():
    # Deterministic antithetic sample gives exact symmetry, not independent doubled draws.
    rng=np.random.default_rng(351);pos=rng.standard_normal(100000);z=np.r_[pos,-pos];h=np.maximum(z,0)
    gaussian={'z':moments(z),'relu':moments(h),'gaussian_prediction_mean':math.sqrt(np.mean(z*z)/(2*math.pi)),
              'gaussian_prediction_variance':np.mean(z*z)*(.5-1/(2*math.pi))}
    # A fixed matrix and correlated coordinates break the independent-input simplification.
    t=np.array([-1.,1.]);x=np.c_[t,t];w=np.array([1.,1.])/math.sqrt(2)
    covariance={'actual_variance':float(np.var(x@w)), 'diagonal_only':float(np.sum(w*w*np.var(x,axis=0))),
                'full_covariance':float(w@np.cov(x,rowvar=False,bias=True)@w)}
    zz=torch.tensor([-1.,0.,1.],dtype=DTYPE,requires_grad=True);rr=torch.relu(zz);gg=torch.autograd.grad(rr.sum(),zz)[0]
    # Exact finite gate dependence: G = 1[Z>0], T = G. E[G^2 T^2]=1/2, factorized=1/4.
    gate={'joint_second':.5,'product_of_seconds':.25,'relu_at_zero':float(gg[1]),'central_difference_at_zero':.5}
    # Equal Frobenius norm, very different singular values.
    jac={'balanced':[1.,1.],'rank_deficient':[math.sqrt(2),0.], 'mean_square_singular_values':1.}
    return {'relu_moments':gaussian,'correlated_input':covariance,'gate_dependence':gate,'jacobian_counterexample':jac}

def width_study(x):
    rows=[]
    for width in (8,32,128):
        for seed in range(40,56):
            net=Network(16,width,12,'relu',math.sqrt(2),seed)
            with torch.no_grad(): _,zs,hs=net.forward(x)
            q0=float(x.square().mean());q=float(hs[-1].square().mean())
            rows.append({'width':width,'seed':seed,'last_second_over_input':q/q0})
    return rows

def serial(obj):
    if isinstance(obj,F):return {'fraction':str(obj),'float':float(obj)}
    if isinstance(obj,dict):return {k:serial(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [serial(v) for v in obj]
    return obj

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def distribution_gzip_bytes(selected):
    """Preserve the full original UTF-8 JSON, including spaces and final newline.

    An empty stored filename and mtime=0 remove path/time dependence. Compressed
    byte equality is verified for the recorded Python/zlib runtime; gzip itself
    is a lossless portable container, not a cross-version compression promise.
    """
    raw=(json.dumps(selected,allow_nan=False)+'\n').encode('utf-8')
    buffer=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',compresslevel=9,mtime=0,fileobj=buffer) as stream:
        stream.write(raw)
    return buffer.getvalue()

def run(output=None, data=None, steps=100):
    # Validate ALL external configuration before creating or overwriting output.
    steps=integer(steps,'steps',1,500)
    x,y=load_data(data);output=Path(output) if output else BASE/'outputs'
    if output.exists() and not output.is_dir():raise ValueError('output must be a directory')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tx=torch.from_numpy(x);ty=torch.from_numpy(y)
    summaries=[];stats=[];curves=[];selected={}
    for act in ACTS:
        for alpha in SCALES:
            for seed in SEEDS:
                summary,rows,curve,arrays=run_case(tx,ty,act,alpha,seed,steps)
                key={'activation':act,'alpha':alpha,'seed':seed}
                summaries.append(summary);stats.extend([dict(key,**r) for r in rows]);curves.extend([dict(key,**r) for r in curve])
                if seed==35 and alpha in (1.,math.sqrt(2),2.):selected[f'{act}_{alpha:.6f}']=arrays
    result={'protocol':{'dtype':'float64','device':'cpu','n':len(x),'input_features':x.shape[1],'depth':12,'width':64,'steps':steps,'lr':.03,'scales':SCALES,'seeds':SEEDS,'probe':'independent terminal normal / sqrt(n*width)','loss':'half-MSE mean over samples','safety_loss_limit':1e10},
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
      'hand':serial(hand_reference()),'mechanisms':extra_mechanisms(),'cases':summaries}
    # Compute first; write only after validation and all experiments succeed.
    widths=width_study(tx)
    distribution_bytes=distribution_gzip_bytes(selected)
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'selected_distributions.json.gz').write_bytes(distribution_bytes)
    write_csv(output/'layer_statistics.csv',stats);write_csv(output/'training_curves.csv',curves);write_csv(output/'width_study.csv',widths)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--data',type=Path);parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args(); result=run(args.output,args.data,args.steps)
    print(json.dumps({'cases':len(result['cases']),'completed':sum(c['status']=='completed' for c in result['cases']),
      'hand_loss':result['hand']['loss'],'hand_new_loss':result['hand']['new_loss'],'protocol':result['protocol']},ensure_ascii=False,sort_keys=True))
