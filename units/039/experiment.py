"""039: explicit weighted risk, resampling, decision costs, and held-out calibration.
All inputs are public original synthetic data. CPU float64, no network access.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,math,os,platform
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
DATA_HASHES={'test.csv':'592b6c7814a48628fcd2afd8d68f628bd672ec9cf47d4c18c3bdf1c522b326b8','train.csv':'87f4b73ac0af0229f805671b29962fbd211d2aac0a7bac127a8c96eb14e4700b','validation.csv':'a77b069b71dc5c61805978023ea079946c553a36841d3dc7c0f765b3cbf7df71'}
SEEDS=(11,23,37); METHODS=('uniform','weighted','resampled'); EPOCHS=240; LR=.08
FP_COST=1.; FN_COST=4.; THRESHOLDS=np.linspace(0,1,21)

def integer(v,name,low=1,high=100000):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not low<=v<=high: raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return int(v)

def array(v,name,ndim=None):
    if isinstance(v,(list,tuple)) and any(isinstance(x,(bool,np.bool_)) for x in np.asarray(v,dtype=object).flat): raise ValueError(f'{name}: booleans are not numeric inputs')
    a=np.asarray(v)
    if a.dtype.kind not in 'iuf' or not np.isfinite(a).all(): raise ValueError(f'{name} must contain finite real numbers, not strings or booleans')
    with np.errstate(over='ignore',invalid='ignore'):
        a=a.astype(np.float64)
    if not np.isfinite(a).all(): raise ValueError(f'{name}: values must remain finite after float64 conversion')
    if not a.size or (ndim is not None and a.ndim!=ndim): raise ValueError(f'{name}: invalid shape or empty input')
    return a

def pair(z,y):
    z=array(z,'logits',1);y=array(y,'labels',1)
    if z.shape!=y.shape or not np.isin(y,[0.,1.]).all() or np.abs(z).max()>1e6: raise ValueError('Logits and binary labels must align; |logit| <= 1e6')
    return z,y

def sigmoid(z):
    z=array(z,'logits');return np.exp(-np.logaddexp(0.,-z))

def bce(z,y):
    z,y=pair(z,y)
    return np.logaddexp(0.,np.where(y==1.,-z,z))

def weighted_risk(z,y,weights,denominator):
    z,y=pair(z,y);w=array(weights,'weights',1)
    if w.shape!=z.shape or (w<=0).any(): raise ValueError('Strictly positive aligned weights required')
    if denominator not in ('count','weights'): raise ValueError('denominator must be count or weights')
    losses=bce(z,y)
    if denominator=='weights':
        # Divide all weights by the same positive maximum before either sum.
        # The normalized average is unchanged, without an overflowing weight sum.
        scaled=w/w.max()
        return float(np.dot(scaled,losses)/scaled.sum())
    # Keep ordinary finite arithmetic unchanged; recover an overflowing sum when
    # the count-mean itself is representable, otherwise reject explicitly.
    with np.errstate(over='ignore',invalid='ignore'):
        total=float(np.dot(w,losses))
        result=total/len(y) if math.isfinite(total) else float(np.mean((w/w.max())*losses)*w.max())
    if not math.isfinite(result): raise ValueError('count-weighted risk exceeds the finite float64 range')
    return result

def metrics(z,y,threshold=.5,bins=10):
    z,y=pair(z,y);bins=integer(bins,'bins',1,1000)
    if isinstance(threshold,(bool,np.bool_)) or not isinstance(threshold,(float,int,np.floating,np.integer)) or not np.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('threshold must be in [0,1]')
    p=sigmoid(z);pred=p>=threshold;n=len(y)
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)));tn=n-tp-fp-fn
    indexes=np.minimum((p*bins).astype(int),bins-1);rows=[];ece=0.
    for k in range(bins):
        mask=indexes==k;count=int(mask.sum())
        mp=float(p[mask].mean()) if count else None;rate=float(y[mask].mean()) if count else None
        rows.append({'bin':k,'left':k/bins,'right':(k+1)/bins,'n':count,'mean_p':mp,'positive_rate':rate})
        if count: ece+=count/n*abs(mp-rate)
    return {'n':n,'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'accuracy':(tp+tn)/n,'recall':tp/(tp+fn) if tp+fn else None,'precision':tp/(tp+fp) if tp+fp else None,'cost':(FP_COST*fp+FN_COST*fn)/n,'nll':float(bce(z,y).mean()),'brier':float(np.square(p-y).mean()),'ece':float(ece),'bins':rows}

def fit_temperature(z,y):
    """Bounded convex optimization over beta=1/T, using only supplied validation pairs."""
    z,y=pair(z,y);lo=.05;hi=20.
    def grad(beta):
        scaled=beta*z
        # Never subtract a rounded probability 1 from the positive label.
        residual=np.where(y==1.,-sigmoid(-scaled),sigmoid(scaled))
        return float(np.mean(residual*z))
    margins=(2*y-1)*z
    if np.all(z==0): beta=1.;status='flat'
    elif np.all(margins>=0) and np.any(margins>0):
        # Every nonzero term strictly decreases, even when exp(-beta*|z|)
        # underflows. The monotonic sign is known without subtracting floats.
        beta=hi;status='upper_bound'
    elif np.all(margins<=0) and np.any(margins<0):
        beta=lo;status='lower_bound'
    else:
        g_lo=grad(lo);g_hi=grad(hi)
        if g_lo==0. and g_hi==0.:
            beta=1.;status='numerically_flat'
        elif g_lo>=0: beta=lo;status='lower_bound'
        elif g_hi<=0: beta=hi;status='upper_bound'
        else:
            for _ in range(90):
                mid=(lo+hi)/2
                if grad(mid)>0: hi=mid
                else: lo=mid
            beta=(lo+hi)/2;status='interior'
    return {'beta':beta,'temperature':1/beta,'status':status,'gradient':grad(beta),'nll_before':float(bce(z,y).mean()),'nll_after':float(np.logaddexp(0.,np.where(y==1.,-beta*z,beta*z)).mean()),'n':len(y)}

def choose_threshold(z,y):
    z,y=pair(z,y);rows=[{'threshold':float(t),'cost':metrics(z,y,float(t))['cost']} for t in THRESHOLDS]
    best=min(r['cost'] for r in rows)
    chosen=max(r['threshold'] for r in rows if r['cost']==best)
    return chosen,rows

def load_split(directory,name):
    path=Path(directory)/f'{name}.csv'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DATA_HASHES[path.name]: raise ValueError(f'{path.name}: data integrity mismatch')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    if not rows or list(rows[0])!=['id','x1','x2','group','y','role']: raise ValueError('CSV schema mismatch')
    x=np.array([[float(r['x1']),float(r['x2'])] for r in rows]);y=np.array([int(r['y']) for r in rows],dtype=float);g=np.array([int(r['group']) for r in rows]);roles=np.array([r['role'] for r in rows])
    array(x,'features',2);pair(np.zeros(len(y)),y)
    if not np.isin(g,[0,1]).all() or len({r['id'] for r in rows})!=len(rows): raise ValueError('Invalid group or duplicate ID')
    return {'x':x,'y':y,'group':g,'role':roles,'id':[r['id'] for r in rows]}

def initial(seed):
    rng=np.random.default_rng(seed)
    return [rng.normal(0,.45,(2,6)),np.zeros(6),rng.normal(0,.35,6),np.array(0.)]

def predict(params,x):
    x=array(x,'features',2)
    if x.shape[1]!=2: raise ValueError('Expected two features')
    w,b,u,c=[array(a,'parameters') for a in params]
    if w.shape!=(2,6) or b.shape!=(6,) or u.shape!=(6,) or c.shape!=(): raise ValueError('Parameter shape mismatch')
    return np.tanh(x@w+b)@u+c

def train(x,y,method,seed,epochs=EPOCHS):
    epochs=integer(epochs,'epochs',1,5000);integer(seed,'seed',0,2**32-1)
    x=array(x,'features',2);_,y=pair(np.zeros(len(x)),y)
    if x.shape!=(len(y),2) or method not in METHODS or not 0<y.mean()<1: raise ValueError('Invalid training data or method')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    pars=[torch.tensor(a,dtype=torch.float64,requires_grad=True) for a in initial(seed)]
    xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(y,dtype=torch.float64)
    pi=y.mean();a=np.where(y==1,.5/pi,.5/(1-pi));at=torch.tensor(a,dtype=torch.float64)
    rng=np.random.default_rng(seed+39000);hist=[];seen=np.zeros(len(y),int)
    for epoch in range(1,epochs+1):
        idx=rng.choice(len(y),len(y),replace=True,p=a/a.sum()) if method=='resampled' else np.arange(len(y))
        np.add.at(seen,idx,1)
        xb=xt[idx];yb=yt[idx];w,b,u,c=pars
        logits=torch.tanh(xb@w+b)@u+c
        losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,yb,reduction='none')
        loss=(losses*at[idx]).mean() if method=='weighted' else losses.mean()
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in pars): raise FloatingPointError('Nonfinite gradient')
        with torch.no_grad():
            for p in pars: p-=LR*p.grad;p.grad=None
        hist.append({'method':method,'seed':seed,'epoch':epoch,'objective_before':float(loss.detach()),'sampled_positive':int(y[idx].sum())})
    return [p.detach().numpy().copy() for p in pars],hist,seen

def hand_trace(theta=None):
    theta=array([.5,-.2,.3,.4,.1,.2,.6,-.5,.1] if theta is None else theta,'theta',1)
    if theta.shape!=(9,) or np.abs(theta).max()>1e6:raise ValueError('Nine finite float64 parameters with |theta| <= 1e6 required')
    x=np.array([[1.,2.],[-1.,1.]]);y=np.array([1.,0.]);a=np.array([3.,1.]);w=theta[:4].reshape(2,2);b=theta[4:6];u=theta[6:8];c=theta[8]
    z=x@w+b;h=np.maximum(z,0.);s=h@u+c
    if not np.isfinite(s).all() or np.abs(s).max()>30:
        raise ValueError('Explicit probability-local hand trace requires |logit| <= 30; use stable BCE for extreme logits')
    p=sigmoid(s);ell=bce(s,y);r=a/4*(p-y)
    dh=r[:,None]*u;dz=dh*(z>0);paths=[]
    for i in range(2): paths.append(np.r_[(x[i,:,None]*dz[i]).ravel(),dz[i],r[i]*h[i],r[i]])
    gradient=np.sum(paths,axis=0);new=theta-.1*gradient
    out={'theta':theta.tolist(),'x':x.tolist(),'y':y.tolist(),'weights':a.tolist(),'denominator':4,'preactivation':z.tolist(),'hidden':h.tolist(),'logits':s.tolist(),'probabilities':p.tolist(),'losses':ell.tolist(),'weighted_contributions':(a*ell/4).tolist(),'risk':float(a@ell/4),'local_dloss_dp':np.where(y==1,-1/p,1/(1-p)).tolist(),'local_dp_ds':(p*(1-p)).tolist(),'dlogits':r.tolist(),'dhidden':dh.tolist(),'dpreactivation':dz.tolist(),'paths':np.array(paths).tolist(),'gradient':gradient.tolist(),'dinput':(dz@w.T).tolist(),'theta_next':new.tolist()}
    return out

def json_bytes(obj): return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def csv_bytes(rows):
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return f.getvalue().encode()
def atomic(path,raw):
    temp=path.with_name(path.name+'.tmp');temp.write_bytes(raw);os.replace(temp,path)

def run(data_dir=None,output=None,epochs=EPOCHS):
    epochs=integer(epochs,'epochs',1,5000);data_dir=Path(data_dir) if data_dir is not None else ROOT/'data';output=Path(output) if output is not None else ROOT/'outputs'
    trainset=load_split(data_dir,'train');validation=load_split(data_dir,'validation')
    # Test byte integrity is checked without parsing labels before model selection.
    if hashlib.sha256((data_dir/'test.csv').read_bytes()).hexdigest()!=DATA_HASHES['test.csv']:raise ValueError('test.csv: data integrity mismatch')
    if output.resolve()==data_dir.resolve() or data_dir.resolve() in output.resolve().parents:raise ValueError('Output must not overwrite input directory')
    cm=validation['role']=='calibration';dm=validation['role']=='decision';pi=float(trainset['y'].mean());offset=math.log((1-pi)/pi)
    history=[];runs=[];curves=[]
    for method in METHODS:
        for seed in SEEDS:
            params,hist,seen=train(trainset['x'],trainset['y'],method,seed,epochs);history+=hist
            raw=predict(params,validation['x']);shift=0. if method=='uniform' else offset;corrected=raw-shift
            calibration=fit_temperature(corrected[cm],validation['y'][cm]);calraw=fit_temperature(raw[cm],validation['y'][cm])
            tuned,curve=choose_threshold(calibration['beta']*corrected[dm],validation['y'][dm])
            curves.extend({'method':method,'seed':seed,**r} for r in curve)
            runs.append({'method':method,'seed':seed,'params':[p.tolist() for p in params],'prior_logit_subtract':shift,'temperature':calibration,'temperature_without_prior':calraw,'threshold':tuned,'seen_min':int(seen.min()),'seen_max':int(seen.max()),'unique_seen':int((seen>0).sum()),'updates':epochs,'presentations':int(seen.sum())})
    frozen={'protocol':'039-v1','epochs':epochs,'learning_rate':LR,'train_prevalence':pi,'validation_calibration_n':int(cm.sum()),'validation_decision_n':int(dm.sum()),'cost_fp':FP_COST,'cost_fn':FN_COST,'threshold_candidates':THRESHOLDS.tolist(),'threshold_tie_break':'largest threshold','temperature_beta_bounds':[.05,20.],'test_labels_used_in_choices':False,'runs':runs}
    # All models, transformations and thresholds are fixed before reading test labels.
    test=load_split(data_dir,'test');reports=[];predictions=[];binrows=[]
    for r in runs:
        raw=predict(r['params'],test['x']);corrected=raw-r['prior_logit_subtract'];cal=r['temperature']['beta']*corrected
        views=[('raw',raw,.5),('raw_temperature',raw*r['temperature_without_prior']['beta'],.5),('prior',corrected,.5),('prior_temperature',cal,.5),('cost_theory',cal,.2),('cost_validation',cal,r['threshold'])]
        for view,z,t in views:
            for group in ['all','0','1']:
                mask=np.ones(len(z),bool) if group=='all' else test['group']==int(group)
                m=metrics(z[mask],test['y'][mask],t)
                head={'method':r['method'],'seed':r['seed'],'view':view,'group':group,'threshold':t}
                reports.append({**head,**{k:v for k,v in m.items() if k!='bins'},'ece5':metrics(z[mask],test['y'][mask],t,5)['ece'],'ece20':metrics(z[mask],test['y'][mask],t,20)['ece']})
                binrows.extend({**head,**b} for b in m['bins'])
            p=sigmoid(z)
            predictions.extend({'method':r['method'],'seed':r['seed'],'view':view,'id':test['id'][i],'group':int(test['group'][i]),'y':int(test['y'][i]),'logit':float(z[i]),'probability':float(p[i]),'threshold':t,'predicted':int(p[i]>=t)} for i in range(len(z)))
    hand=hand_trace();calculations={'first':hand,'next':hand_trace(hand['theta_next']),'risk_count_denominator':weighted_risk(hand['logits'],hand['y'],hand['weights'],'count'),'risk_weights_denominator':hand['risk']}
    files={'calculations.json':json_bytes(calculations),'frozen-plan.json':json_bytes(frozen),'training-history.csv':csv_bytes(history),'metrics.csv':csv_bytes(reports),'predictions.csv':csv_bytes(predictions),'reliability.csv':csv_bytes(binrows),'validation-cost-curves.csv':csv_bytes(curves)}
    output.mkdir(parents=True,exist_ok=True)
    for name,raw in files.items():atomic(output/name,raw)
    return {'protocol':'039-v1','epochs':epochs,'runs':len(runs),'updates':len(history),'presentations':sum(r['presentations'] for r in runs),'test_prediction_rows':len(predictions),'metric_rows':len(reports),'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--epochs',type=int,default=EPOCHS);a=p.parse_args()
    print(json.dumps(run(a.data_dir,a.output,a.epochs),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
