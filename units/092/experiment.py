"""DL092: reproducible multimodal evidence audit. CPU, NumPy, synthetic data only."""
from pathlib import Path
import argparse, json, time, platform
import numpy as np
SEEDS=(11,23,37)
METHODS=('vision','text','concat','interaction','randomized_interaction')

def data(n,seed):
    r=np.random.default_rng(seed)
    a=r.choice([-1.,1.],n); b=r.choice([-1.,1.],n)
    y=(a*b>0).astype(float); group=r.integers(0,2,n)
    v=a+r.normal(size=n)*np.where(group==0,.25,.9)
    shortcut=(2*y-1)*np.where(r.random(n)<.95,1.,-1.)
    return np.column_stack((v,b,shortcut)),y,group,a

def features(x,method):
    v,b,z=x.T
    if method=='vision': return np.column_stack((v,np.ones(len(x))))
    if method=='text': return np.column_stack((b,z,np.ones(len(x))))
    if method=='concat': return np.column_stack((v,b,z,np.ones(len(x))))
    return np.column_stack((v,b,z,v*b,np.ones(len(x))))

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-35,35)))
def fit(x,y,method,seed,steps=600):
    r=np.random.default_rng(seed); w=r.normal(0,.02,features(x,method).shape[1]); losses=[]
    for step in range(steps):
        xx=x.copy()
        if method=='randomized_interaction': xx[:,2]=r.choice([-1.,1.],len(x))
        f=features(xx,method); p=sigmoid(f@w)
        grad=f.T@(p-y)/len(y)+1e-4*w
        w-=.12*grad
        if step%50==0 or step==steps-1:
            losses.append(float(-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))))
    return w,losses

def domains(x,y,a):
    d={'iid':(x.copy(),y.copy())}
    xx=x.copy();xx[:,2]*=-1;d['shortcut_flip']=(xx,y.copy())
    xx=x.copy();xx[:,2]=np.random.default_rng(92).choice([-1.,1.],len(x));d['shortcut_random']=(xx,y.copy())
    xx=x.copy();xx[:,0]=0;d['missing_vision']=(xx,y.copy())
    xx=x.copy();xx[:,1:]=0;d['missing_text']=(xx,y.copy())
    xx=x.copy();xx[:,1]*=-1;d['question_counterfactual']=(xx,1-y)
    perm=np.random.default_rng(93).permutation(len(x));xx=x.copy();xx[:,1:]=xx[perm,1:]
    d['permuted_original_label']=(xx,y.copy())
    d['permuted_relabelled']=(xx,(a*xx[:,1]>0).astype(float))
    return d

def paired_bootstrap(diff,seed=2026,repeats=2000):
    r=np.random.default_rng(seed)
    means=np.mean(diff[r.integers(0,len(diff),(repeats,len(diff)))],axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def main(output='outputs'):
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    train,yt,gt,at=data(1200,9201);val,yv,gv,av=data(400,9202);test,y,g,a=data(800,9203)
    np.savez_compressed(out/'data.npz',train=train,train_y=yt,validation=val,validation_y=yv,test=test,test_y=y,group=g,latent_a=a)
    ds=domains(test,y,a);rows=[];pred={};weights={};curves={};reload_error=0.
    for method in METHODS:
        for seed in SEEDS:
            w,loss=fit(train,yt,method,seed); key=f'{method}_{seed}';weights[key]=w;curves[key]=loss
            for domain,(x,labels) in ds.items():
                p=sigmoid(features(x,method)@w);pred[key+'_'+domain]=p;err=(p>=.5)!=labels
                rows.append(dict(method=method,seed=seed,domain=domain,n=len(y),error=float(err.mean()),group0_error=float(err[g==0].mean()),group1_error=float(err[g==1].mean())))
    np.savez(out/'weights.npz',**weights);np.savez_compressed(out/'predictions.npz',**pred)
    loaded=np.load(out/'weights.npz')
    for key,w in weights.items(): reload_error=max(reload_error,float(np.max(np.abs(w-loaded[key]))))
    grouped=[]
    for method in METHODS:
        for domain in ds:
            vals=[r['error'] for r in rows if r['method']==method and r['domain']==domain]
            grouped.append(dict(method=method,domain=domain,error_mean=float(np.mean(vals)),seed_sd=float(np.std(vals,ddof=1))))
    # Average the three seed-specific paired errors per test example, then resample examples.
    delta=np.mean([((pred[f'randomized_interaction_{s}_shortcut_flip']>=.5)!=y).astype(float)-((pred[f'interaction_{s}_shortcut_flip']>=.5)!=y).astype(float) for s in SEEDS],axis=0)
    result=dict(schema_version=1,n_train=1200,n_validation=400,n_test=800,seeds=list(SEEDS),steps=600,learning_rate=.12,results=rows,aggregate=grouped,losses=curves,paired_delta=float(delta.mean()),paired_example_bootstrap_ci=paired_bootstrap(delta),reload_error=reload_error,seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,limitations=['Synthetic two-bit task; fixed known interaction feature','Three seeds vary initialization and randomization, not population','Bootstrap over fixed test examples after seed averaging; not full seed uncertainty','Missing-modality zeros are diagnostic OOD inputs, not retrained baselines'])
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    import csv
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'seconds':result['seconds'],'paired_delta':result['paired_delta'],'ci':result['paired_example_bootstrap_ci'],'reload_error':reload_error},indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args();main(a.output)
