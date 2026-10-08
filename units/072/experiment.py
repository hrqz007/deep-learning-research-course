"""DL072 independently runnable research project: corner shortcuts in tiny images.

Prespecified intervention: replace the top-left corner with label-independent
signs during training. The wrong-corner intervention is a specificity ablation.
All variants use the same fixed train/validation/test examples and step budget.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import torch
from torch import nn

SEEDS=[11,23,37]
VARIANTS=['linear','erm','random_corner','wrong_corner','mask_corner']
DOMAINS=['id','flip','blank','core_noise']
SIZE=12


def generate(n, seed):
    if n<=0: raise ValueError('n must be positive')
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,n,dtype=np.int64)
    x=rng.normal(0,.60,(n,1,SIZE,SIZE)).astype('float32')
    # Class 0: horizontal line; class 1: vertical line. The line never crosses corners.
    for i,label in enumerate(y):
        pos=int(rng.integers(5,8))
        if label==0: x[i,0,pos,3:10]+=.85
        else: x[i,0,3:10,pos]+=.85
    # A 95%-consistent corner is easier to fit but is not the intended shape signal.
    sign=2*y-1
    agreement=rng.random(n)<.95
    corner_sign=np.where(agreement,sign,-sign)
    x[:,:,0:3,0:3]=2.2*corner_sign[:,None,None,None]
    return x,y,agreement.astype(np.int64)


def domains_from(x, seed=4004):
    rng=np.random.default_rng(seed)
    d={'id':x.copy(),'flip':x.copy(),'blank':x.copy(),'core_noise':x.copy()}
    d['flip'][:,:,0:3,0:3]*=-1
    d['blank'][:,:,0:3,0:3]=0
    # Added noise affects the core region only; it leaves the shortcut unchanged.
    d['core_noise'][:,:,3:10,3:10]+=rng.normal(0,.9,(len(x),1,7,7)).astype('float32')
    return d


def network(variant):
    if variant not in VARIANTS: raise ValueError('unknown variant')
    if variant=='linear': return nn.Sequential(nn.Flatten(),nn.Linear(SIZE*SIZE,2))
    return nn.Sequential(nn.Flatten(),nn.Linear(SIZE*SIZE,32),nn.Tanh(),nn.Linear(32,2))


def transform(x, variant, generator=None, training=False):
    """Labels are deliberately absent from this function's signature."""
    z=x.clone()
    if variant=='mask_corner': z[:,:,0:3,0:3]=0
    if training and variant in ['random_corner','wrong_corner']:
        if generator is None: raise ValueError('training augmentation requires a seeded generator')
        signs=(2*torch.randint(0,2,(len(x),1,1,1),generator=generator)-1).float()*2.2
        if variant=='random_corner': z[:,:,0:3,0:3]=signs
        else: z[:,:,9:12,9:12]=signs
    return z


def fit(x,y,variant,seed,steps=160):
    if steps<=0: raise ValueError('steps must be positive')
    torch.manual_seed(seed)
    m=network(variant)
    opt=torch.optim.Adam(m.parameters(),lr=.006,weight_decay=1e-4)
    g=torch.Generator().manual_seed(seed+1000)
    xt,yt=torch.from_numpy(x),torch.from_numpy(y)
    losses=[];start=time.perf_counter()
    m.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        # Same base examples, full-batch size and update count in every variant.
        z=transform(xt,variant,g,training=True)
        loss=nn.functional.cross_entropy(m(z),yt)
        loss.backward();opt.step();losses.append(float(loss.detach()))
    seconds=time.perf_counter()-start
    m.eval()
    return m,losses,seconds


def predict(m,x,variant):
    m.eval()
    with torch.inference_mode():
        logits=m(transform(torch.as_tensor(x,dtype=torch.float32),variant,training=False))
        return logits.softmax(1).double().numpy()


def metrics(p,y):
    p,y=np.asarray(p),np.asarray(y)
    if p.ndim!=2 or p.shape!=(len(y),2) or not len(y): raise ValueError('nonempty matching inputs required')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-6): raise ValueError('invalid probabilities')
    if not np.isin(y,[0,1]).all(): raise ValueError('labels must be binary')
    err=(p.argmax(1)!=y)
    return {'n':len(y),'errors':int(err.sum()),'error':float(err.mean()),
            'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            'confident_wrong':int((err & (p.max(1)>=.95)).sum())}


def selective(p,y,tau):
    if not 0<=tau<=1: raise ValueError('threshold outside [0,1]')
    accept=p.max(1)>=tau;k=int(accept.sum())
    errors=int(((p.argmax(1)!=y)&accept).sum())
    return {'accepted':k,'coverage':k/len(y),'risk':errors/k if k else None,'errors':errors,'threshold':float(tau)}


def curve(p,y):
    confidence=p.max(1);order=np.argsort(-confidence,kind='stable');conf=confidence[order]
    endpoints=np.r_[np.flatnonzero(conf[:-1]!=conf[1:])+1,len(y)]
    errors=(p.argmax(1)!=y)[order]
    return {'coverage':(endpoints/len(y)).tolist(),'risk':(np.cumsum(errors)[endpoints-1]/endpoints).tolist()}


def summary(runs):
    s={}
    for v in VARIANTS:
        selected=[r for r in runs if r['variant']==v]
        s[v]={'parameters':selected[0]['parameters'],'train_seconds_mean':float(np.mean([r['train_seconds'] for r in selected])),
              'domains':{}}
        for d in DOMAINS:
            values=[r['domains'][d]['error'] for r in selected]
            s[v]['domains'][d]={'mean_error':float(np.mean(values)),'sd_error':float(np.std(values,ddof=1)),'values':values}
    deltas=[]
    for seed in SEEDS:
        erm=next(r for r in runs if r['seed']==seed and r['variant']=='erm')
        new=next(r for r in runs if r['seed']==seed and r['variant']=='random_corner')
        deltas.append(erm['domains']['flip']['error']-new['domains']['flip']['error'])
    return s,{'definition':'ERM flip error minus random_corner flip error; positive favors intervention',
               'by_seed':deltas,'mean':float(np.mean(deltas)),'sd':float(np.std(deltas,ddof=1)),
               'not_a_generalization_CI':True}


def run(output='outputs',steps=160):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    train=generate(800,4001);val=generate(240,4002);test=generate(600,4003)
    domains=domains_from(test[0])
    data={'train_x':train[0],'train_y':train[1],'validation_x':val[0],'validation_y':val[1],
          'test_y':test[1],'test_shortcut_agreement':test[2]}
    data.update({name+'_x':x for name,x in domains.items()})
    np.savez_compressed(out/'data.npz',**data)
    preds={};runs=[]
    for seed in SEEDS:
        for variant in VARIANTS:
            m,losses,seconds=fit(train[0],train[1],variant,seed,steps)
            pv=predict(m,val[0],variant)
            tau=float(np.quantile(pv.max(1),.2,method='lower'))
            ck={'state_dict':m.state_dict(),'variant':variant,'seed':seed,'steps':steps,'threshold':tau,
                'image_size':SIZE,'purpose':'inference only'}
            ckpath=out/f'{variant}_seed{seed}.pt';torch.save(ck,ckpath)
            reload=network(variant);reload.load_state_dict(torch.load(ckpath,weights_only=True)['state_dict'])
            diff=float(np.max(np.abs(predict(m,test[0],variant)-predict(reload,test[0],variant))))
            r={'variant':variant,'seed':seed,'parameters':sum(p.numel() for p in m.parameters()),
               'losses':losses,'train_seconds':seconds,'validation':metrics(pv,val[1]),
               'threshold':tau,'checkpoint_reload_max_abs':diff,'domains':{},'selective':{},'risk_coverage':{}}
            for name,x in domains.items():
                p=predict(m,x,variant);preds[f'{variant}_seed{seed}_{name}']=p
                r['domains'][name]=metrics(p,test[1]);r['selective'][name]=selective(p,test[1],tau)
                r['risk_coverage'][name]=curve(p,test[1])
            # Groups defined before evaluation by test shortcut agreement.
            p=preds[f'{variant}_seed{seed}_id'];r['id_groups']={}
            for g in [0,1]:
                mask=test[2]==g;r['id_groups'][str(g)]=metrics(p[mask],test[1][mask])
            runs.append(r)
    np.savez_compressed(out/'predictions.npz',**preds)
    aggregates,delta=summary(runs)
    result={'unit':'072','research_question':'Does label-independent corner randomization reduce shortcut-reversal error?',
            'config':{'seeds':SEEDS,'variants':VARIANTS,'domains':DOMAINS,'steps':steps,'n_train':800,
             'n_validation':240,'n_test':600,'data_seeds':[4001,4002,4003,4004],
             'target_validation_coverage':.8,'device':'cpu','dtype':'float32','threads':1,
             'predeclared_primary_metric':'flip test error','search_trials':0,
             'hypothesis':'random_corner reduces flip error versus erm; ID cost is reported separately'},
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
             'elapsed_seconds':time.perf_counter()-start},'runs':runs,'aggregate':aggregates,'paired_difference':delta,
            'data_sha256':hashlib.sha256((out/'data.npz').read_bytes()).hexdigest(),
            'budget':{'models':len(runs),'optimizer_updates':len(runs)*steps,'training_examples_processed':len(runs)*steps*800,
             'sum_training_seconds':sum(r['train_seconds'] for r in runs),'training_tensor_bytes':int(train[0].nbytes+train[1].nbytes),
             'checkpoint_total_bytes':sum(p.stat().st_size for p in out.glob('*.pt')),
             'timing_scope':'per-model optimization only; process total includes data, evaluation, saving; installation and document build excluded'},
            'limitations':['Synthetic known nuisance location; no real-image validation.',
             'Fixed data and three seeds capture initialization/augmentation variability, not population uncertainty.',
             'Equal update counts are not equal FLOPs; linear has fewer parameters.',
             'No hyperparameter search or architecture search; no novelty claim.',
             'Inference checkpoints omit optimizer/RNG states and do not implement exact resumption.']}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');p.add_argument('--steps',type=int,default=160)
    a=p.parse_args();r=run(a.output,a.steps)
    for variant,v in r['aggregate'].items():
        print(variant,{k:round(d['mean_error'],4) for k,d in v['domains'].items()})
    print('paired improvement',r['paired_difference']['by_seed'])
