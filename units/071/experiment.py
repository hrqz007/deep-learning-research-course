"""DL071: offline, CPU-only selective prediction under prespecified shifts.

The calibration split is the only source for temperature and threshold choices.
Test labels are used only in reporting. Synthetic domains are paired transforms
of a common test split; they are not independent draws from a real deployment.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import torch
from torch import nn

SEEDS = [17, 29, 43]
DOMAINS = ['id', 'core_noise', 'shortcut_fade', 'shortcut_flip', 'subgroup_flip']


def validate(p, y):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y)
    if p.ndim != 2 or p.shape[1] != 2 or y.shape != (len(p),) or len(p) == 0:
        raise ValueError('Expected nonempty (N,2) probabilities and (N,) labels')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Probabilities must be finite in [0,1]')
    if not np.allclose(p.sum(1), 1, atol=1e-6) or not np.isin(y, [0,1]).all():
        raise ValueError('Rows must sum to one and labels must be binary')
    return p, y.astype(np.int64)


def generate(n, seed):
    if n <= 0:
        raise ValueError('n must be positive')
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n, dtype=np.int64)
    sign = 2*y-1
    x = np.column_stack([sign + rng.normal(0, 1.15, n),
                         2.2*sign + rng.normal(0, .45, n)]).astype('float32')
    group = (rng.random(n) < .25).astype(np.int64)
    return x, y, group


def shift_domains(x, group, seed=904):
    """The list and severities are set before fitting any model."""
    rng = np.random.default_rng(seed)
    domains = {'id': x.copy()}
    domains['core_noise'] = x.copy()
    domains['core_noise'][:,0] += rng.normal(0, 1.8, len(x))
    domains['shortcut_fade'] = x.copy()
    domains['shortcut_fade'][:,1] *= .1
    domains['shortcut_flip'] = x.copy()
    domains['shortcut_flip'][:,1] *= -1
    domains['subgroup_flip'] = x.copy()
    domains['subgroup_flip'][group == 1,1] *= -1
    return domains


def model():
    return nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 24),
                         nn.Tanh(), nn.Linear(24, 2))


def fit(x, y, seed, steps=220):
    if steps <= 0:
        raise ValueError('steps must be positive')
    torch.manual_seed(seed)
    m = model()
    opt = torch.optim.Adam(m.parameters(), lr=.012)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    losses = []
    start = time.perf_counter()
    m.train()
    for step in range(steps):
        # CrossEntropyLoss takes logits; do not apply softmax before it.
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(m(xt), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter()-start
    m.eval()
    return m, losses, elapsed


def logits(m, x):
    m.eval()
    with torch.inference_mode():
        return m(torch.as_tensor(x, dtype=torch.float32)).double().numpy()


def softmax(z, temperature=1.):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    a = np.asarray(z, dtype=np.float64)/temperature
    a -= a.max(1, keepdims=True)
    e = np.exp(a)
    return e/e.sum(1, keepdims=True)


def nll(p, y):
    p,y = validate(p,y)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def calibrate(z, y):
    # Finite predeclared search; calibration is post-hoc, never test-tuned.
    grid = np.geomspace(.5, 5., 81)
    vals = [nll(softmax(z,t), y) for t in grid]
    i = int(np.argmin(vals))
    return float(grid[i]), {'temperatures':grid.tolist(), 'nll':vals,
                           'on_boundary':i in [0,len(grid)-1]}


def threshold_for_coverage(p, target=.8):
    if not 0 < target <= 1:
        raise ValueError('target must be in (0,1]')
    p,_ = validate(p, np.zeros(len(p), dtype=int))
    # Accept >= threshold. Ties are retained together; coverage may exceed target.
    return float(np.quantile(p.max(1), 1-target, method='lower'))


def wilson(errors, total, z=1.96):
    if total == 0:
        return None
    r = errors/total
    d = 1 + z*z/total
    center = (r+z*z/(2*total))/d
    radius = z*np.sqrt(r*(1-r)/total + z*z/(4*total*total))/d
    return [float(max(0.,center-radius)), float(min(1.,center+radius))]


def selective(p, y, threshold):
    p,y = validate(p,y)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0,1]')
    accepted = p.max(1) >= threshold
    error = p.argmax(1) != y
    k = int(accepted.sum())
    e = int((accepted & error).sum())
    return {'n':len(y), 'accepted':k, 'errors_accepted':e, 'threshold':float(threshold),
            'coverage':k/len(y), 'risk':e/k if k else None,
            'risk_wilson95':wilson(e,k), 'full_error':float(error.mean())}


def risk_coverage(p,y):
    p,y = validate(p,y)
    conf = p.max(1)
    order = np.argsort(-conf, kind='stable')
    c = conf[order]
    err = (p.argmax(1)!=y)[order]
    # Only threshold-realizable endpoints: never split equal-confidence ties.
    endpoints = np.r_[np.flatnonzero(c[:-1] != c[1:])+1, len(c)]
    return {'coverage':(endpoints/len(c)).tolist(),
            'risk':(np.cumsum(err)[endpoints-1]/endpoints).tolist(),
            'threshold':c[endpoints-1].tolist()}


def reliability(p,y,bins=10):
    p,y = validate(p,y)
    conf, correct = p.max(1), p.argmax(1)==y
    indices = np.minimum((conf*bins).astype(int),bins-1)
    rows=[]
    ece=0.
    for b in range(bins):
        mask=indices==b
        count=int(mask.sum())
        if count:
            c,a=float(conf[mask].mean()),float(correct[mask].mean())
            ece += count/len(y)*abs(a-c)
            rows.append({'bin':b,'n':count,'confidence':c,'accuracy':a})
    return {'ece':ece,'bins':rows}


def entropy(p):
    p=np.asarray(p,dtype=float)
    return -(p*np.log(np.clip(p,1e-12,1))).sum(-1)


def run(output='outputs', steps=220):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(output); out.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter()
    train=generate(800,901); cal=generate(400,902); test=generate(600,903)
    domains=shift_domains(test[0],test[2])
    data={'train_x':train[0],'train_y':train[1],'cal_x':cal[0],'cal_y':cal[1],
          'test_y':test[1],'test_group':test[2]}
    for name,x in domains.items(): data[name+'_x']=x
    np.savez_compressed(out/'data.npz',**data)
    runs=[]; all_prob={name:[] for name in DOMAINS}; pred_data={}
    for seed in SEEDS:
        m,losses,seconds=fit(train[0],train[1],seed,steps)
        t,search=calibrate(logits(m,cal[0]),cal[1])
        tau=threshold_for_coverage(softmax(logits(m,cal[0]),t))
        ck={'state_dict':m.state_dict(),'seed':seed,'steps':steps,'temperature':t,
            'threshold':tau,'architecture':'2-24-tanh-24-tanh-2','purpose':'inference only'}
        torch.save(ck,out/f'model_seed{seed}.pt')
        summary={'seed':seed,'temperature':t,'threshold':tau,'calibration_search':search,
                 'losses':losses,'train_seconds':seconds,'domains':{},'checkpoint_reload_max_abs':0.}
        reloaded=model(); reloaded.load_state_dict(torch.load(out/f'model_seed{seed}.pt',weights_only=True)['state_dict'])
        summary['checkpoint_reload_max_abs']=float(np.max(np.abs(logits(m,test[0])-logits(reloaded,test[0]))))
        for name,x in domains.items():
            z=logits(m,x); p=softmax(z,t); raw=softmax(z)
            all_prob[name].append(p)
            d={'selective':selective(p,test[1],tau),'raw_nll':nll(raw,test[1]),'calibrated_nll':nll(p,test[1]),
               'raw_reliability':reliability(raw,test[1]),'calibrated_reliability':reliability(p,test[1]),
               'risk_coverage':risk_coverage(p,test[1]),'groups':{}}
            for g in [0,1]:
                mask=test[2]==g
                d['groups'][str(g)]=selective(p[mask],test[1][mask],tau)
            wrong=(p.argmax(1)!=test[1]) & (p.max(1)>=.95)
            ids=np.flatnonzero(wrong)
            d['confident_wrong_count']=int(len(ids))
            d['confident_wrong_examples']=[{'index':int(i),'x':x[i].tolist(),'y':int(test[1][i]),
                'prediction':int(p[i].argmax()),'confidence':float(p[i].max())} for i in ids[:5]]
            summary['domains'][name]=d
            pred_data[f'seed{seed}_{name}_p']=p
            pred_data[f'seed{seed}_{name}_raw']=raw
        runs.append(summary)
    ensembles={}
    for name,ps in all_prob.items():
        stack=np.stack(ps); mean=stack.mean(0)
        # Entropy difference is disagreement, not a certified epistemic probability.
        pred_h=entropy(mean); avg_h=entropy(stack).mean(0); disagreement=np.maximum(pred_h-avg_h,0)
        ensembles[name]={'full_error':float((mean.argmax(1)!=test[1]).mean()),
                         'mean_predictive_entropy':float(pred_h.mean()),
                         'mean_member_entropy':float(avg_h.mean()),
                         'mean_disagreement':float(disagreement.mean())}
        pred_data[name+'_ensemble_p']=mean; pred_data[name+'_disagreement']=disagreement
    np.savez_compressed(out/'predictions.npz',**pred_data)
    result={'unit':'071','config':{'seeds':SEEDS,'steps':steps,'n_train':800,'n_calibration':400,'n_test':600,
            'data_seeds':[901,902,903,904],'target_calibration_coverage':.8,
            'domains':DOMAINS,'threads':1,'device':'cpu','dtype':'float32'},
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
                       'elapsed_seconds':time.perf_counter()-started},'runs':runs,'ensembles':ensembles,
            'data_sha256':hashlib.sha256((out/'data.npz').read_bytes()).hexdigest(),
            'limitations':['Synthetic paired stress tests are not a deployment safety guarantee.',
             'Three seeds vary initialization only; data are fixed.','Calibration threshold targets coverage, not a risk bound.',
             'Ensemble members can share the same shortcut.','Checkpoints support inference, not exact mid-run resumption.']}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs');parser.add_argument('--steps',type=int,default=220)
    args=parser.parse_args();r=run(args.output,args.steps)
    for name,d in r['runs'][0]['domains'].items():
        s=d['selective'];print(name,'coverage',round(s['coverage'],4),'risk',s['risk'],'confident_wrong',d['confident_wrong_count'])
