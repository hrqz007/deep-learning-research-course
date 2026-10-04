"""Unit 038: finite, offline CPU mechanisms of capacity and interpolation.

All public artifacts are calculated and serialized before any output is replaced.
The output directory replacement is per-file atomic, not a directory transaction.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, platform, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
N = 24
P_MAX = 96
WIDTHS = (1, 2, 3, 4, 8, 12, 16, 20, 22, 23, 24, 25, 26, 28, 32, 48, 72, 96)
NOISES = (0.0, 0.2, 0.8)
SEEDS = tuple(range(3800, 3812))
RIDGES = (0.0, 0.01, 0.1)
FIXTURE_HASH = '5918bdbd00997445ba621d8c0b95e989042535bb6bb5e6c93ed3eb89ddb8a90e'
X_HAND = np.array([[1., 0., 1.], [0., 1., 1.]])
Y_HAND = np.array([1., 2.])

def require(ok, message):
    if not ok:
        raise ValueError(message)

def finite_array(value, name, ndim=None):
    a = np.asarray(value, dtype=np.float64)
    require(np.all(np.isfinite(a)), name + ' must be finite')
    if ndim is not None:
        require(a.ndim == ndim, name + ' has wrong dimension')
    return a

def validate_xy(x, y):
    x = finite_array(x, 'x', 2); y = finite_array(y, 'y', 1)
    require(x.shape[0] == y.size and y.size > 0 and x.shape[1] > 0, 'incompatible x/y')
    return x, y

def linear_step(x, y, w, lr):
    x, y = validate_xy(x, y); w = finite_array(w, 'w', 1)
    require(w.size == x.shape[1], 'wrong w shape')
    require(np.isfinite(lr) and lr > 0, 'lr must be positive and finite')
    pred = x @ w; residual = pred-y; n = len(y)
    terms = x*w[None, :]
    dloss_dpred = residual/n
    contributions = dloss_dpred[:, None]*x
    gradient = contributions.sum(axis=0)
    after = w-lr*gradient
    return dict(w=w, products=terms, prediction=pred, residual=residual,
                sample_half_square=residual**2/2, loss=np.mean(residual**2)/2,
                dloss_dpred=dloss_dpred, dpred_dw=x, contributions=contributions,
                gradient=gradient, delta=-lr*gradient, after=after,
                next_prediction=x@after, next_loss=np.mean((x@after-y)**2)/2)

def hand_trace():
    w = np.zeros(3); steps=[]
    for _ in range(2):
        s=linear_step(X_HAND,Y_HAND,w,0.5);steps.append(s);w=s['after']
    t=torch.tensor(np.zeros(3),dtype=torch.float64,requires_grad=True)
    tx=torch.tensor(X_HAND);ty=torch.tensor(Y_HAND)
    loss=((tx@t-ty)**2).mean()/2;loss.backward()
    np.testing.assert_allclose(t.grad.numpy(),steps[0]['gradient'],atol=1e-14)
    return {'x':X_HAND,'y':Y_HAND,'lr':0.5,'steps':steps,
            'third_prediction':X_HAND@w,'third_loss':np.mean((X_HAND@w-Y_HAND)**2)/2,
            'autograd_first_gradient':t.grad.numpy()}

def least_squares(x, y, ridge=0.0, rcond=1e-12):
    x, y=validate_xy(x,y)
    require(np.isfinite(ridge) and ridge >= 0, 'ridge must be nonnegative and finite')
    require(np.isfinite(rcond) and 0 < rcond < 1, 'rcond must be in (0,1)')
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    kept=s>rcond*s[0]; rank=int(kept.sum())
    if ridge == 0:
        filt=np.divide(1.,s,out=np.zeros_like(s),where=kept)
    else:
        filt=s/(s*s+len(y)*ridge)
    w=vt.T@(filt*(u.T@y))
    return {'w':w,'singular_values':s,'rank':rank,'filter':filt,
            'train_mse':np.mean((x@w-y)**2),'norm2':w@w}

def gd_path(x,y,init,lr,steps):
    x,y=validate_xy(x,y);w=finite_array(init,'init',1).copy()
    require(w.size==x.shape[1], 'wrong init shape')
    require(np.isfinite(lr) and lr>0,'invalid lr')
    require(type(steps) is int and steps>=1, 'steps must be positive integer')
    rows=[]
    for t in range(steps+1):
        r=x@w-y;g=x.T@r/len(y)
        rows.append({'step':t,'w':w.copy(),'prediction':x@w,'loss':np.mean(r*r)/2,'gradient':g})
        if t<steps:w=w-lr*g
    return rows

def implicit_bias():
    x,y=X_HAND,Y_HAND; null=np.array([-1.,-1.,1.]); minimum=least_squares(x,y)['w']
    paths=[]
    for t in (0.,1.,-1.):
        rows=gd_path(x,y,t*null,0.5,80)
        paths.append({'null_coefficient':t,'rows':rows,'limit':minimum+t*null})
    scale=np.diag([1.,1.,2.]);u=least_squares(x@scale,y)['w'];scaled=scale@u
    return {'null_direction':null,'minimum_norm':minimum,'paths':paths,
            'query':np.array([1.,1.,0.]),'rescaled_coordinate_solution':u,
            'rescaled_function_coefficients':scaled,'coordinate_scale':scale,
            'rescaled_training_predictions':x@scaled}

def neural_step(width):
    require(width in (1,2),'only the two declared duplication cases are supported')
    x=X_HAND;y=Y_HAND;n=len(y)
    w=np.tile([.5,1.,0.],(width,1));b=np.full(width,.5);a=np.full(width,1/width);c=0.
    z=x@w.T+b;h=np.maximum(z,0);pred=h@a+c;r=pred-y;gout=r/n
    dz=gout[:,None]*a[None,:]*(z>0)
    contributions=[]
    for i in range(n):
        contributions.append(np.concatenate([(dz[i,:,None]*x[i]).ravel(),dz[i],gout[i]*h[i],[gout[i]]]))
    vector=np.concatenate([w.ravel(),b,a,[c]]);contributions=np.array(contributions);g=contributions.sum(axis=0)
    new=vector-.2*g;wn=new[:3*width].reshape(width,3);bn=new[3*width:4*width];an=new[4*width:5*width];cn=new[-1]
    new_z=x@wn.T+bn;new_pred=np.maximum(new_z,0)@an+cn
    return {'width':width,'parameter_order':[f'W{j+1}{k+1}' for j in range(width) for k in range(3)]+[f'b{j+1}' for j in range(width)]+[f'a{j+1}' for j in range(width)]+['c'],
            'before':vector,'z':z,'h':h,'prediction':pred,'residual':r,'loss':np.mean(r*r)/2,
            'dloss_dprediction':gout,'dpred_dh':a,'dh_dz':(z>0).astype(float),'dloss_dz':dz,
            'sample_contributions':contributions,'gradient':g,'lr':.2,'after':new,
            'next_z':new_z,'next_prediction':new_pred,'next_loss':np.mean((new_pred-y)**2)/2}

def load_fixture(path=None):
    p=ROOT/'data/gaussian-fixture.json' if path is None else Path(path)
    raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==FIXTURE_HASH,'fixture SHA256 mismatch')
    f=json.loads(raw);require(f['n']==N and f['p_max']==P_MAX and f['seeds']==list(SEEDS),'fixture metadata mismatch')
    require(len(f['draws'])==len(SEEDS),'wrong number of draws')
    for seed,d in zip(SEEDS,f['draws']):
        require(d['seed']==seed,'wrong seed order')
        x=finite_array(d['x'],'fixture x',2);e=finite_array(d['epsilon'],'epsilon',1)
        require(x.shape==(N,P_MAX) and e.shape==(N,),'wrong fixture shape')
    return f

def capacity_runs(fixture):
    beta=np.zeros(P_MAX);beta[:3]=[1.,-1.,.5]
    rows=[];details=[]
    for d in fixture['draws']:
        full=np.array(d['x']);eps=np.array(d['epsilon']);clean=full@beta
        for p in WIDTHS:
            x=full[:,:p];truth=beta[:p];omitted=float(beta[p:]@beta[p:])
            for sigma in NOISES:
                y=clean+sigma*eps
                for lam in RIDGES:
                    result=least_squares(x,y,lam);w=result['w'];s=result['singular_values']
                    signal=least_squares(x,clean,lam)['w'];noise=least_squares(x,sigma*eps,lam)['w']
                    biasvec=signal-truth;bias=float(biasvec@biasvec+omitted);variance_realized=float(noise@noise);cross=float(2*biasvec@noise)
                    risk=float((w-truth)@(w-truth)+omitted)
                    # Conditional on this full design, average only over new label noise.
                    conditional_noise_risk=float(bias+sigma*sigma*np.sum(result['filter']**2))
                    row={'seed':d['seed'],'p':p,'sigma':sigma,'ridge':lam,'rank':result['rank'],
                         'train_mse':float(result['train_mse']),'clean_risk':risk,'noisy_risk':risk+sigma*sigma,
                         'norm2':float(result['norm2']),'smallest_singular':float(s[-1]),
                         'largest_singular':float(s[0]),'noise_amplification':float(np.sum(result['filter']**2)),
                         'signal_error':bias,'realized_noise_norm2':variance_realized,'cross_term':cross,
                         'conditional_noise_mean_risk':conditional_noise_risk}
                    rows.append(row);details.append({'seed':d['seed'],'p':p,'sigma':sigma,'ridge':lam,
                         'weights':w,'prediction':x@w,'y':y,'singular_values':s,'signal_weights':signal,'noise_weights':noise})
    return rows,details

def conditional_demo():
    # Population target below is exact; these are additional independent Monte Carlo checks.
    rng=np.random.default_rng(3899);p=32;n=24
    x=rng.normal(size=(n,p));beta=np.zeros(p);beta[:3]=[1,-1,.5];clean=x@beta;sigma=.8
    fit=least_squares(x,clean);biasvec=fit['w']-beta
    expectation=float(biasvec@biasvec+sigma*sigma*np.sum(fit['filter']**2))
    risks=[]
    for _ in range(400):
        w=least_squares(x,clean+sigma*rng.normal(size=n))['w'];risks.append(float(np.sum((w-beta)**2)))
    w=least_squares(x,clean+sigma*rng.normal(size=n))['w'];test=rng.normal(size=(4096,p));err=(test@(w-beta))**2
    return {'seed':3899,'x':x,'beta':beta,'sigma':sigma,'noise_repetitions':400,'noise_risks':risks,
            'noise_expected_risk':expectation,'noise_mean':float(np.mean(risks)),
            'noise_mcse':float(np.std(risks,ddof=1)/20),'fixed_fit':w,
            'test_size':4096,'test_clean_mse':float(np.mean(err)),
            'test_mcse':float(np.std(err,ddof=1)/64),'exact_fixed_fit_risk':float(np.sum((w-beta)**2))}

def spectral_demo():
    x=np.diag([2.,.2]);y=np.array([1.,1.]);n=2;eta=.4;out=[]
    for t in (0,1,2,10,100,1000):
        s=np.array([2.,.2]);f=(1-(1-eta*s*s/n)**t)/s
        out.append({'steps':t,'filter':f,'w':f*y,'prediction':s*f*y,'loss':np.mean((s*f*y-y)**2)/2})
    return {'x':x,'y':y,'lr':eta,'gd':out,'ridge':[{'lambda':lam,**least_squares(x,y,lam)} for lam in (.001,.01,.1,1.)]}

def bound_demo():
    return [{'n':n,'M':m,'delta':.05,'epsilon':float(np.sqrt(np.log(2*m/.05)/(2*n)))}
            for n in (5,24,100,1000) for m in (1,100,1000000)]

def serializable(x):
    if isinstance(x,np.ndarray):return serializable(x.tolist())
    if isinstance(x,np.generic):return serializable(x.item())
    if isinstance(x,dict):return {str(k):serializable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serializable(v) for v in x]
    if isinstance(x,float):require(np.isfinite(x),'non-finite output')
    return x

def json_bytes(x):return (json.dumps(serializable(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def calculate():
    fixture=load_fixture();rows,details=capacity_runs(fixture)
    return {'hand-trace.json':hand_trace(),'implicit-bias.json':implicit_bias(),
            'duplication.json':{'same_x':X_HAND,'same_y':Y_HAND,'models':[neural_step(1),neural_step(2)]},
            'capacity-detail.json':details,'conditional-risk.json':conditional_demo(),
            'spectral-filters.json':spectral_demo(),'finite-class-bound.json':bound_demo()}, rows

def run(output):
    directory=Path(output)
    require(not directory.exists() or directory.is_dir(),'output must be directory')
    # No filesystem writes before all numerical work and serialization succeed.
    docs,rows=calculate();payload={name:json_bytes(doc) for name,doc in docs.items()}
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(serializable(rows));payload['capacity.csv']=stream.getvalue().encode()
    summary={'n':N,'widths':WIDTHS,'noise_levels':NOISES,'ridges':RIDGES,'seeds':SEEDS,'fit_count':len(rows),
             'fixture_sha256':FIXTURE_HASH,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
             'device':'CPU','dtype':'float64','interpretation':'Fixed 12 design/noise draws; exact clean population risk for known isotropic synthetic target. No real-world test ranking.'}
    payload['summary.json']=json_bytes(summary)
    directory.mkdir(parents=True,exist_ok=True)
    for name,data in payload.items():
        fd,tmp=tempfile.mkstemp(prefix='.'+name+'.',dir=directory)
        try:
            with os.fdopen(fd,'wb') as f:f.write(data)
            os.replace(tmp,directory/name)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args()
    torch.set_num_threads(1);print(json.dumps(run(a.output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
