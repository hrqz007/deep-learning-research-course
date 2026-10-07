"""DL062: discrete-time, continuous-state DDPM; indexing is t=0,...,T.
Network is genuinely trained. All random data are original synthetic data.
"""
from pathlib import Path
import argparse,json,platform,time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent

def positive_int(name,v):
    if not isinstance(v,int) or isinstance(v,bool) or v<=0:raise ValueError(name+' must be a positive integer')

def schedule(T=200,beta_start=.0001,beta_end=.05):
    positive_int('T',T)
    if not 0<beta_start<=beta_end<1:raise ValueError('need 0 < beta_start <= beta_end < 1')
    beta=torch.cat([torch.zeros(1),torch.linspace(beta_start,beta_end,T)])
    alpha=1-beta;abar=torch.cumprod(alpha,0)
    posterior=torch.zeros_like(beta);posterior[1:]=beta[1:]*(1-abar[:-1])/(1-abar[1:])
    return {'T':T,'beta':beta,'alpha':alpha,'abar':abar,'posterior_var':posterior}

def valid_t(t,n,s,allow_zero=False):
    if t.ndim!=1 or len(t)!=n or t.dtype!=torch.long:raise ValueError('t must be a long tensor of shape (batch,)')
    if bool((t<(0 if allow_zero else 1)).any()) or bool((t>s['T']).any()):raise ValueError('t outside schedule')

def q_sample(x0,t,eps,s):
    if x0.ndim!=2 or x0.shape[1]!=2 or x0.shape!=eps.shape or len(x0)==0:raise ValueError('x0 and eps must be matching nonempty N x 2')
    if not bool(torch.isfinite(x0).all()) or not bool(torch.isfinite(eps).all()):raise ValueError('nonfinite input')
    valid_t(t,len(x0),s,True);a=s['abar'][t,None]
    return a.sqrt()*x0+(1-a).sqrt()*eps

def reverse_mean(xt,t,eps_pred,s):
    if xt.shape!=eps_pred.shape or xt.ndim!=2 or xt.shape[1]!=2 or len(xt)==0:raise ValueError('shape mismatch')
    valid_t(t,len(xt),s)
    return (xt-s['beta'][t,None]/(1-s['abar'][t,None]).sqrt()*eps_pred)/s['alpha'][t,None].sqrt()

def posterior_mean(xt,x0,t,s):
    valid_t(t,len(xt),s)
    b=s['beta'][t,None];a=s['alpha'][t,None];ab=s['abar'][t,None];prev=s['abar'][t-1,None]
    return b*prev.sqrt()/(1-ab)*x0+(1-prev)*a.sqrt()/(1-ab)*xt

def centers():
    angle=np.arange(8)*2*np.pi/8
    return np.stack([2*np.cos(angle),2*np.sin(angle)],axis=1).astype('float32')

def make_data(n=4096,seed=620):
    positive_int('n',n);rng=np.random.default_rng(seed);labels=rng.integers(0,8,n)
    return (centers()[labels]+rng.normal(0,.08,(n,2))).astype('float32'),labels

def metrics(samples,radius=.35):
    x=np.asarray(samples)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)==0 or not np.isfinite(x).all():raise ValueError('finite N x 2 required')
    d=np.linalg.norm(x[:,None]-centers()[None],axis=2);ids=d.argmin(1);ok=d.min(1)<=radius
    counts=np.bincount(ids[ok],minlength=8);p=counts/max(counts.sum(),1)
    return dict(coverage=int((counts/len(x)>=.01).sum()),valid_fraction=float(ok.mean()),counts=counts.tolist(),conditional_max_mass=float(p.max()),mean_nearest_distance=float(d.min(1).mean()))

class EpsilonNet(nn.Module):
    def __init__(self,T=200):
        super().__init__();positive_int('T',T);self.T=T
        self.register_buffer('frequencies',torch.tensor([1.,2.,4.,8.,16.,32.]))
        self.net=nn.Sequential(nn.Linear(15,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,2))
    def embedding(self,t):
        u=t.float()[:,None]/self.T;angles=u*self.frequencies[None]*np.pi
        return torch.cat([u,angles.sin(),angles.cos()],dim=1)
    def forward(self,x,t):return self.net(torch.cat([x,self.embedding(t)],dim=1))

@torch.no_grad()
def sample(model,s,n=4096,seed=1,terminal_noise=False,return_path=False):
    positive_int('n',n);rng=torch.Generator().manual_seed(seed);x=torch.randn(n,2,generator=rng);path={str(s['T']):x.numpy().copy()}
    for t_int in range(s['T'],0,-1):
        t=torch.full((n,),t_int,dtype=torch.long);mu=reverse_mean(x,t,model(x,t),s)
        if t_int>1:x=mu+s['posterior_var'][t_int].sqrt()*torch.randn(n,2,generator=rng)
        elif terminal_noise:x=mu+s['beta'][1].sqrt()*torch.randn(n,2,generator=rng)
        else:x=mu
        if t_int-1 in [150,100,50,10,0]:path[str(t_int-1)]=x.numpy().copy()
    return (x.numpy(),path) if return_path else x.numpy()

def train(seed,steps=7000,output=None):
    positive_int('steps',steps);torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    s=schedule();x,_=make_data();real=torch.from_numpy(x);model=EpsilonNet();opt=torch.optim.Adam(model.parameters(),lr=.001)
    validation,_=make_data(2048,621);vx=torch.from_numpy(validation);rg=torch.Generator().manual_seed(622)
    vt=torch.randint(1,201,(len(vx),),generator=rg);ve=torch.randn(vx.shape,generator=rg);vxt=q_sample(vx,vt,ve,s)
    history=[];start=time.monotonic()
    for step in range(steps+1):
        if step>0:
            idx=torch.randint(len(real),(256,));x0=real[idx];t=torch.randint(1,201,(len(x0),));eps=torch.randn_like(x0)
            xt=q_sample(x0,t,eps,s);loss=((model(xt,t)-eps)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%500==0 or step==steps:
            with torch.no_grad():vl=float(((model(vxt,vt)-ve)**2).mean())
            history.append(dict(step=step,train_mse=None if step==0 else float(loss.detach()),validation_mse=vl))
    generated,path=sample(model,s,seed=7000+seed,return_path=True)
    exact=sample(model,s,seed=7000+seed,terminal_noise=True)
    with torch.no_grad():squared=(model(vxt,vt)-ve)**2
    bins=[]
    for lo,hi in [(1,20),(21,60),(61,120),(121,200)]:
        mask=(vt>=lo)&(vt<=hi);bins.append(dict(t_start=lo,t_end=hi,n=int(mask.sum()),mse=float(squared[mask].mean())))
    result=dict(seed=seed,steps=steps,batch=256,learning_rate=.001,seconds=round(time.monotonic()-start,3),history=history,final=metrics(generated),terminal_gaussian_sample=metrics(exact),zero_predictor_validation_mse=float((ve**2).mean()),time_bins=bins)
    if output is not None:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        torch.save({'epsilon_model':model.state_dict(),'T':200,'beta_start':.0001,'beta_end':.05,'seed':seed,'steps':steps},out/f'ddpm_seed{seed}.pt')
        np.savez_compressed(out/f'ddpm_seed{seed}_samples.npz',samples=generated,terminal_gaussian_samples=exact,**{'path_'+k:v for k,v in path.items()})
    return result

def main(output=ROOT/'outputs',steps=7000):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    x,l=make_data();test,tl=make_data(4096,623);np.savez_compressed(ROOT/'data/mixture.npz',train=x,train_labels=l,reference=test,reference_labels=tl,centers=centers())
    s=schedule();np.savez_compressed(out/'schedule.npz',**{k:v.numpy() for k,v in s.items() if k!='T'})
    r=dict(unit='062',python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,data_seed=620,validation_seed=621,validation_noise_seed=622,reference_seed=623,T=200,beta_start=.0001,beta_end=.05,abar_T=float(s['abar'][-1]),evaluation=dict(radius=.35,min_fraction_all_samples=.01,sample_count=4096),reference=metrics(test),runs=[])
    for seed in [11,23,37]:
        row=train(seed,steps,out);r['runs'].append(row);print(seed,row['final'],row['history'][-1],flush=True)
    (out/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=7000);p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args();main(a.output,a.steps)
