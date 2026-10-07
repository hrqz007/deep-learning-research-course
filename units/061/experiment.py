"""DL061: genuine alternating GAN training on an original eight-mode mixture.
All data, optimizer trajectories and diagnostic samples are generated locally.
"""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent

def require_positive(name,value):
    if not isinstance(value,int) or isinstance(value,bool) or value<=0: raise ValueError(name+' must be a positive integer')

def centers():
    angle=np.arange(8)*2*np.pi/8
    return np.stack([2*np.cos(angle),2*np.sin(angle)],axis=1).astype('float32')

def make_data(n=4096,seed=610):
    require_positive('n',n)
    rng=np.random.default_rng(seed); labels=rng.integers(0,8,size=n)
    return (centers()[labels]+rng.normal(0,.08,(n,2))).astype('float32'),labels

def mode_metrics(samples,radius=.35,min_fraction=.01):
    x=np.asarray(samples)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)==0 or not np.isfinite(x).all(): raise ValueError('samples must be finite nonempty N x 2')
    if not 0<radius<.7 or not 0<min_fraction<1:raise ValueError('invalid evaluation thresholds')
    dist=((x[:,None,:]-centers()[None,:,:])**2).sum(2)**.5
    labels=dist.argmin(1);valid=dist.min(1)<=radius
    counts=np.bincount(labels[valid],minlength=8)
    fractions=counts/len(x);coverage=int((fractions>=min_fraction).sum())
    p=counts/max(int(counts.sum()),1)
    entropy=float(-(p[p>0]*np.log(p[p>0])).sum())
    return dict(coverage=coverage,valid_fraction=float(valid.mean()),counts=counts.tolist(),fractions=fractions.tolist(),conditional_max_mass=float(p.max()),conditional_entropy=entropy,mean_nearest_distance=float(dist.min(1).mean()))

class Generator(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,2))
    def forward(self,z):return self.net(z)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def discriminator_step(g,d,opt,real,z):
    opt.zero_grad(set_to_none=True)
    loss=F.softplus(-d(real)).mean()+F.softplus(d(g(z).detach())).mean()
    loss.backward();opt.step();return float(loss.detach())

def generator_step(g,d,opt,z):
    opt.zero_grad(set_to_none=True)
    # Freeze parameters, not computation: d's input Jacobian must remain alive.
    for p in d.parameters():p.requires_grad_(False)
    loss=F.softplus(-d(g(z))).mean();loss.backward();opt.step()
    for p in d.parameters():p.requires_grad_(True)
    return float(loss.detach())

def train(seed,steps=3000,stress=False,output=None):
    require_positive('steps',steps)
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    x,labels=make_data();real=torch.from_numpy(x)
    g,d=Generator(),Discriminator()
    lr_g,lr_d,g_steps=(.002,.0001,5) if stress else (.0002,.0002,1)
    go=torch.optim.Adam(g.parameters(),lr=lr_g,betas=(.5,.999));do=torch.optim.Adam(d.parameters(),lr=lr_d,betas=(.5,.999))
    # The evaluation RNG never advances the training RNG.
    erng=torch.Generator().manual_seed(9000+seed);fixed_z=torch.randn(4096,2,generator=erng)
    history=[];snapshots={};saved_states={};start=time.monotonic()
    for step in range(steps+1):
        if step>0:
            idx=torch.randint(len(real),(128,));ld=discriminator_step(g,d,do,real[idx],torch.randn(128,2))
            for _ in range(g_steps):lg=generator_step(g,d,go,torch.randn(128,2))
        if step%100==0 or step==steps:
            with torch.no_grad():sample=g(fixed_z).numpy();pr=float(d(real[:1024]).sigmoid().mean());pf=float(d(g(fixed_z[:1024])).sigmoid().mean())
            row=dict(step=step,d_loss=None if step==0 else ld,g_loss=None if step==0 else lg,d_real_mean=pr,d_fake_mean=pf,**mode_metrics(sample))
            history.append(row);snapshots[str(step)]=sample.copy()
            if stress:saved_states[step]={k:v.detach().clone() for k,v in g.state_dict().items()}
    # A diagnostic snapshot is selected transparently, never fabricated. Whole history is saved.
    candidates=[r for r in history if r['step']>=min(500,steps//2) and r['valid_fraction']>=.5]
    selected=min(candidates,key=lambda r:(r['coverage'],-r['conditional_max_mass'])) if candidates else history[-1]
    report=dict(seed=seed,stress=stress,steps=steps,batch=128,lr_g=lr_g,lr_d=lr_d,g_steps_per_d=g_steps,seconds=round(time.monotonic()-start,3),final=history[-1],diagnostic=selected if stress else None,history=history)
    if output is not None:
        out=Path(output);out.mkdir(parents=True,exist_ok=True);name=('stress' if stress else 'baseline')+f'_seed{seed}'
        torch.save({'generator':g.state_dict(),'discriminator':d.state_dict(),'seed':seed,'steps':steps,'config':{k:report[k] for k in ['stress','lr_g','lr_d','g_steps_per_d']}},out/(name+'.pt'))
        if stress:torch.save({'generator':saved_states[selected['step']],'seed':seed,'step':selected['step'],'selection_rule':'checkpoints after min(500,steps//2) with valid_fraction >= 0.5; minimize coverage then maximize conditional_max_mass'},out/(name+'_diagnostic.pt'))
        np.savez_compressed(out/(name+'_samples.npz'),**snapshots)
    return report

def main(output=ROOT/'outputs',steps=3000,stress_steps=2200):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    x,labels=make_data();test,tl=make_data(4096,611);np.savez_compressed(ROOT/'data/mixture.npz',train=x,train_labels=labels,test=test,test_labels=tl,centers=centers())
    results={'unit':'061','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'data_seed_train':610,'data_seed_reference':611,'evaluation':{'radius':.35,'min_fraction_all_samples':.01,'sample_count':4096},'reference':mode_metrics(test),'runs':[]}
    for seed in [11,23,37]:
        r=train(seed,steps,False,out);results['runs'].append(r);print('baseline',seed,r['final'],flush=True)
    results['stress']=train(17,stress_steps,True,out)
    print('stress',results['stress']['diagnostic'],flush=True)
    # Negative control verifies the metric only; not claimed as a training failure.
    control=np.repeat(centers()[0:1],4096,axis=0)
    results['forced_control']=dict(kind='constant samples; deliberately constructed, NOT trained GAN collapse',**mode_metrics(control))
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=3000);p.add_argument('--stress-steps',type=int,default=2200);p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args();main(a.output,a.steps,a.stress_steps)
