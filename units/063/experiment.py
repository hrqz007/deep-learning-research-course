"""DL063: fair, bounded comparison of a 2-D VAE and GAN, plus metric audits.

Every network is trained here. Analytic controls are labelled as controls, never
as model outputs. Metric helpers use explicit exceptions (also active under -O).
"""
from pathlib import Path
import argparse,json,time,platform,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
SEEDS=(11,23,37)

def matrix(x,min_n=1):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)<min_n or x.shape[1]==0 or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty N by D matrix')
    return x

def centers():
    a=np.arange(8)*np.pi/4
    return np.column_stack((2*np.cos(a),2*np.sin(a))).astype('float32')

def make_data(n=2048,seed=630):
    if not isinstance(n,int) or isinstance(n,bool) or n<1:raise ValueError('n must be a positive integer')
    rng=np.random.default_rng(seed);y=rng.integers(8,size=n)
    return (centers()[y]+rng.normal(0,.08,(n,2))).astype('float32'),y

def mode_metrics(x,radius=.35,minimum=.01):
    x=matrix(x)
    if x.shape[1]!=2 or not 0<radius<.7 or not 0<minimum<1:raise ValueError('Invalid dimensions or thresholds')
    d=np.linalg.norm(x[:,None,:]-centers()[None,:,:],axis=2)
    labels=d.argmin(1);valid=d.min(1)<=radius
    counts=np.bincount(labels[valid],minlength=8);p=counts/max(1,counts.sum())
    return dict(valid_fraction=float(valid.mean()),coverage=int((counts/len(x)>=minimum).sum()),counts=counts.tolist(),entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))),distance_to_mode=float(d.min(1).mean()))

def features(x,kind='xy'):
    x=matrix(x)
    if x.shape[1]!=2:raise ValueError('Expected N by 2')
    if kind=='xy':return x
    if kind=='radius':return np.linalg.norm(x,axis=1,keepdims=True)
    if kind=='radial_angle':
        # Original fixed features: coordinates, radius, and fourfold angle.
        a=np.arctan2(x[:,1],x[:,0]);r=np.linalg.norm(x,axis=1)
        return np.column_stack((x,r,np.cos(4*a),np.sin(4*a)))
    raise ValueError('Unknown feature map')

def psd_sqrt(a):
    a=(a+a.T)/2;w,v=np.linalg.eigh(a)
    if w.min() < -1e-8:raise ValueError('Matrix is not positive semidefinite')
    return (v*np.sqrt(np.maximum(w,0)))@v.T

def gaussian_feature_distance(x,y,kind='xy'):
    """Squared 2-Wasserstein distance of fitted Gaussians; NOT Inception FID."""
    x=features(x,kind);y=features(y,kind)
    if len(x)<2 or len(y)<2:raise ValueError('At least two observations needed')
    mx,my=x.mean(0),y.mean(0)
    cx=np.atleast_2d(np.cov(x,rowvar=False,ddof=1));cy=np.atleast_2d(np.cov(y,rowvar=False,ddof=1))
    sx=psd_sqrt(cx);cross=psd_sqrt(sx@cy@sx)
    return float(max(0,np.sum((mx-my)**2)+np.trace(cx+cy-2*cross)))

def nearest_distance(query,reference,chunk=256):
    q=matrix(query);r=matrix(reference)
    if q.shape[1]!=r.shape[1] or not isinstance(chunk,int) or chunk<1:raise ValueError('Invalid nearest-neighbor inputs')
    return np.concatenate([np.sqrt(((q[i:i+chunk,None,:]-r[None,:,:])**2).sum(2).min(1)) for i in range(0,len(q),chunk)])

def audit(samples,train,holdout,tolerance=1e-7):
    if len(train)!=len(holdout):raise ValueError('Audit reference sets must have equal cardinality')
    dt=nearest_distance(samples,train);dh=nearest_distance(samples,holdout)
    return dict(train_median=float(np.median(dt)),holdout_median=float(np.median(dh)),train_closer_fraction=float(np.mean(dt<dh)),exact_train_fraction=float(np.mean(dt<=tolerance)),exact_holdout_fraction=float(np.mean(dh<=tolerance))),dict(train_distance=dt,holdout_distance=dh)

class VAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.mu=nn.Linear(64,2);self.lv=nn.Linear(64,2)
        self.decoder=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2))
    def encode(self,x):
        h=self.encoder(x);return self.mu(h),self.lv(h).clamp(-12,8)
    def forward(self,x,eps):
        mu,lv=self.encode(x);return self.decoder(mu+torch.exp(.5*lv)*eps),mu,lv

class Generator(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,2))
    def forward(self,x):return self.net(x)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def vae_terms(pred,x,mu,lv,sigma=.15):
    nll=(.5*((pred-x)/sigma)**2+math.log(sigma)+.5*math.log(2*math.pi)).sum(1)
    kl=.5*(mu.square()+lv.exp()-1-lv).sum(1)
    return nll,kl

def sample_model(model,family,seed,n=4096):
    gen=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        z=torch.randn(n,2,generator=gen)
        if family=='vae':
            means=model.decoder(z);sample=means+.15*torch.randn(n,2,generator=gen)
        elif family=='gan':sample=model(z);means=sample
        else:raise ValueError('Unknown family')
    return sample.numpy(),means.numpy()

def train(family,seed,real,steps=4000):
    if not isinstance(steps,int) or isinstance(steps,bool) or steps<1:raise ValueError('steps must be positive integer')
    torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+6300);x=torch.from_numpy(real)
    model=VAE() if family=='vae' else Generator() if family=='gan' else None
    if model is None:raise ValueError('Unknown family')
    lr=.001 if family=='vae' else .0002
    opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(.5,.999) if family=='gan' else (.9,.999))
    d=Discriminator() if family=='gan' else None
    dop=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999)) if d is not None else None
    trace=[];start=time.monotonic()
    for step in range(1,steps+1):
        batch=x[torch.randint(len(x),(128,),generator=rng)]
        if family=='vae':
            pred,mu,lv=model(batch,torch.randn(128,2,generator=rng));nll,kl=vae_terms(pred,batch,mu,lv)
            loss=(nll+kl).mean();opt.zero_grad();loss.backward();opt.step()
            row=dict(step=step,nll=float(nll.mean().detach()),kl=float(kl.mean().detach()),loss=float(loss.detach()))
        else:
            dop.zero_grad();fake=model(torch.randn(128,2,generator=rng)).detach()
            ld=F.softplus(-d(batch)).mean()+F.softplus(d(fake)).mean();ld.backward();dop.step()
            for p in d.parameters():p.requires_grad_(False)
            opt.zero_grad();loss=F.softplus(-d(model(torch.randn(128,2,generator=rng)))).mean();loss.backward();opt.step()
            for p in d.parameters():p.requires_grad_(True)
            row=dict(step=step,d_loss=float(ld.detach()),g_loss=float(loss.detach()))
        if step==1 or step%200==0 or step==steps:trace.append(row)
    model.eval();return model,d,dict(history=trace,seconds=round(time.monotonic()-start,3),parameters=sum(p.numel() for p in model.parameters()),auxiliary_parameters=0 if d is None else sum(p.numel() for p in d.parameters()))

def main(output=ROOT/'outputs',steps=4000):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    train_x,train_y=make_data(2048,630);test,test_y=make_data(2048,631)
    ref,_=make_data(4096,632);oracle,_=make_data(4096,633)
    np.savez_compressed(ROOT/'data/mixture.npz',train=train_x,train_labels=train_y,holdout=test,holdout_labels=test_y,reference=ref,oracle=oracle,centers=centers())
    protocol=dict(unit='063',seeds=list(SEEDS),steps=steps,batch=128,training_rows=2048,generated_rows=4096,reference_rows=4096,equalized='same real-data minibatch count and primary generative-model optimizer updates; GAN also updates D once per round',not_equalized='FLOPs, total optimizer calls, parameter counts and convergence',selection='Fixed final endpoint for every seed; no cherry-picked samples or test tuning',feature_maps=['xy','radius','radial_angle'],quality_radius=.35,mode_minimum_total_fraction=.01,vae_sigma=.15,vae_beta=1.,evaluation_seed_base=9300)
    result=dict(protocol=protocol,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,runs=[],controls={})
    for family in ['vae','gan']:
        for seed in SEEDS:
            model,d,info=train(family,seed,train_x,steps);sample,means=sample_model(model,family,9300+seed)
            mem,dist=audit(sample,train_x,test);key=f'{family}_seed{seed}'
            np.savez_compressed(out/(key+'_samples.npz'),samples=sample,means=means,**dist)
            checkpoint=dict(family=family,seed=seed,steps=steps,state_dict=model.state_dict(),sampling_seed=9300+seed)
            if d is not None:checkpoint['discriminator_state_dict']=d.state_dict()
            torch.save(checkpoint,out/(key+'.pt'))
            row=dict(family=family,seed=seed,**info,metrics=mode_metrics(sample),feature_distance={k:gaussian_feature_distance(sample,ref,k) for k in protocol['feature_maps']},audit=mem)
            result['runs'].append(row);print(family,seed,row['metrics'],flush=True)
    # Independent controls reveal what the indicators cannot identify.
    rng=np.random.default_rng(634)
    controls={'oracle':oracle,'copier':train_x[rng.integers(len(train_x),size=4096)],'one_mode':np.repeat(centers()[:1],4096,axis=0),'rotated_modes':np.column_stack((2*np.cos(np.arange(4096)%8*np.pi/4+np.pi/8),2*np.sin(np.arange(4096)%8*np.pi/4+np.pi/8))).astype('float32')}
    for key,x in controls.items():
        mem,dist=audit(x,train_x,test);result['controls'][key]=dict(metrics=mode_metrics(x),feature_distance={k:gaussian_feature_distance(x,ref,k) for k in protocol['feature_maps']},audit=mem)
        np.savez_compressed(out/(key+'_control.npz'),samples=x,**dist)
    # The finite-sample experiment uses independent oracle draws, not training.
    result['sample_size_audit']=[]
    for n in [32,128,512,2048]:
        vals=[]
        for rep in range(20):
            a,_=make_data(n,10000+rep);b,_=make_data(n,11000+rep)
            vals.append(gaussian_feature_distance(a,b))
        result['sample_size_audit'].append(dict(n=n,repetitions=20,mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)),values=vals))
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--steps',type=int,default=4000);a=p.parse_args();main(a.output,a.steps)
