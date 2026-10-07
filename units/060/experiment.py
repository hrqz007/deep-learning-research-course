"""DL060: Gaussian 2D and Bernoulli 8x8 VAEs, explicit units and reductions."""
from pathlib import Path
import argparse,json,time,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
SEEDS=(6001,6002)
CONFIGS=('standard','high_beta','restore_beta')
STEPS=600

def make_data():
    rng=np.random.default_rng(6000);d={}
    for split,n in [('train',768),('validation',192),('test',384)]:
        a=rng.uniform(0,2*np.pi,n);xy=1.7*np.stack([np.cos(a),np.sin(a)],1)+rng.normal(0,.12,(n,2))
        # Bernoulli observations: a random horizontal or vertical bright bar.
        image=np.full((n,8,8),.03);axis=rng.integers(0,2,n);pos=rng.integers(1,7,n)
        for i in range(n):
            if axis[i]==0:image[i,pos[i],:]=.95
            else:image[i,:,pos[i]]=.95
        d['xy_'+split]=xy.astype('float32');d['image_'+split]=(rng.random(image.shape)<image).reshape(n,64).astype('float32')
    return d
class VAE(nn.Module):
    def __init__(self,dim):
        super().__init__();self.enc=nn.Sequential(nn.Linear(dim,32),nn.Tanh());self.mu=nn.Linear(32,2);self.logvar=nn.Linear(32,2);self.dec=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,dim))
    def encode(self,x):
        h=self.enc(x);return self.mu(h),self.logvar(h).clamp(-12,12)
    def forward(self,x,eps):
        mu,lv=self.encode(x);return self.dec(reparameterize(mu,lv,eps)),mu,lv

def matching(*xs):
    if any(x.ndim!=2 or x.numel()==0 or x.shape!=xs[0].shape or not torch.isfinite(x).all() for x in xs):raise ValueError('Matching finite nonempty matrices required')
def reparameterize(mu,logvar,eps):
    matching(mu,logvar,eps);return mu+torch.exp(.5*logvar)*eps

def kl_standard(mu,logvar):
    matching(mu,logvar);return .5*(mu.square()+logvar.exp()-1-logvar).sum(1)

def reconstruction_nll(pred,x,kind,sigma=.25):
    matching(pred,x)
    if kind=='xy':
        if not math.isfinite(sigma) or sigma<=0:raise ValueError('sigma must be positive finite')
        return (.5*((x-pred)/sigma).square()+math.log(sigma)+.5*math.log(2*math.pi)).sum(1)
    if kind=='image':
        if bool(((x<0)|(x>1)).any()):raise ValueError('Bernoulli targets must lie in [0,1]')
        return F.binary_cross_entropy_with_logits(pred,x,reduction='none').sum(1)
    raise ValueError('Unknown observation model')

def save_model(path,m):np.savez(path,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
def load_model(path,dim):
    m=VAE(dim);s=np.load(path);m.load_state_dict({k:torch.from_numpy(s[k]) for k in s.files});m.eval();return m

def evaluate(m,x,kind,seed=6099):
    """16 shared reproducible MC draws for E_q NLL; exact analytic KL."""
    g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mu,lv=m.encode(x);nll=[]
        for _ in range(16):
            eps=torch.randn(mu.shape,generator=g);nll.append(reconstruction_nll(m.dec(reparameterize(mu,lv,eps)),x,kind))
        nll=torch.stack(nll).mean(0);kl=kl_standard(mu,lv)
        perm=torch.arange(len(x)-1,-1,-1);det=m.dec(mu);shuffled=m.dec(mu[perm])
        shuffle_delta=(reconstruction_nll(shuffled,x,kind)-reconstruction_nll(det,x,kind)).mean().item()
        return dict(mc_nll=float(nll.mean()),kl=float(kl.mean()),negative_elbo=float((nll+kl).mean()),mu_variance=float(mu.var(0,unbiased=False).mean()),shuffle_nll_delta=shuffle_delta),dict(mu=mu.numpy(),logvar=lv.numpy(),nll=nll.numpy(),kl=kl.numpy(),mean_decoding=det.numpy())

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
    d=make_data();np.savez(out/'data.npz',**d);rows=[];plot={}
    for kind,dim in [('xy',2),('image',64)]:
        train=torch.from_numpy(d[kind+'_train']);test=torch.from_numpy(d[kind+'_test'])
        for seed in SEEDS:
            for config in CONFIGS:
                torch.manual_seed(seed);m=VAE(dim);g=torch.Generator().manual_seed(seed+100);opt=torch.optim.Adam(m.parameters(),lr=.003);trace=[]
                # Standard: beta=1 for 600. High beta:100 for 600.
                # Intervention: same high-beta prefix, then only beta changes to1.
                for step in range(STEPS):
                    beta=1. if config=='standard' or (config=='restore_beta' and step>=300) else 100.
                    idx=torch.randint(len(train),(64,),generator=g);x=train[idx];eps=torch.randn((len(x),2),generator=g)
                    pred,mu,lv=m(x,eps);nll=reconstruction_nll(pred,x,kind);kl=kl_standard(mu,lv);loss=(nll+beta*kl).mean()
                    opt.zero_grad();loss.backward();opt.step()
                    if step%10==0:trace.append([step,float(nll.mean().detach()),float(kl.mean().detach()),beta])
                m.eval();key=f'{kind}_{config}_{seed}';metrics,arrays=evaluate(m,test,kind);save_model(out/f'weights_{key}.npz',m)
                # Prior samples include observation noise, as distinct from decoder means.
                with torch.no_grad():
                    gen=torch.Generator().manual_seed(seed+700);z=torch.randn((128,2),generator=gen);raw=m.dec(z)
                    if kind=='xy':sample=raw+.25*torch.randn(raw.shape,generator=gen);mean=raw
                    else:mean=raw.sigmoid();sample=torch.bernoulli(mean,generator=gen)
                for k,v in arrays.items():plot[key+'_'+k]=v
                plot[key+'_samples']=sample.numpy();plot[key+'_sample_means']=mean.numpy();plot[key+'_trace']=np.array(trace)
                rows.append(dict(kind=kind,config=config,seed=seed,**metrics))
    np.savez(out/'plot_data.npz',**plot)
    protocol=dict(unit='060',seeds=list(SEEDS),steps=STEPS,batch=64,latent_dim=2,gaussian_sigma=.25,reduction='sum observation and latent coordinates, then mean batch; natural log nats/example',mc_eval_draws=16,mc_eval_seed=6099,configs={'standard':'beta1 all600','high_beta':'beta100 all600','restore_beta':'beta100 first300 then beta1 last300'},selection='All predeclared endpoints, no test selection',device='cpu',threads=1)
    result=dict(protocol=protocol,results=rows,runtime=dict(seconds=time.monotonic()-start,torch=torch.__version__,numpy=np.__version__))
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs');a=p.parse_args();r=run(a.out)
    for row in r['results']:print(row)
