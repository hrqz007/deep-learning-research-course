"""DL064: a falsifiable KL-warmup research project on original 8x8 images.
Same initialization and minibatch/noise streams pair the two equal-budget arms.
A doubled-budget constant-beta baseline tests the importance of more updates.
"""
from pathlib import Path
import argparse,json,math,platform,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
SEEDS=(11,23,37,53,71)
ARMS=('constant','warmup','long_constant')

def positive(name,x):
    if not isinstance(x,int) or isinstance(x,bool) or x<1:raise ValueError(name+' must be positive integer')

def templates():
    t=np.zeros((12,8,8),np.float32)
    for i in range(6):t[i,i+1,:]=1;t[6+i,:,i+1]=1
    return t.reshape(12,64)

def make_data():
    d={}
    for name,n,seed in [('train',1024,640),('validation',256,641),('test',512,642)]:
        rng=np.random.default_rng(seed);y=rng.integers(12,size=n);p=.03+.92*templates()[y]
        d[name]=(rng.random(p.shape)<p).astype('float32');d[name+'_labels']=y
    return d

class VAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(64,48),nn.Tanh());self.mu=nn.Linear(48,4);self.lv=nn.Linear(48,4)
        self.decoder=nn.Sequential(nn.Linear(4,48),nn.Tanh(),nn.Linear(48,64))
    def encode(self,x):
        h=self.encoder(x);return self.mu(h),self.lv(h).clamp(-12,8)
    def forward(self,x,epsilon):
        mu,lv=self.encode(x);return self.decoder(mu+torch.exp(.5*lv)*epsilon),mu,lv

def terms(logits,x,mu,lv):
    if logits.shape!=x.shape or logits.ndim!=2 or mu.shape!=lv.shape or len(x)!=len(mu):raise ValueError('Invalid objective shapes')
    if not torch.isfinite(x).all() or ((x<0)|(x>1)).any():raise ValueError('Bernoulli targets outside [0,1]')
    nll=F.binary_cross_entropy_with_logits(logits,x,reduction='none').sum(1)
    kl=.5*(mu.square()+lv.exp()-1-lv).sum(1)
    return nll,kl

def beta_at(step,arm,warmup=500):
    positive('step',step);positive('warmup',warmup)
    if arm not in ARMS:raise ValueError('Unknown arm')
    return min(1.,step/warmup) if arm=='warmup' else 1.

def evaluate(model,x,draws=32,seed=6499):
    positive('draws',draws);gen=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mu,lv=model.encode(x);nlls=[]
        for _ in range(draws):
            eps=torch.randn(mu.shape,generator=gen);logits=model.decoder(mu+torch.exp(.5*lv)*eps);n,k=terms(logits,x,mu,lv);nlls.append(n)
        nll=torch.stack(nlls).mean(0);kl=.5*(mu.square()+lv.exp()-1-lv).sum(1);mean=model.decoder(mu).sigmoid()
        # Permutation is a diagnostic of input information, not an ELBO.
        perm=model.decoder(mu.flip(0)).sigmoid();mse=(mean-x).square().mean();shuffle=(perm-x).square().mean()
        row=dict(nll=float(nll.mean()),kl=float(kl.mean()),negative_elbo=float((nll+kl).mean()),mean_decode_mse=float(mse),shuffle_mse_delta=float(shuffle-mse),active_dimensions=int((mu.var(0,unbiased=False)>.01).sum()),mu_variances=mu.var(0,unbiased=False).tolist())
    return row,dict(nll=nll.numpy(),kl=kl.numpy(),negative_elbo=(nll+kl).numpy(),mu=mu.numpy(),logvar=lv.numpy(),mean_reconstruction=mean.numpy())

def sample_model(model,seed,n=1024):
    positive('n',n);g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        p=model.decoder(torch.randn(n,4,generator=g)).sigmoid();x=torch.bernoulli(p,generator=g)
    return x.numpy(),p.numpy()

def sample_metrics(samples,tolerance=4):
    x=np.asarray(samples)
    if x.ndim!=2 or x.shape[1]!=64 or len(x)==0 or not np.isfinite(x).all() or not np.isin(x,[0,1]).all():raise ValueError('Expected nonempty binary N by 64 images')
    dist=np.abs(x[:,None,:]-templates()[None,:,:]).sum(2);labels=dist.argmin(1);valid=dist.min(1)<=tolerance;counts=np.bincount(labels[valid],minlength=12)
    return dict(template_valid_fraction=float(valid.mean()),coverage=int((counts/len(x)>=.01).sum()),mean_nearest_hamming=float(dist.min(1).mean()),counts=counts.tolist())

def train(arm,seed,train_x,steps=2000,warmup=500):
    positive('steps',steps);positive('warmup',warmup)
    if arm not in ARMS:raise ValueError('Unknown arm')
    torch.manual_seed(seed);g=torch.Generator().manual_seed(6400+seed);model=VAE();opt=torch.optim.Adam(model.parameters(),lr=.003);trace=[]
    total=2*steps if arm=='long_constant' else steps;start=time.monotonic()
    for step in range(1,total+1):
        x=train_x[torch.randint(len(train_x),(64,),generator=g)];eps=torch.randn(len(x),4,generator=g)
        logits,mu,lv=model(x,eps);nll,kl=terms(logits,x,mu,lv);beta=beta_at(step,arm,warmup);loss=(nll+beta*kl).mean()
        opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%100==0 or step==total:trace.append(dict(step=step,beta=beta,nll=float(nll.mean().detach()),kl=float(kl.mean().detach()),training_objective=float(loss.detach())))
    model.eval();return model,dict(steps=total,history=trace,seconds=round(time.monotonic()-start,3))

def main(output=ROOT/'outputs',steps=2000,warmup=500):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out=Path(output);out.mkdir(parents=True,exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    data=make_data();np.savez_compressed(ROOT/'data/bars.npz',**data,templates=templates())
    protocol=dict(unit='064',seeds=list(SEEDS),arms=list(ARMS),base_steps=steps,long_steps=2*steps,warmup_steps=warmup,batch=64,learning_rate=.003,latent_dimensions=4,train_rows=1024,validation_rows=256,test_rows=512,mc_draws=32,evaluation_seed=6499,sample_count=1024,hypothesis='Warmup minus constant mean held-out negative ELBO <= -0.2 nats/example, at least 4/5 paired wins, and mean generated template-valid drop <= 0.02',selection='Fixed endpoint; no test-based selection; validation is a diagnostic, not a tuning loop',paired='same architecture, initialization, training minibatch and epsilon streams in equal-budget arms',limitations='one fixed data split; seeds quantify optimization variation, not new-dataset uncertainty')
    result=dict(protocol=protocol,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,runs=[])
    # Strong non-neural baseline: independent empirical Bernoulli pixel marginals.
    p=np.clip(data['train'].mean(0),1e-5,1-1e-5);test=data['test']
    result['independent_pixel_baseline_nll']=float(-(test*np.log(p)+(1-test)*np.log1p(-p)).sum(1).mean())
    result['reference_sample_metrics']=sample_metrics(test)
    for arm in ARMS:
        for seed in SEEDS:
            model,info=train(arm,seed,torch.from_numpy(data['train']),steps,warmup)
            validation,_=evaluate(model,torch.from_numpy(data['validation']));metrics,arr=evaluate(model,torch.from_numpy(data['test']))
            sample,prob=sample_model(model,9400+seed);sm=sample_metrics(sample);key=f'{arm}_seed{seed}'
            # Rank failures by held-out per-example negative ELBO, preserve full array.
            fail=np.argsort(arr['negative_elbo'])[-12:][::-1]
            np.savez_compressed(out/(key+'_arrays.npz'),**arr,samples=sample,sample_probabilities=prob,failure_indices=fail)
            torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=seed,steps=info['steps'],sampling_seed=9400+seed),out/(key+'.pt'))
            row=dict(arm=arm,seed=seed,**info,validation=validation,test=metrics,sample=sm,failure_indices=fail.tolist());result['runs'].append(row)
            print(arm,seed,metrics,sm,flush=True)
    lookup={(r['arm'],r['seed']):r for r in result['runs']};pairs=[]
    for seed in SEEDS:
        c,w,l=[lookup[(a,seed)] for a in ARMS]
        pairs.append(dict(seed=seed,warmup_minus_constant=w['test']['negative_elbo']-c['test']['negative_elbo'],long_minus_constant=l['test']['negative_elbo']-c['test']['negative_elbo'],warmup_valid_minus_constant=w['sample']['template_valid_fraction']-c['sample']['template_valid_fraction']))
    delta=np.array([p['warmup_minus_constant'] for p in pairs]);quality=np.array([p['warmup_valid_minus_constant'] for p in pairs]);se=delta.std(ddof=1)/np.sqrt(len(delta))
    result['paired']=pairs;result['decision']=dict(mean_delta=float(delta.mean()),std_delta=float(delta.std(ddof=1)),wins=int((delta<0).sum()),mean_valid_delta=float(quality.mean()),t_interval_95=[float(delta.mean()-2.776*se),float(delta.mean()+2.776*se)],supports_predeclared_hypothesis=bool(delta.mean()<=-.2 and (delta<0).sum()>=4 and quality.mean()>=-.02),interval_note='approximate t interval over five training seeds, df=4, fixed split; not proof of generalization')
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');(out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--steps',type=int,default=2000);p.add_argument('--warmup',type=int,default=500);a=p.parse_args();main(a.output,a.steps,a.warmup)
