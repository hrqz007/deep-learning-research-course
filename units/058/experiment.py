"""DL058: reconstruction is not a semantic guarantee.

Original8-D data contain two high-variance nuisances and a low-variance label signal.
All representations are trained without labels; linear probes use128 fixed labels.
"""
from pathlib import Path
import argparse,json,platform,time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent

def write_json(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def make_data():
    rng=np.random.default_rng(5800);q,_=np.linalg.qr(rng.normal(size=(8,8)));data={}
    for split,n in [('train',640),('validation',160),('test',320)]:
        y=np.tile(np.array([-1.,1.]),n//2);rng.shuffle(y)
        factors=rng.normal(size=(n,8));factors[:,0]*=3;factors[:,1]*=2
        factors[:,2]=.5*y+.08*rng.normal(size=n);factors[:,3:]*=.10
        data['x_'+split]=(factors@q.T).astype(np.float32);data['y_'+split]=y.astype(np.float32)
    data['rotation']=q;return data

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,2))
        self.decoder=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,8))
    def forward(self,x):return self.decoder(self.encoder(x))

def reconstruction_loss(pred,target):
    if pred.shape!=target.shape or pred.ndim!=2 or pred.numel()==0:raise ValueError('matching nonempty N,D matrices required')
    return ((pred-target)**2).mean() # Mean over samples AND8 coordinates.

def sparse_penalty(z,coefficient):
    if not np.isfinite(coefficient) or coefficient<0:raise ValueError('nonnegative finite penalty')
    return coefficient*z.abs().mean()

def fit_probe(train_z,train_y,test_z,ridge=1.):
    if len(train_z)!=len(train_y) or train_z.ndim!=2 or test_z.shape[1]!=train_z.shape[1]:raise ValueError('probe shape mismatch')
    if not np.isfinite(ridge) or ridge<=0:raise ValueError('positive finite ridge')
    # Fit both normalization and weights using the128 labeled training examples only.
    mu=train_z.mean(0);scale=train_z.std(0);scale=np.where(scale<1e-8,1.,scale)
    a=np.column_stack([(train_z-mu)/scale,np.ones(len(train_z))]);b=np.column_stack([(test_z-mu)/scale,np.ones(len(test_z))])
    penalty=np.eye(a.shape[1])*ridge;penalty[-1,-1]=0 # Do not penalize intercept.
    weights=np.linalg.solve(a.T@a+penalty,a.T@train_y);scores=b@weights
    return scores,{'mean':mu,'scale':scale,'weights':weights}

def run(output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);d=make_data();np.savez(out/'data.npz',**d)
    proto={'seeds':[5801,5802,5803],'steps':250,'batch':64,'lr':.005,'bottleneck':2,'architecture':'8->16 ReLU->2->16 ReLU->8',
       'noise_std':.5,'split_sizes':{'train':640,'validation':160,'test':320},'labeled_probe_train':128,'ridge':1.,
       'normalization':'subtract full training input mean only for AE/PCA; probe feature mean/std fit on128 labeled training examples',
       'selection':'fixed250 steps; no checkpoint search, no tuning with test or validation labels',
       'max_train_seconds_per_run':120,'metrics':'MSE averaged over all N*8 coordinates; probe accuracy on320 test examples',
       'scope':'3 initializations, one fixed data split; no significance or universal representation claim'}
    write_json(out/'protocol.json',proto);mean=d['x_train'].mean(0);train=d['x_train']-mean;test=d['x_test']-mean;val=d['x_validation']-mean
    tx=torch.from_numpy(train);vx=torch.from_numpy(val);sx=torch.from_numpy(test)
    fixed_noise=np.random.default_rng(5891).normal(0,.5,size=test.shape).astype(np.float32);noisy=sx+torch.from_numpy(fixed_noise)
    _,singular,vt=np.linalg.svd(train,full_matrices=False);components=vt[:2].T
    np.savez(out/'pca.npz',mean=mean,components=components,singular_values=singular)
    rows=[];latents={};recons={};started=time.perf_counter()
    def add(name,seed,train_z,test_z,recon,noisy_recon,trace=None):
        scores,probe=fit_probe(train_z[:128].astype(float),d['y_train'][:128].astype(float),test_z.astype(float))
        accuracy=float(np.mean(np.where(scores>=0,1.,-1.)==d['y_test']))
        rows.append({'representation':name,'seed':seed,'test_reconstruction_mse':float(np.mean((recon-test)**2)),
          'corrupted_test_denoising_mse':float(np.mean((noisy_recon-test)**2)),
          'probe_accuracy':accuracy,'probe_correct':int(round(accuracy*len(test))),'probe_n':len(test),'trace':trace or []})
        key=f'{name}_{seed}';latents[key]=test_z;recons[key]=recon
        np.savez(out/f'probe_{key}.npz',**probe,scores=scores)
    add('raw',0,train,test,test,noisy.numpy())
    add('pca2',0,train@components,test@components,test@components@components.T,noisy.numpy()@components@components.T)
    for seed in proto['seeds']:
        torch.manual_seed(seed);random_model=Autoencoder().eval()
        with torch.no_grad():
            add('random2',seed,random_model.encoder(tx).numpy(),random_model.encoder(sx).numpy(),random_model(sx).numpy(),random_model(noisy).numpy())
        for name,sigma in [('ae',0.),('dae',.5)]:
            torch.manual_seed(seed);model=Autoencoder();optimizer=torch.optim.Adam(model.parameters(),lr=.005)
            order=torch.Generator().manual_seed(5890);noise_rng=torch.Generator().manual_seed(5892);trace=[];begin=time.perf_counter()
            for step in range(1,251):
                ids=torch.randperm(len(tx),generator=order)[:64];clean=tx[ids]
                # DAE target stays clean; replacing this target by corrupted input changes the question.
                corrupted=clean+sigma*torch.randn(clean.shape,generator=noise_rng)
                optimizer.zero_grad(set_to_none=True);loss=reconstruction_loss(model(corrupted),clean);loss.backward();optimizer.step()
                if step in [1,50,100,150,200,250]:
                    with torch.no_grad():trace.append({'step':step,'minibatch_loss':loss.item(),'clean_validation_mse':reconstruction_loss(model(vx),vx).item()})
                if time.perf_counter()-begin>120:raise RuntimeError('training time budget exceeded')
            model.eval()
            with torch.no_grad():add(name,seed,model.encoder(tx).numpy(),model.encoder(sx).numpy(),model(sx).numpy(),model(noisy).numpy(),trace)
            np.savez(out/f'weights_{name}_{seed}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()},input_mean=mean)
    np.savez(out/'plot_data.npz',**{'latent_'+k:v for k,v in latents.items()},**{'recon_'+k:v for k,v in recons.items()},target=test,labels=d['y_test'])
    result={'protocol':proto,'results':rows,'pca_train_eigenvalues':(singular**2/len(train)).tolist(),
       'pca_train_optimum_mse':float(np.sum(singular[2:]**2)/(len(train)*8)),
       'identity_reconstruction_mse':0.,'identity_bottleneck_dimension':8,
       'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':1,'device':'cpu','wall_seconds':time.perf_counter()-started},
       'limits':'labels describe a synthetic low-variance factor; common test set and one data split; random-encoder reconstruction is untrained baseline'}
    write_json(out/'results.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');args=p.parse_args();r=run(args.output)
    for row in r['results']:print({k:v for k,v in row.items() if k!='trace'})
