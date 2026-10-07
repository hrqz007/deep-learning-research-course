"""DL059: contrastive views and a genuinely frozen linear probe, CPU/offline."""
from pathlib import Path
import argparse,json,time,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
SEEDS=(5901,5902,5903)
METHODS=('random','contrastive_good','contrastive_bad','supervised')
STEPS=350

def make_data():
    """Labels are quadrants; four large nuisance coordinates are independent."""
    rng=np.random.default_rng(5900);data={}
    for split,n in [('train',768),('validation',192),('test',384)]:
        y=np.tile(np.arange(4),n//4);rng.shuffle(y)
        a=(y+rng.uniform(.06,.94,n))*np.pi/2
        signal=np.stack([np.cos(a),np.sin(a),np.cos(2*a),np.sin(2*a)],1)
        data['x_'+split]=np.concatenate([signal,2*rng.normal(size=(n,4))],1).astype('float32')
        data['y_'+split]=y.astype('int64')
    return data

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):return self.net(x)

def augment(x,kind,generator):
    if x.ndim!=2 or x.shape[1]!=8 or not len(x):raise ValueError('Expected nonempty B by 8')
    noise=lambda shape:torch.randn(shape,generator=generator,dtype=x.dtype)
    v=x.clone()
    if kind=='good':
        v[:,:4]+= .04*noise(v[:,:4].shape);v[:,4:]=2*noise(v[:,4:].shape)
    elif kind=='bad':
        a=torch.rand((len(x),),generator=generator,dtype=x.dtype)*2*math.pi
        v[:,:4]=torch.stack([a.cos(),a.sin(),(2*a).cos(),(2*a).sin()],1)
        v[:,4:]+= .04*noise(v[:,4:].shape)
    else:raise ValueError('Unknown view policy')
    return v

def info_nce(a,b,temperature=.2):
    """2B anchors; self excluded, positive retained in denominator; stable CE."""
    if a.ndim!=2 or a.shape!=b.shape or len(a)<2 or a.shape[1]<1:raise ValueError('Matching B by D with B>=2 required')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive finite')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite embeddings required')
    z=F.normalize(torch.cat([a,b]),dim=1);n=len(a)
    logits=z@z.T/temperature
    logits=logits.masked_fill(torch.eye(2*n,dtype=torch.bool),float('-inf'))
    target=(torch.arange(2*n)+n)%(2*n)
    return F.cross_entropy(logits,target)

def fit_probe(z,y,ridge=1.):
    """Train-only standardization and closed-form multiclass ridge; no encoder."""
    if z.ndim!=2 or len(z)!=len(y) or not len(z) or ridge<=0:raise ValueError('Invalid probe inputs')
    mean=z.mean(0);scale=z.std(0);scale=np.where(scale>1e-8,scale,1.)
    a=np.c_[(z-mean)/scale,np.ones(len(z))];target=np.eye(4)[y]
    penalty=np.diag([ridge]*z.shape[1]+[0.])
    w=np.linalg.solve(a.T@a+penalty,a.T@target)
    return dict(mean=mean,scale=scale,weight=w)

def probe_scores(z,p):return np.c_[(z-p['mean'])/p['scale'],np.ones(len(z))]@p['weight']
def save_model(path,model):np.savez(path,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
def load_model(path):
    model=Encoder();s=np.load(path);model.load_state_dict({k:torch.from_numpy(s[k]) for k in s.files});model.eval();return model

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.monotonic();data=make_data();np.savez(out/'data.npz',**data);x=torch.from_numpy(data['x_train']);rows=[];plots={}
    for seed in SEEDS:
        for method in METHODS:
            torch.manual_seed(seed);enc=Encoder();head=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,8)) if method.startswith('contrastive') else nn.Linear(2,4)
            opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=.003)
            g=torch.Generator().manual_seed(seed+100);losses=[]
            for step in range(0 if method=='random' else STEPS):
                idx=torch.randint(len(x),(64,),generator=g);batch=x[idx];opt.zero_grad()
                if method.startswith('contrastive'):
                    kind=method.split('_')[-1];loss=info_nce(head(enc(augment(batch,kind,g))),head(enc(augment(batch,kind,g))))
                else:loss=F.cross_entropy(head(enc(batch)),torch.from_numpy(data['y_train'])[idx])
                loss.backward();opt.step();losses.append(loss.item())
            enc.eval()
            before={k:v.detach().clone() for k,v in enc.state_dict().items()}
            with torch.no_grad():emb={s:enc(torch.from_numpy(data['x_'+s])).numpy() for s in ('train','validation','test')}
            p=fit_probe(emb['train'][:128],data['y_train'][:128])
            if any(not torch.equal(v,enc.state_dict()[k]) for k,v in before.items()):raise RuntimeError('Probe changed encoder')
            score=probe_scores(emb['test'],p);key=f'{method}_{seed}'
            save_model(out/f'weights_{key}.npz',enc);np.savez(out/f'probe_{key}.npz',**p)
            plots[key+'_embedding']=emb['test'];plots[key+'_loss']=np.array(losses);plots[key+'_scores']=score
            rows.append(dict(method=method,seed=seed,test_accuracy=float((score.argmax(1)==data['y_test']).mean()),validation_accuracy=float((probe_scores(emb['validation'],p).argmax(1)==data['y_validation']).mean()),mean_embedding_std=float(emb['test'].std(0).mean()),train_steps=len(losses),frozen_probe=True))
    np.savez(out/'plot_data.npz',**plots)
    protocol=dict(unit='059',steps=STEPS,batch_originals=64,anchors=128,temperature=.2,probe_labeled_prefix=128,probe_ridge=1.,probe_features=2,pretrain_labeled_count={'random':0,'contrastive_good':0,'contrastive_bad':0,'supervised':768},seeds=list(SEEDS),selection='No validation or test model selection; all fixed endpoints reported',data='Original synthetic independent splits; quadrants are the designated target',device='cpu',threads=1)
    result=dict(protocol=protocol,results=rows,runtime=dict(seconds=time.monotonic()-start,torch=torch.__version__,numpy=np.__version__))
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs');a=p.parse_args();r=run(a.out)
    for row in r['results']:print(row)
