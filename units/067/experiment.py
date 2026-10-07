"""DL067: single-process numerical simulation of data-parallel semantics.
No process group/network is initialized. Never interpret runtime as cluster speed.
"""
from pathlib import Path
import argparse,copy,json,platform
import numpy as np
import torch
from torch import nn
from torch.utils.data import DistributedSampler


def data(n=128,seed=67):
    g=torch.Generator().manual_seed(seed);x=torch.randn(n,5,generator=g,dtype=torch.float64)
    y=(x[:,0]+.7*x[:,1]-.4*x[:,2]+.3*x[:,3]*x[:,4]>0).long();return x,y

def model(seed=11):
    torch.manual_seed(seed);return nn.Sequential(nn.Linear(5,12),nn.Tanh(),nn.Linear(12,2)).double()

def gradient(m,x,y,mask=None):
    m.zero_grad(set_to_none=True)
    losses=nn.functional.cross_entropy(m(x),y,reduction='none')
    if mask is None:mask=torch.ones_like(losses)
    if mask.shape!=losses.shape or not torch.isfinite(mask).all() or bool((mask<0).any()):raise ValueError('mask must be finite nonnegative and match examples')
    n=float(mask.sum());summed=(losses*mask).sum()
    # An empty local rank may still need participate; its contribution is zero.
    loss=summed/n if n>0 else summed*0
    loss.backward()
    return [p.grad.detach().clone() for p in m.parameters()],n,float(loss.detach())

def flatten(gs):return torch.cat([g.flatten() for g in gs])

def average_gradients(grads,counts,mode='weighted'):
    if not grads or len(grads)!=len(counts) or any(c<0 for c in counts) or sum(counts)<=0:raise ValueError('valid positive global count required')
    if mode not in ['weighted','rank_mean','double_divide']:raise ValueError('unknown reduction')
    world=len(grads)
    weights=[c/sum(counts) for c in counts] if mode=='weighted' else [1/world]*world
    if mode=='double_divide':weights=[w/world for w in weights]
    return [sum(w*g[j] for w,g in zip(weights,grads)) for j in range(len(grads[0]))]

def simulate_step(m,opt,x,y,parts,mode='weighted',masks=None):
    """Every rank starts at the same pre-step parameters; exactly one optimizer step."""
    grads=[];counts=[];local_losses=[]
    for rank,idx in enumerate(parts):
        replica=copy.deepcopy(m)
        g,n,l=gradient(replica,x[idx],y[idx],None if masks is None else masks[rank]);grads.append(g);counts.append(n);local_losses.append(l)
    averaged=average_gradients(grads,counts,mode)
    opt.zero_grad(set_to_none=True)
    for p,g in zip(m.parameters(),averaged):p.grad=g
    opt.step();return {'counts':counts,'local_means':local_losses,'global_weighted_loss':sum(n*l for n,l in zip(counts,local_losses))/sum(counts)}

def sampler_indices(n=10,world=3,drop_last=False,epoch=0,shuffle=False):
    if n<1 or world<1:raise ValueError('positive n/world required')
    result=[]
    for rank in range(world):
        s=DistributedSampler(list(range(n)),num_replicas=world,rank=rank,shuffle=shuffle,seed=67,drop_last=drop_last)
        s.set_epoch(epoch);result.append(list(s))
    return result

def ring_model(payload_bytes,world,latency_s,bandwidth_bytes_s):
    """Ideal non-overlapped ring estimate, not measured communication."""
    if world<1 or payload_bytes<0 or latency_s<0 or bandwidth_bytes_s<=0:raise ValueError('invalid communication inputs')
    volume=2*(world-1)/world*payload_bytes
    return {'per_rank_transfer_bytes':volume,'seconds':2*(world-1)*latency_s+volume/bandwidth_bytes_s,
            'scope':'ideal ring model, no overlap; illustrative assumptions only'}

def main(output=Path('outputs')):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    x,y=data();parts=[torch.arange(0,20),torch.arange(20,50),torch.arange(50,128)]
    base=model();full,_,_=gradient(base,x,y);locals_=[gradient(copy.deepcopy(base),x[p],y[p]) for p in parts]
    grad_checks={}
    for mode in ['weighted','rank_mean','double_divide']:
        v=flatten(average_gradients([z[0] for z in locals_],[z[1] for z in locals_],mode));ref=flatten(full)
        grad_checks[mode]={'max_abs_error':float((v-ref).abs().max()),'relative_error':float(torch.linalg.vector_norm(v-ref)/torch.linalg.vector_norm(ref))}
    # Token/valid-example masks have denominators unequal even for equal local rows.
    eq=[torch.arange(0,64),torch.arange(64,128)];masks=[torch.ones(64,dtype=torch.float64),torch.cat([torch.ones(7),torch.zeros(57)]).double()]
    mg=[gradient(copy.deepcopy(base),x[p],y[p],mask) for p,mask in zip(eq,masks)]
    masked_full=gradient(copy.deepcopy(base),x,y,torch.cat(masks))[0]
    masked={k:float((flatten(average_gradients([z[0] for z in mg],[z[1] for z in mg],k))-flatten(masked_full)).abs().max()) for k in ['weighted','rank_mean']}
    rows=[];vx,vy=data(512,6701)
    for seed in [11,23,37]:
        models={k:model(seed) for k in ['full','weighted','rank_mean','duplicated_rank']};opts={k:torch.optim.SGD(m.parameters(),lr=.15,momentum=.8) for k,m in models.items()};history=[]
        for t in range(80):
            for mode,m in models.items():
                if mode=='full':
                    g,_,_=gradient(m,x,y);opts[mode].step()
                else:
                    chosen=[parts[0]]*3 if mode=='duplicated_rank' else parts
                    simulate_step(m,opts[mode],x,y,chosen,'rank_mean' if mode=='rank_mean' else 'weighted')
            if t%10==0 or t==79:
                row={'step':t+1}
                with torch.no_grad():
                    for k,m in models.items():row[k]={'test_accuracy':float((m(vx).argmax(1)==vy).double().mean()),'test_loss':float(nn.functional.cross_entropy(m(vx),vy))}
                row['full_weighted_parameter_max_error']=max(float((a-b).abs().max().detach()) for a,b in zip(models['full'].parameters(),models['weighted'].parameters()));history.append(row)
        for k,m in models.items():torch.save({'state_dict':m.state_dict(),'seed':seed,'mode':k,'steps':80},output/f'{k}_seed{seed}.pt')
        rows.append({'seed':seed,'history':history})
    indices={str(drop):sampler_indices(drop_last=drop) for drop in [False,True]}
    r={'unit':'067','runtime':{'python':platform.python_version(),'torch':torch.__version__,'dtype':'float64','device':'cpu','threads':1},
       'execution_kind':'single-process mathematical simulation; no real ranks or collective communication',
       'gradient_checks':grad_checks,'masked_gradient_max_errors':masked,'partition_sizes':[20,30,78],
       'sampler':indices,'shuffled_epoch0':sampler_indices(shuffle=True,epoch=0),'shuffled_epoch1':sampler_indices(shuffle=True,epoch=1),
       'communication_model':[{**ring_model(100*2**20,w,5e-6,12.5e9),'world':w,'payload_bytes':100*2**20,'latency_s':5e-6,'bandwidth_bytes_s':12.5e9} for w in [1,2,4,8,16]],
       'runs':rows,'real_distributed_run':{'status':'not_run','reason':'this lesson verifies numerical semantics on CPU, not network or multi-process behavior'}}
    np.savez_compressed(output/'data.npz',train_x=x.numpy(),train_y=y.numpy(),test_x=vx.numpy(),test_y=vy.numpy())
    (output/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs'));a=p.parse_args();r=main(a.output)
    print(json.dumps({k:r[k] for k in ['gradient_checks','masked_gradient_max_errors','sampler']},indent=2))
