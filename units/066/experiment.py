"""DL066: mixed precision, unequal microbatches, saved tensors and recomputation.
CPU BF16 is actually tested. CUDA FP16/GradScaler is explained, not claimed run.
"""
from pathlib import Path
import argparse,copy,json,platform,time
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class Net(nn.Module):
    def __init__(self,dropout=0.,batchnorm=False):
        super().__init__();layers=[]
        for _ in range(4):
            layers += [nn.Linear(64,64)]
            if batchnorm:layers += [nn.BatchNorm1d(64)]
            layers += [nn.Tanh()]
            if dropout:layers += [nn.Dropout(dropout)]
        self.body=nn.Sequential(*layers);self.head=nn.Linear(64,1)
    def forward(self,x,recompute=False,preserve_rng=True):
        if recompute:
            x=checkpoint(self.body,x,use_reentrant=False,preserve_rng_state=preserve_rng)
        else:x=self.body(x)
        return self.head(x)


def data(n=128,seed=66):
    g=torch.Generator().manual_seed(seed);x=torch.randn(n,64,generator=g)
    y=torch.sin(x[:,0:1])+0.3*x[:,1:2]-0.2*x[:,2:3]
    return x,y


def model(seed=11,**kwargs):torch.manual_seed(seed);return Net(**kwargs)

def grad_vector(m):return torch.cat([p.grad.detach().flatten() for p in m.parameters()])

def relative_error(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-12))


def gradients(m,x,y,bf16=False,recompute=False,sizes=None,preserve_rng=True,wrong_equal=False):
    m.zero_grad(set_to_none=True)
    if sizes is None:sizes=[len(x)]
    if any(s<=0 for s in sizes) or sum(sizes)!=len(x):raise ValueError('positive sizes must partition batch')
    total=0.;start=0
    for size in sizes:
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
            pred=m(x[start:start+size],recompute,preserve_rng)
            # mean on each microbatch must be weighted by its fraction of examples.
            loss=nn.functional.mse_loss(pred,y[start:start+size])
            weight=1/len(sizes) if wrong_equal else size/len(x)
            weighted=loss*weight
        weighted.backward();total+=float(weighted.detach());start+=size
    return total,grad_vector(m).clone()


def saved_payload(m,x,y,recompute=False):
    """Count tensors saved in forward, excluding parameter storage.

    Unique storage deduplication prevents counting views several times. This
    measures retained payload visible to autograd hooks, NOT process/GPU peak.
    """
    parameter_storages={p.untyped_storage().data_ptr() for p in m.parameters()}
    storages={};calls=0
    def pack(t):
        nonlocal calls
        calls+=1;st=t.untyped_storage();key=st.data_ptr()
        if key not in parameter_storages and st.nbytes()>0:storages[key]=st.nbytes()
        return t
    m.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        loss=nn.functional.mse_loss(m(x,recompute),y)
    # Snapshot before backward; hooks are deliberately scoped to forward only.
    result={'saved_nonparameter_unique_storage_bytes':sum(storages.values()),'saved_tensor_calls':calls,
            'scope':'unique non-parameter storage retained by forward autograd; includes input/target; not peak memory'}
    loss.backward();return result


def timed_backward(base,x,y,recompute,repeats=7):
    m=copy.deepcopy(base)
    for _ in range(3):gradients(m,x,y,recompute=recompute)
    times=[]
    for _ in range(repeats):
        t=time.perf_counter()
        for _ in range(5):gradients(m,x,y,recompute=recompute)
        times.append((time.perf_counter()-t)/5)
    return {'seconds_per_forward_backward':times,'median_seconds':float(np.median(times)),
            'scope':'CPU forward+backward+zero_grad; no optimizer step; separate from saved-tensor instrumentation'}


def train(seed,mode,steps=160):
    m=model(seed);x,y=data();vx,vy=data(256,6601);o=torch.optim.SGD(m.parameters(),lr=.06);history=[]
    for i in range(steps):
        kwargs={'bf16':mode=='bf16','recompute':mode=='checkpoint','sizes':[31,47,50] if mode=='accumulate' else None}
        loss,_=gradients(m,x,y,**kwargs);o.step()
        if i%20==0 or i==steps-1:
            with torch.no_grad():val=float(nn.functional.mse_loss(m(vx),vy))
            history.append({'step':i+1,'train_mse':loss,'test_mse_fp32_inference':val})
    return m,history


def main(output=Path('outputs')):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    x,y=data();base=model();loss,g=gradients(base,x,y);comparisons={}
    for name,kw in [('fp32',{}),('bf16',{'bf16':True}),('accumulate',{'sizes':[31,47,50]}),('checkpoint',{'recompute':True}),('wrong_equal',{'sizes':[31,47,50],'wrong_equal':True})]:
        v,h=gradients(copy.deepcopy(base),x,y,**kw)
        comparisons[name]={'loss':v,'gradient_relative_error':relative_error(g,h),'gradient_max_abs_error':float((g-h).abs().max())}
    drop=model(dropout=.4)
    torch.manual_seed(901);_,a=gradients(copy.deepcopy(drop),x,y)
    torch.manual_seed(901);_,b=gradients(copy.deepcopy(drop),x,y,recompute=True,preserve_rng=True)
    torch.manual_seed(901);_,c=gradients(copy.deepcopy(drop),x,y,recompute=True,preserve_rng=False)
    bn=model(batchnorm=True);_,u=gradients(copy.deepcopy(bn),x,y);_,v=gradients(copy.deepcopy(bn),x,y,sizes=[31,47,50])
    values=torch.tensor([1e-8,1.,1.+2**-10,70000.],dtype=torch.float32)
    casts={str(t):[str(float(v)) for v in values.to(t)] for t in [torch.float32,torch.float16,torch.bfloat16]}
    # Scaling before conversion can preserve a tiny representable gradient;
    # scaling an already-rounded zero cannot recover lost information.
    tiny=torch.tensor(1e-8);scale=65536.
    scaling={'original':float(tiny),'fp16_direct':float(tiny.half()),'scale_then_cast_unscale':float((tiny*scale).half().float()/scale),
             'cast_then_scale_unscale':float(tiny.half().float()*scale/scale),'scope':'scalar arithmetic demonstration, not an actual CUDA GradScaler run'}
    rows=[]
    for seed in [11,23,37]:
        for mode in ['fp32','bf16','accumulate','checkpoint']:
            trained,history=train(seed,mode)
            torch.save({'state_dict':trained.state_dict(),'seed':seed,'mode':mode,'steps':160},output/f'{mode}_seed{seed}.pt')
            rows.append({'seed':seed,'mode':mode,'history':history,'final_test_mse':history[-1]['test_mse_fp32_inference']})
    np.savez_compressed(output/'data.npz',train_x=x.numpy(),train_y=y.numpy(),test_x=data(256,6601)[0].numpy(),test_y=data(256,6601)[1].numpy())
    r={'unit':'066','runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':1},
       'comparisons':comparisons,'casting':casts,'scaling':scaling,
       'dropout':{'preserve_rng_relative_error':relative_error(a,b),'no_preserve_relative_error':relative_error(a,c)},
       'batchnorm_accumulation_relative_error':relative_error(u,v),
       'saved_tensors':{'full':saved_payload(copy.deepcopy(base),x,y),'checkpoint':saved_payload(copy.deepcopy(base),x,y,True)},
       'timing':{'full':timed_backward(base,x,y,False),'checkpoint':timed_backward(base,x,y,True)},'runs':rows,
       'cuda_fp16_gradscaler':{'status':'not_run','reason':'CPU-only verified environment'},
       'gpu_peak_memory':None}
    (output/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs'));a=p.parse_args();r=main(a.output)
    print(json.dumps({k:r[k] for k in ['comparisons','dropout','batchnorm_accumulation_relative_error','saved_tensors','timing']},indent=2))
