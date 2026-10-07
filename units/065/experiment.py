"""DL065: actual CPU/CUDA training, timing boundaries and explicit memory accounting.

No data download. Timing is descriptive of the running machine, not a benchmark
leaderboard. All exported durations are measured; GPU fields are null on CPU.
"""
from pathlib import Path
import argparse, copy, json, platform, time
import numpy as np
import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function


def make_data(n=1024, seed=65, device='cpu'):
    """Independent local generator avoids accidental dependence on global RNG."""
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(n,32,generator=g)
    y=(x[:,:4].sum(1)+0.5*x[:,4]*x[:,5]>0).long()
    return x.to(device), y.to(device)


def make_model(seed=11, device='cpu'):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2)).to(device)


def sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def step(model, opt, x, y, implementation='vectorized'):
    opt.zero_grad(set_to_none=True)
    with record_function('course_forward'):
        if implementation=='vectorized': logits=model(x)
        elif implementation=='row_loop': logits=torch.cat([model(row[None]) for row in x],dim=0)
        else: raise ValueError('unknown implementation')
        loss=nn.functional.cross_entropy(logits,y)
    with record_function('course_backward'): loss.backward()
    with record_function('course_optimizer'): opt.step()
    return loss.detach()


def tensor_bytes(t): return t.numel()*t.element_size()


def memory_ledger(model,opt):
    """Logical payload bytes. NOT unique storage/RSS/allocator peak/workspace."""
    p=sum(tensor_bytes(p) for p in model.parameters())
    g=sum(tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)
    states=sum(tensor_bytes(v) for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    return {'parameter_bytes':p,'gradient_bytes':g,'optimizer_tensor_bytes':states,
            'logical_total_bytes':p+g+states,
            'scope':'tensor element payload only; excludes activations, allocator, workspaces and Python overhead'}


def measure(implementation,batch=64,repeats=7,steps_per_repeat=10,warmup=5,device='cpu'):
    if min(batch,repeats,steps_per_repeat)<1 or warmup<0: raise ValueError('positive sizes required')
    x,y=make_data(batch,device=device);m=make_model(device=device);opt=torch.optim.Adam(m.parameters(),lr=.002,foreach=False)
    sync(device);t=time.perf_counter();step(m,opt,x,y,implementation);sync(device)
    first=time.perf_counter()-t
    # First step lazily allocates Adam states; warm-up is outside all steady blocks.
    for _ in range(warmup): step(m,opt,x,y,implementation)
    durations=[]
    for _ in range(repeats):
        sync(device);t=time.perf_counter()
        for _ in range(steps_per_repeat):step(m,opt,x,y,implementation)
        sync(device);durations.append((time.perf_counter()-t)/steps_per_repeat)
    a=np.asarray(durations)
    return {'implementation':implementation,'batch':batch,'first_step_seconds':first,
            'warmup_steps':warmup,'repeats':repeats,'steps_per_repeat':steps_per_repeat,
            'seconds_per_step_blocks':durations,'median_seconds':float(np.median(a)),
            'q25_seconds':float(np.quantile(a,.25)),'q75_seconds':float(np.quantile(a,.75)),
            'samples_per_second':float(batch/np.median(a)),'boundary':'resident tensors, zero_grad + forward + mean CE + backward + Adam; no I/O or transfer'}


def profile_steps(output,device='cpu'):
    x,y=make_data(128,device=device);m=make_model(device=device);o=torch.optim.Adam(m.parameters(),lr=.002,foreach=False)
    for _ in range(3):step(m,o,x,y)
    acts=[ProfilerActivity.CPU]
    if torch.device(device).type=='cuda':acts.append(ProfilerActivity.CUDA)
    # Profiling runs separately: instrumentation overhead must not contaminate timing.
    with profile(activities=acts,record_shapes=True,profile_memory=True) as prof:
        for _ in range(5):step(m,o,x,y)
        sync(device)
    events=sorted(prof.key_averages(),key=lambda v:v.self_cpu_time_total,reverse=True)
    rows=[{'operator':v.key,'calls':v.count,'self_cpu_us':v.self_cpu_time_total,
           'cpu_total_us':v.cpu_time_total,'self_cpu_memory_bytes':v.self_cpu_memory_usage} for v in events]
    (output/'profiler.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=18))
    return rows


def main(output=Path('outputs'),device='cpu'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run --device cpu')
    torch.set_num_threads(1)
    x,y=make_data(device=device);tx,ty=make_data(512,seed=6501,device=device)
    m=make_model(device=device);o=torch.optim.Adam(m.parameters(),lr=.002,foreach=False)
    before=memory_ledger(m,o);history=[]
    for epoch in range(20):
        for start in range(0,len(x),64):step(m,o,x[start:start+64],y[start:start+64])
        with torch.no_grad():history.append({'epoch':epoch+1,'train_loss':float(nn.functional.cross_entropy(m(x),y)),
                                            'test_accuracy':float((m(tx).argmax(1)==ty).float().mean())})
    after=memory_ledger(m,o)
    torch.save({'state_dict':m.cpu().state_dict(),'seed':11,'epochs':20,'architecture':[32,128,128,2]},output/'trained_model.pt')
    np.savez_compressed(output/'data.npz',train_x=x.cpu().numpy(),train_y=y.cpu().numpy(),test_x=tx.cpu().numpy(),test_y=ty.cpu().numpy())
    measures=[measure(k,device=device) for k in ['vectorized','row_loop']]
    sweeps=[measure('vectorized',b,device=device) for b in [16,64,256]]
    gpu=None
    if device=='cuda':
        gm=make_model(device=device);go=torch.optim.Adam(gm.parameters(),foreach=False);gx,gy=make_data(256,device=device)
        for _ in range(3):step(gm,go,gx,gy)
        sync(device);torch.cuda.reset_peak_memory_stats(device)
        step(gm,go,gx,gy);sync(device)
        gpu={'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
             'scope':'process allocator counters during warmed step; includes other live CUDA tensors in process'}
    r={'unit':'065','runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':device,'threads':1},
       'history':history,'memory_before_first_step':before,'memory_after_training':after,'measurements':measures,'batch_sweep':sweeps,
       'profile':profile_steps(output,device),'gpu_memory':gpu,
       'limitations':['CPU timing changes with load; no portability claim','CPU logical ledger is not peak memory','Profiler self memory values are event attribution, not peak RSS']}
    (output/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs'));p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    r=main(a.output,a.device);print(json.dumps({'accuracy':r['history'][-1]['test_accuracy'],'timings':r['measurements'],'memory':r['memory_after_training']},indent=2))
