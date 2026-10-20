"""CPU attention mechanics: exact blocked softmax, local masking, synthetic retrieval."""
from pathlib import Path
import os,time,json,csv,platform
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl-course-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/dl-course-cache')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs';FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
def dense(q,k,v):return torch.softmax(q@k.T/(q.shape[1]**.5),dim=-1)@v

def blocked(q,k,v,block=32):
    result=[]
    for i in range(0,len(q),block):
        qi=q[i:i+block];m=torch.full((len(qi),1),-torch.inf,dtype=q.dtype);l=torch.zeros_like(m);acc=torch.zeros((len(qi),v.shape[1]),dtype=q.dtype)
        for j in range(0,len(k),block):
            s=qi@k[j:j+block].T/(q.shape[1]**.5);nm=torch.maximum(m,s.max(dim=1,keepdim=True).values)
            alpha=torch.exp(m-nm);p=torch.exp(s-nm)
            acc=alpha*acc+p@v[j:j+block];l=alpha*l+p.sum(dim=1,keepdim=True);m=nm
        result.append(acc/l)
    return torch.cat(result)

def local(q,k,v,window=32):
    n=len(q);idx=torch.arange(n);mask=(idx[None,:]<=idx[:,None]) & (idx[None,:]>idx[:,None]-window)
    score=q@k.T/q.shape[1]**.5;return torch.softmax(score.masked_fill(~mask,-torch.inf),dim=-1)@v

def main():
    torch.set_num_threads(1);rows=[];arrays={};rng=np.random.default_rng(307)
    for n in (64,128,256,512):
        q,k,v=[torch.tensor(rng.normal(size=(n,32)),dtype=torch.float64) for _ in range(3)]
        for name,a in zip(('q','k','v'),(q,k,v)):arrays[f'{name}_{n}']=a.numpy()
        reference=dense(q,k,v);candidate=blocked(q,k,v);arrays[f'dense_{n}']=reference.numpy();arrays[f'blocked_{n}']=candidate.numpy()
        for fn in (dense,blocked):fn(q,k,v)
        times={'dense':[],'blocked':[]}
        for repeat in range(9):
            order=['dense','blocked'] if repeat%2==0 else ['blocked','dense']
            for name in order:
                t=time.perf_counter();out=({'dense':dense,'blocked':blocked}[name])(q,k,v);times[name].append(time.perf_counter()-t)
        for name in ('dense','blocked'):
            for repeat,sec in enumerate(times[name]):rows.append(dict(length=n,d=32,method=name,repeat=repeat,seconds=sec,tokens_per_second=n/sec,max_abs_error=float((candidate-reference).abs().max()) if name=='blocked' else 0,score_tile_bytes=(min(32,n)**2 if name=='blocked' else n*n)*8,memory_kind='analytical score tile only; not process peak RSS'))
    retrieval=[]
    for n in (64,128,256,512):
        for position in (0,n//2,n-1):
            q=torch.zeros((1,32),dtype=torch.float64);q[0,0]=12*np.sqrt(32)
            k=torch.zeros((n,32),dtype=torch.float64);k[position,0]=1
            v=torch.zeros((n,1),dtype=torch.float64);v[position,0]=1
            full=float(dense(q,k,v)[0,0]);window=float(dense(q,k[-32:],v[-32:])[0,0]);blocked_val=float(blocked(q,k,v)[0,0])
            retrieval.append(dict(length=n,target_position=position,full_value=full,blocked_value=blocked_val,local32_value=window,full_success=full>.5,local_success=window>.5,target_visible_to_local=position>=n-32))
    np.savez(OUT/'data.npz',**arrays)
    for name,data in [('timings.csv',rows),('retrieval.csv',retrieval)]:
        with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    metrics=dict(protocol='CPU float64, one thread, warmup 1, 9 timed forward passes; alternating method order; block size 32. Bidirectional attention correctness. Synthetic single-query retrieval is not a trained language-model benchmark.',max_abs_error=max(r['max_abs_error'] for r in rows),timings=rows,retrieval=retrieval,full_successes=sum(r['full_success'] for r in retrieval),local_successes=sum(r['local_success'] for r in retrieval),retrieval_count=len(retrieval),memory_scope='Score tile size is an analytical buffer lower-level accounting, not measured peak memory; GPU memory not available.')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(8,3.5))
    for method in ('dense','blocked'):
        ns=[64,128,256,512];med=[np.median([r['seconds'] for r in rows if r['length']==n and r['method']==method])*1000 for n in ns];ax[0].plot(ns,med,'o-',label=method);ax[1].plot(ns,[(n*n if method=='dense' else 32*32)*8 for n in ns],'o-',label=method)
    ax[0].set(xlabel='Sequence length',ylabel='Measured median CPU milliseconds');ax[1].set(xlabel='Sequence length',ylabel='Analytical score tile bytes',yscale='log');ax[0].legend();ax[1].legend();fig.tight_layout();fig.savefig(FIG/'01_cost.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3));a=np.array([[r['local32_value'] for r in retrieval if r['length']==n] for n in (64,128,256,512)]);im=ax.imshow(a,vmin=0,vmax=1,cmap='YlGnBu');ax.set_xticks([0,1,2],['Beginning','Middle','End']);ax.set_yticks(range(4),['64','128','256','512']);ax.set(xlabel='Target position',ylabel='Context length',title='Local window retrieval: value near 1 means recovered');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIG/'02_retrieval.png',dpi=160);plt.close(fig)
    (ROOT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='CPU',threads=1,dtype='float64',timing_scope='forward function only; excludes input creation and output serialization'),indent=2));print(json.dumps({k:v for k,v in metrics.items() if k not in ('timings','retrieval')},indent=2))
if __name__=='__main__':main()
