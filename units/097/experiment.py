"""原创 CPU 教学实验。运行 python experiment.py；无需网络、GPU 或模型下载。"""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dl-mpl")
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=160); plt.close()

def generate(z,noise_seed,steps,kind):
    # 同一标准正态 z 用于各方法/步数的起点，降低起点抽样混杂。
    mu=2.;v0=.36;T=4.;h=T/steps;x=mu+np.sqrt(v0+T)*z
    rng=np.random.default_rng(noise_seed);start=time.perf_counter()
    for k in range(steps):
        t=T-k*h;score=-(x-mu)/(v0+t)
        if kind=='sde':x+=h*score+np.sqrt(h)*rng.normal(size=len(x))
        else:x+=.5*h*score
    return x,time.perf_counter()-start

def main():
    records=[];arrays={};target_mu=2.;target_std=.6
    for seed in (11,23,37,51,73):
        z=np.random.default_rng(seed).normal(size=4000);arrays[f'z_{seed}']=z
        for steps in (10,40,160):
            for kind in ('sde','ode'):
                x,seconds=generate(z,seed+1000,steps,kind);arrays[f'{kind}_{steps}_{seed}']=x
                # Gaussian矩指标只观察均值/方差，不能排除非Gaussian形状。
                moment_distance=np.sqrt((x.mean()-target_mu)**2+(x.std()-target_std)**2)
                records.append(dict(seed=seed,steps=steps,method=kind,n=4000,nfe=steps,seconds=seconds,mean=float(x.mean()),std=float(x.std()),moment_distance=float(moment_distance),central_95_coverage=float(np.mean(abs(x-target_mu)<1.96*target_std))))
    summary=[]
    for steps in (10,40,160):
        s=[r for r in records if r['steps']==steps and r['method']=='sde'];o=[r for r in records if r['steps']==steps and r['method']=='ode']
        d=np.array([a['moment_distance']-b['moment_distance'] for a,b in zip(s,o)])
        summary.append(dict(steps=steps,paired_difference_mean=float(d.mean()),paired_difference_se=float(d.std(ddof=1)/np.sqrt(len(d)))))
    metrics={"protocol":"fixed before evaluation; analytic score; no model training", "records":records,"paired_summary":summary}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez(OUT/'data.npz',**arrays)
    import csv
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig,ax=plt.subplots(1,2,figsize=(9,3))
    for kind in ('sde','ode'):
        for a,key in zip(ax,['moment_distance','central_95_coverage']):
            vals=[[r[key] for r in records if r['method']==kind and r['steps']==s] for s in (10,40,160)]
            a.errorbar([10,40,160],[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) for v in vals],marker='o',label=kind);a.set_xscale('log');a.set_xlabel('score evaluations');a.set_ylabel(key);a.legend()
    savefig('01_quality.png')
    plt.figure(figsize=(7,3))
    for kind in ('sde','ode'):
        vals=[r for r in records if r['method']==kind];plt.scatter([r['seconds'] for r in vals],[r['moment_distance'] for r in vals],label=kind)
    plt.xlabel('sampling wall seconds (this CPU)');plt.ylabel('moment distance');plt.legend();savefig('02_cost.png')
    plt.figure(figsize=(7,3))
    for kind in ('sde','ode'):plt.hist(arrays[f'{kind}_160_11'],bins=60,density=True,histtype='step',label=kind)
    xx=np.linspace(-.5,4.5,300);plt.plot(xx,np.exp(-.5*((xx-2)/.6)**2)/(.6*np.sqrt(2*np.pi)),label='target');plt.xlabel('x');plt.ylabel('density');plt.legend();savefig('03_distribution.png')
    print(json.dumps({"paired_summary":summary},indent=2));return metrics
if __name__=='__main__':main()
