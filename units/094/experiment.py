"""DL094: analytic Gaussian VE diffusion, reverse SDE and probability-flow ODE.
Oracle-score numerical laboratory, not trained generative modeling.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl094-mpl')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
V0=.5
BETA=2.
T=1.
MU=.7

def variance(t): return V0+BETA*np.asarray(t)
def score(x,t): return -(x-MU)/variance(t)

def reverse(initial,steps,kind,rng=None,paths=False):
    """Use u=T-t, hence dt=-du. Euler-Maruyama or explicit Euler.
    kind=sde: +beta*score*du + sqrt(beta*du)*z.
    kind=ode: +beta/2*score*du. Fresh independent z at each step.
    """
    if kind not in ('sde','ode'): raise ValueError('kind must be sde or ode')
    if steps < 1: raise ValueError('steps must be positive')
    x=np.asarray(initial).copy();h=T/steps;history=[x[:8].copy()]
    for k in range(steps):
        t=T-k*h
        x += (BETA if kind=='sde' else BETA/2)*score(x,t)*h
        if kind=='sde':
            if rng is None: raise ValueError('SDE requires a random generator')
            x += np.sqrt(BETA*h)*rng.normal(size=x.shape)
        history.append(x[:8].copy())
    return (x,np.array(history)) if paths else x

def discrete_variance(steps,kind):
    """Exact variance of the numerical method, without Monte Carlo uncertainty."""
    h=T/steps;v=variance(T)
    for k in range(steps):
        t=T-k*h;a=1-(BETA if kind=='sde' else BETA/2)*h/variance(t)
        v=a*a*v+(BETA*h if kind=='sde' else 0.)
    return float(v)

def normal_cdf(x):
    from math import erf
    return np.array([.5*(1+erf(float(z)/np.sqrt(2))) for z in x])

def ks_distance(x):
    z=np.sort((x-MU)/np.sqrt(V0)); f=normal_cdf(z); n=len(x)
    return float(max(np.max(np.arange(1,n+1)/n-f),np.max(f-np.arange(n)/n)))

def main():
    out,figs=ROOT/'outputs',ROOT/'figures';out.mkdir(exist_ok=True);figs.mkdir(exist_ok=True)
    n=40000;steps_list=[10,20,50,100,200]
    initial=np.random.default_rng(9401).normal(MU,np.sqrt(variance(T)),n)
    rows=[];saved={};last={}
    for kind in ('sde','ode'):
        for steps in steps_list:
            samples,path=reverse(initial,steps,kind,np.random.default_rng(9402+steps),True)
            v_exact=discrete_variance(steps,kind)
            rows.append({'method':kind,'steps':steps,'h':T/steps,'sample_mean':float(samples.mean()),
              'sample_variance':float(samples.var()),'discrete_theory_variance':v_exact,
              'theory_variance_abs_error':abs(v_exact-V0),'ks_to_target':ks_distance(samples)})
            saved[f'{kind}_{steps}']=samples
            if steps==steps_list[-1]: last[kind]=(samples,path)
    # Exact probability-flow map is a useful numerical reference and unit test.
    exact_ode=MU+(initial-MU)*np.sqrt(V0/variance(T))
    forward_rng=np.random.default_rng(9403)
    forward=forward_rng.normal(MU,np.sqrt(V0),n);forward_vars=[float(forward.var())]
    for _ in range(200):
        forward += np.sqrt(BETA*T/200)*forward_rng.normal(size=n)
        forward_vars.append(float(forward.var()))
    metrics={'n':n,'mean':MU,'v0':V0,'beta':BETA,'T':T,
      'description':'oracle score, exact terminal initialization, Euler numerical study',
      'forward_final_variance':forward_vars[-1], 'forward_theory_final_variance':float(variance(T)),
      'initial_sample_variance':float(initial.var()),
      'ode_200_rmse_to_exact_map':float(np.sqrt(np.mean((last['ode'][0]-exact_ode)**2))), 'results':rows}
    np.savez_compressed(out/'data.npz',initial=initial,exact_ode=exact_ode,
       forward_final=forward,forward_variances=forward_vars,sde_paths=last['sde'][1],ode_paths=last['ode'][1],**saved)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(10,4));u=np.linspace(0,T,201)
    for j,kind in enumerate(('sde','ode')):
        ax[j].plot(u,last[kind][1]);ax[j].set(xlabel='Reverse elapsed time u=1-t',ylabel='State x',title=f'{kind.upper()} paths, same initial points')
    fig.tight_layout();fig.savefig(figs/'01_paths.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));xx=np.linspace(-2,3.5,400)
    for kind in ('sde','ode'):ax.hist(last[kind][0],bins=70,density=True,histtype='step',label=kind.upper())
    ax.plot(xx,np.exp(-(xx-MU)**2/(2*V0))/np.sqrt(2*np.pi*V0),'k--',label='Target Gaussian')
    ax.set(xlabel='x at t=0',ylabel='Density',title='Final marginal distributions');ax.legend();fig.tight_layout();fig.savefig(figs/'02_marginals.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for kind in ('sde','ode'):
        rr=[r for r in rows if r['method']==kind]
        ax[0].loglog([r['h'] for r in rr],[r['theory_variance_abs_error'] for r in rr],'o-',label=kind.upper())
    ax[0].set(xlabel='Step size h',ylabel='Exact numerical variance bias',title='Discretization error without MC noise');ax[0].legend()
    ax[1].plot(u,forward_vars,label='Empirical');ax[1].plot(u,variance(u),'--',label='Analytic');ax[1].set(xlabel='Forward time t',ylabel='Variance',title='Forward Gaussian marginals');ax[1].legend()
    fig.tight_layout();fig.savefig(figs/'03_step_error.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':main()
