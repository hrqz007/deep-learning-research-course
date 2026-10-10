"""Bandit and a two-decision MDP: exact enumeration versus policy gradients."""
from pathlib import Path
import json,platform
import numpy as np
ROOT=Path(__file__).resolve().parent

def sigmoid(x):return np.exp(-np.logaddexp(0,-np.asarray(x)))
def bandit_exact(theta,rewards=(.2,.8)):
 p=float(sigmoid(theta));return (1-p)*rewards[0]+p*rewards[1],p*(1-p)*(rewards[1]-rewards[0])
def bandit_samples(theta,n,seed=84,baseline=0):
 rng=np.random.default_rng(seed);p=float(sigmoid(theta));a=rng.binomial(1,p,n);r=rng.binomial(1,np.where(a==1,.8,.2));return (a-p)*(r-baseline)
def enumerate_mdp(theta,gamma=.9):
 # action a0 chooses second state (0 or 1); a1 chooses final outcome.
 p=sigmoid(theta);rows=[]
 for a0 in (0,1):
  for a1 in (0,1):
   q=(p[0] if a0 else 1-p[0])*(p[1+a0] if a1 else 1-p[1+a0])
   r0=-.1*a0;r1=[[.1,.4],[0.,1.]][a0][a1];G=r0+gamma*r1
   score=np.zeros(3);score[0]=a0-p[0];score[1+a0]=a1-p[1+a0]
   rows.append({'a0':a0,'a1':a1,'probability':float(q),'return':float(G),'score':score})
 return rows

def mdp_exact(theta,gamma=.9):
 rows=enumerate_mdp(theta,gamma);return sum(r['probability']*r['return'] for r in rows),sum((r['probability']*r['return']*r['score'] for r in rows),np.zeros(3))
def mdp_mc(theta,n=50000,seed=84,gamma=.9,baseline=0):
 rng=np.random.default_rng(seed);p=sigmoid(theta);a0=rng.binomial(1,p[0],n);a1=rng.binomial(1,p[1+a0]);r0=-.1*a0;r1=np.where(a0==0,np.where(a1==0,.1,.4),a1.astype(float));G=r0+gamma*r1
 score=np.zeros((n,3));score[:,0]=a0-p[0];score[np.arange(n),1+a0]=a1-p[1+a0]
 return score*(G[:,None]-baseline)

def importance_bandit(target,behavior,n=50000,seed=84):
 rng=np.random.default_rng(seed);p=float(sigmoid(target));q=float(sigmoid(behavior));a=rng.binomial(1,q,n);r=rng.binomial(1,np.where(a==1,.8,.2));ratio=np.where(a,p/q,(1-p)/(1-q));raw=(a-p)*r
 return {'uncorrected':float(raw.mean()),'importance_corrected':float((ratio*raw).mean()),'max_ratio':float(ratio.max()),'effective_sample_size':float(ratio.sum()**2/(ratio@ratio))}

def main(out=None):
 out=Path(out or ROOT/'outputs');out.mkdir(parents=True,exist_ok=True);theta=np.array([.2,-.4,.7]);J,g=mdp_exact(theta);rows=enumerate_mdp(theta)
 (ROOT/'data'/'mdp.json').write_text(json.dumps({'gamma':.9,'initial_logits':theta.tolist(),'first_reward':[0.,-.1],'terminal_reward':[[.1,.4],[0.,1.]],'transition':'first action selects second state; second action terminates'},indent=2))
 samples=mdp_mc(theta);center=mdp_mc(theta,baseline=J);history=[];t=theta.copy()
 for step in range(101):
  v,d=mdp_exact(t);history.append([step,v,*t]);t+=.8*d
 b0=bandit_samples(.3,100000,baseline=0);b1=bandit_samples(.3,100000,baseline=.5);bj,bg=bandit_exact(.3)
 data={'seed':84,'mdp':{'theta':theta.tolist(),'gamma':.9,'exact_return':J,'exact_gradient':g.tolist(),'mc_gradient':samples.mean(0).tolist(),'mc_standard_error':(samples.std(0,ddof=1)/np.sqrt(len(samples))).tolist(),'variance_no_baseline':samples.var(0,ddof=1).tolist(),'variance_constant_baseline':center.var(0,ddof=1).tolist(),'trained_return':history[-1][1]},'bandit':{'exact_return':bj,'exact_gradient':bg,'mc_gradient':float(b0.mean()),'variance_no_baseline':float(b0.var(ddof=1)),'variance_baseline_half':float(b1.var(ddof=1))},'off_policy':importance_bandit(.3,-1.3)}
 (out/'metrics.json').write_text(json.dumps(data,indent=2));(out/'history.json').write_text(json.dumps(history));(out/'trajectories.json').write_text(json.dumps([{**r,'score':r['score'].tolist()} for r in rows],indent=2));np.savez(out/'gradient_samples.npz',raw=samples,baseline=center)
 (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__},indent=2));return data
if __name__=='__main__':print(json.dumps(main(),indent=2))
