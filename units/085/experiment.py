"""Two-response preference model: SFT/DPO, reference drift and shortcut stress tests."""
from pathlib import Path
import json,platform
import numpy as np
ROOT=Path(__file__).resolve().parent

def sigmoid(z):return np.exp(-np.logaddexp(0,-np.asarray(z)))
def fixture(n=400,correlation=.95,seed=85):
 rng=np.random.default_rng(seed);correct=rng.choice([-1.,1.],n);verbose=correct*np.where(rng.random(n)<correlation,1.,-1.);delta=np.column_stack([correct,verbose]);chosen=rng.binomial(1,sigmoid(2*correct));return delta,chosen

def objective(w,X,y,method='dpo',beta=.5,reference=None):
 if beta<=0:raise ValueError('beta must be positive')
 if method not in ('dpo','sft'):raise ValueError('unknown method')
 reference=np.array([.8,0.]) if reference is None else np.asarray(reference)
 scale=beta if method=='dpo' else 1.;effective=w-reference if method=='dpo' else w
 z=scale*(X@effective);loss=float(np.mean(np.logaddexp(0,z)-y*z));g=scale*X.T@(sigmoid(z)-y)/len(y);return loss,g

def train(X,y,method='dpo',beta=.5,steps=600,lr=.3):
 w=np.array([.8,0.]);history=[]
 for step in range(steps):
  l,g=objective(w,X,y,method,beta);w-=lr*g
  if step%20==0:history.append([step,l,*w])
 return w,history

def evaluate(w,X,y):
 z=X@w;p=sigmoid(z);q=sigmoid(X@np.array([.8,0.]));tiny=1e-15
 kl=p*np.log(np.maximum(p,tiny)/q)+(1-p)*np.log(np.maximum(1-p,tiny)/(1-q))
 return {'sampled_correctness':float(np.mean(np.where(X[:,0]>0,p,1-p))),'greedy_correctness':float(np.mean((z>0)==(X[:,0]>0))),'annotator_agreement':float(np.mean((z>0)==y)), 'mean_reference_kl':float(np.mean(kl)), 'mean_verbose_probability':float(np.mean(np.where(X[:,1]>0,p,1-p)))}

def main(out=None):
 out=Path(out or ROOT/'outputs');out.mkdir(parents=True,exist_ok=True)
 X,y=fixture();sets={k:fixture(n=1000,correlation=c,seed=s) for k,c,s in [('id',.95,851),('balanced',.5,852),('conflict',0.,853)]};results={};history={};weights={}
 for name,method,beta in [('reference',None,.5),('sft','sft',.5),('dpo_b0.1','dpo',.1),('dpo_b0.5','dpo',.5),('dpo_b2','dpo',2.)]:
  w=np.array([.8,0.]) if method is None else train(X,y,method,beta)[0];weights[name]=w
  results[name]={'weights':w.tolist(),'beta':beta if method=='dpo' else None,'evaluation':{k:evaluate(w,*v) for k,v in sets.items()}}
  if method:history[name]=train(X,y,method,beta)[1]
 # Biased annotator deliberately rewards verbosity; independent evaluator still checks correctness.
 biased=(X[:,1]>0).astype(int);w,h=train(X,biased,'dpo',.5);weights['biased_judge_dpo']=w;results['biased_judge_dpo']={'weights':w.tolist(),'evaluation':{k:evaluate(w,*v) for k,v in sets.items()},'training_judge_agreement':float(np.mean((X@w>0)==biased))};history['biased_judge_dpo']=h
 np.savez(ROOT/'data'/'preferences.npz',train_X=X,train_y=y,**{f'{k}_X':v[0] for k,v in sets.items()},**{f'{k}_y':v[1] for k,v in sets.items()})
 data={'seed':85,'model':'two-candidate conditional policy; not a text-generating LLM','train_pairs':400,'test_pairs_per_condition':1000,'results':results};(out/'metrics.json').write_text(json.dumps(data,indent=2));(out/'history.json').write_text(json.dumps(history));np.savez(out/'weights.npz',**weights);(out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__},indent=2));return data
if __name__=='__main__':print(json.dumps(main(),indent=2))
