"""Original CPU miniature conditional language model: frozen/full/LoRA adaptation."""
from pathlib import Path
import json, platform, time
import numpy as np
ROOT=Path(__file__).resolve().parent

def softmax(z):
 z=z-np.max(z,axis=-1,keepdims=True);e=np.exp(z);return e/e.sum(axis=-1,keepdims=True)

def loss_grad(X,y,W,mask=None):
 p=softmax(X@W.T);m=np.ones(len(y)) if mask is None else np.asarray(mask,float)
 if m.shape!=(len(y),) or np.any(m<0) or m.sum()==0:raise ValueError('mask must contain positive total weight')
 loss=-float(np.sum(m*np.log(np.maximum(p[np.arange(len(y)),y],1e-300)))/m.sum())
 g=p.copy();g[np.arange(len(y)),y]-=1;g*=m[:,None]/m.sum()
 return loss,g.T@X

def fixture(seed=83):
 rng=np.random.default_rng(seed);d,v=12,8
 # Input feature 0 encodes two domains, remaining features a miniature prompt.
 X=rng.normal(size=(720,d));X[:360,0]=1;X[360:,0]=-1
 W0=rng.normal(0,.35,(v,d));U=rng.normal(size=(v,2));V=rng.normal(size=(2,d));V[:,0]=0
 target=W0+.65*U@V
 # Adaptation and retention have different conditional targets, never the same row.
 y=np.array([rng.choice(v,p=p) for p in softmax(X@target.T)])
 base=np.array([rng.choice(v,p=p) for p in softmax(X@W0.T)])
 return X[:240],y[:240],X[240:360],y[240:360],X[360:],base[360:],W0

def train(X,y,W0,mode='full',rank=2,steps=700,lr=.35,seed=83):
 if mode not in ('frozen','full','lora'):raise ValueError('unknown mode')
 rng=np.random.default_rng(seed);W=W0.copy();A=rng.normal(0,.1,(rank,W.shape[1]));B=np.zeros((W.shape[0],rank));history=[]
 for step in range(steps if mode!='frozen' else 1):
  current=W0+B@A if mode=='lora' else W
  loss,G=loss_grad(X,y,current)
  if step%20==0:history.append([step,loss])
  if mode=='full':W-=lr*G
  elif mode=='lora':
   dA=B.T@G;dB=G@A.T;A-=lr*dA;B-=lr*dB
 return (W0+B@A if mode=='lora' else W),np.array(history),A,B

def evaluate(X,y,W):
 return {'nll':loss_grad(X,y,W)[0],'accuracy':float(np.mean(np.argmax(X@W.T,axis=1)==y))}

def main(out=None):
 out=Path(out or ROOT/'outputs');out.mkdir(parents=True,exist_ok=True)
 X,y,Xt,yt,Xr,yr,W0=fixture();results={};hist={};weights={'base':W0}
 np.savez(ROOT/'data'/'fixture.npz',train_X=X,train_y=y,test_X=Xt,test_y=yt,retention_X=Xr,retention_y=yr,base_W=W0)
 for mode,rank in [('frozen',2),('full',2),('lora',1),('lora',2),('lora',4)]:
  key=mode if mode!='lora' else f'lora_r{rank}';start=time.perf_counter();W,h,A,B=train(X,y,W0,mode,rank)
  params=0 if mode=='frozen' else (W0.size if mode=='full' else rank*sum(W0.shape))
  results[key]={'adaptation_test':evaluate(Xt,yt,W),'retention_test':evaluate(Xr,yr,W),'train_nll':loss_grad(X,y,W)[0], 'trainable_parameters':params,'trainable_float64_bytes':params*8,'observed_cpu_seconds':time.perf_counter()-start,'delta_rank':int(np.linalg.matrix_rank(W-W0)), 'merge_max_abs_error':float(np.max(np.abs(Xt@W.T-(Xt@W0.T+(Xt@A.T)@B.T)))) if mode=='lora' else 0}
  hist[key]=h.tolist();weights[key]=W
 data={'seed':83,'model':'single-step conditional softmax; not a Transformer or pretrained LLM','train_rows':240,'adaptation_test_rows':120,'retention_rows':360,'results':results}
 (out/'metrics.json').write_text(json.dumps(data,indent=2,ensure_ascii=False));(out/'history.json').write_text(json.dumps(hist,indent=2));np.savez(out/'weights.npz',**weights)
 (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__},indent=2))
 return data
if __name__=='__main__':print(json.dumps(main(),indent=2,ensure_ascii=False))
