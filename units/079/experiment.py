"""S2: equivariant radial vector field vs ordinary and augmented networks.
Positions are in metres, force in newtons; rotation angles are radians.
"""
from pathlib import Path
import argparse,json,time
import numpy as np

def rotation(theta):
    c,s=np.cos(theta),np.sin(theta);return np.array([[c,-s],[s,c]])

def truth(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all():raise ValueError('x must be finite N by 2')
    return -(0.7+0.25*np.sum(x*x,axis=1,keepdims=True))*x

def data(seed=79,n=192,sector=True):
    rng=np.random.default_rng(seed);r=rng.uniform(.3,1.8,n);angle=rng.uniform(-.22,.22,n) if sector else rng.uniform(-np.pi,np.pi,n)
    x=r[:,None]*np.stack([np.cos(angle),np.sin(angle)],axis=1)
    return x,truth(x)

def init(kind,seed=0):
    rng=np.random.default_rng(seed);d,o=(1,1) if kind=='equivariant' else (2,2)
    return {'w':rng.normal(scale=.4,size=(d,16)),'b':np.zeros(16),'v':rng.normal(scale=.2,size=(16,o)),'c':np.zeros(o)}

def predict(x,p,kind):
    z=np.sum(x*x,axis=1,keepdims=True) if kind=='equivariant' else x
    scalar=np.tanh(z@p['w']+p['b'])@p['v']+p['c']
    return -scalar*x if kind=='equivariant' else scalar

def loss_grad(x,y,p,kind):
    z=np.sum(x*x,axis=1,keepdims=True) if kind=='equivariant' else x
    h=np.tanh(z@p['w']+p['b']);o=h@p['v']+p['c'];pred=-o*x if kind=='equivariant' else o
    err=pred-y;g=2*err/err.size
    if kind=='equivariant':g=-np.sum(g*x,axis=1,keepdims=True)
    dh=g@p['v'].T*(1-h*h)
    return float(np.mean(err*err)),{'w':z.T@dh,'b':dh.sum(0),'v':h.T@g,'c':g.sum(0)}

def train(x,y,kind,seed=0,augment=False,steps=1600):
    p=init(kind,seed);m={k:np.zeros_like(v) for k,v in p.items()};v={k:q.copy() for k,q in m.items()};rng=np.random.default_rng(seed+700);history=[]
    for t in range(1,steps+1):
        if augment:
            q=rotation(rng.uniform(-np.pi,np.pi));xx=x@q.T;yy=y@q.T
        else:xx,yy=x,y
        loss,g=loss_grad(xx,yy,p,kind)
        for k in p:
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2;p[k]-=.01*(m[k]/(1-.9**t))/(np.sqrt(v[k]/(1-.999**t))+1e-8)
        if t%40==0:history.append(loss)
    return p,np.array(history)

def run(output='outputs'):
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True);x,y=data();xt,yt=data(790,256,False);xv,yv=data(791,96,False)
    arrays={'train_x':x,'train_y':y,'val_x':xv,'val_y':yv,'test_x':xt,'test_y':yt};weights={};preds={};res={};angles=np.linspace(0,2*np.pi,61)
    for name,kind,aug in [('equivariant','equivariant',False),('ordinary','ordinary',False),('augmented','ordinary',True)]:
        metrics=[]
        for seed in [0,1,2]:
            p,h=train(x,y,kind,seed,aug);pred=predict(xt,p,kind);metrics.append(float(np.mean((pred-yt)**2)))
            if seed==0:
                preds[name]=pred;arrays[name+'_history']=h
                for k,v in p.items():weights[name+'_'+k]=v
                res[name+'_val_mse']=float(np.mean((predict(xv,p,kind)-yv)**2))
                errors=[]
                for angle in angles:
                    q=rotation(angle);errors.append(np.max(np.abs(predict(xt@q.T,p,kind)-pred@q.T)))
                arrays[name+'_equivariance_errors']=np.array(errors);res[name+'_max_equivariance_error']=float(max(errors))
        res[name+'_test_mse_seeds']=metrics;res[name+'_test_mse_mean']=float(np.mean(metrics))
    arrays['angles']=angles
    # Translation is valid only when the centre moves too: use relative coordinates.
    centre=np.array([2.,-1.]);res['translated_relative_error']=float(np.max(np.abs(truth((xt+centre)-centre)-truth(xt))))
    # Anisotropic mechanism breaks rotation symmetry; imposing it would be wrong.
    q=rotation(np.pi/2);aniso=lambda z:-z*np.array([1.,2.]);res['anisotropic_symmetry_violation']=float(np.max(np.abs(aniso(xt@q.T)-aniso(xt)@q.T)))
    res.update(seconds=time.perf_counter()-start,steps=1600,train_positions=192,test_positions=256,seed_data=79)
    np.savez(out/'data.npz',**arrays);np.savez(out/'weights.npz',**weights);np.savez(out/'predictions.npz',**preds);(out/'results.json').write_text(json.dumps(res,indent=2));return res
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',default='outputs');print(json.dumps(run(a.parse_args().output),indent=2))
