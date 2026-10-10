"""S1: train a small message-passing network and a relation-blind baseline.
All graphs and measurements are synthetic and dimensionless; no downloads.
"""
from pathlib import Path
import argparse,json,time
import numpy as np

def transition(a):
    a=np.asarray(a,dtype=float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('adjacency must be a finite nonnegative square matrix')
    b=a+np.eye(len(a))
    return b/b.sum(axis=1,keepdims=True)

def make_data(seed=78,n_graphs=150,n=8):
    rng=np.random.default_rng(seed); aa=[]; xx=[]; yy=[]
    for _ in range(n_graphs):
        a=np.zeros((n,n))
        for j in range(n):a[j,(j+1)%n]=a[(j+1)%n,j]=1
        extra=np.triu(rng.random((n,n))<.18,1);a=np.maximum(a,extra+extra.T)
        x=rng.normal(size=(n,1));p=transition(a)
        y=np.tanh(.7*x+1.1*(p@x))+.02*rng.normal(size=(n,1))
        aa.append(a);xx.append(x);yy.append(y)
    return np.array(aa),np.array(xx),np.array(yy)

def features(a,x,use_edges=True):
    return np.concatenate([x,transition(a)@x if use_edges else x],axis=1)

def init(seed=0):
    r=np.random.default_rng(seed)
    return {'w':r.normal(scale=.3,size=(2,12)),'b':np.zeros(12),'v':r.normal(scale=.3,size=(12,1)),'c':np.zeros(1)}

def predict(z,p):return np.tanh(z@p['w']+p['b'])@p['v']+p['c']

def loss_grad(z,y,p):
    h=np.tanh(z@p['w']+p['b']);e=h@p['v']+p['c']-y;g=2*e/e.size
    dh=(g@p['v'].T)*(1-h*h)
    return float(np.mean(e*e)),{'w':z.T@dh,'b':dh.sum(0),'v':h.T@g,'c':g.sum(0)}

def train(z,y,seed=0,steps=1600):
    p=init(seed);m={k:np.zeros_like(v) for k,v in p.items()};v={k:q.copy() for k,q in m.items()};hist=[]
    for t in range(1,steps+1):
        loss,g=loss_grad(z,y,p)
        for k in p:
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2
            p[k]-=.015*(m[k]/(1-.9**t))/(np.sqrt(v[k]/(1-.999**t))+1e-8)
        if t%40==0:hist.append(loss)
    return p,np.array(hist)

def run(output='outputs'):
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    a,x,y=make_data();train_ids=np.arange(100);val_ids=np.arange(100,125);test_ids=np.arange(125,150)
    arrays={'adjacency':a,'x':x,'y':y,'train_ids':train_ids,'val_ids':val_ids,'test_ids':test_ids};stats={};preds={};weights={}
    for name,edges in [('gnn',True),('blind',False)]:
        z=np.array([features(ai,xi,edges) for ai,xi in zip(a,x)])
        metrics=[]
        for seed in [0,1,2]:
            p,h=train(z[train_ids].reshape(-1,2),y[train_ids].reshape(-1,1),seed)
            pred=predict(z[test_ids],p);metrics.append(float(np.mean((pred-y[test_ids])**2)))
            if seed==0:
                preds[name]=pred;arrays[name+'_history']=h
                for k,w in p.items():weights[name+'_'+k]=w
                stats[name+'_val_mse']=float(np.mean((predict(z[val_ids],p)-y[val_ids])**2))
                if edges:
                    perm=np.array([3,7,1,0,6,4,2,5]);p1=predict(features(a[125][perm][:,perm],x[125][perm]),p)
                    stats['permutation_max_error']=float(np.max(np.abs(p1-predict(features(a[125],x[125]),p)[perm])))
        stats[name+'_test_mse_seeds']=metrics;stats[name+'_test_mse_mean']=float(np.mean(metrics))
    p0=transition(a[125]);h=x[125].copy();variances=[]
    for depth in range(41):
        variances.append(float(np.var(h)));h=p0@h
    arrays['smooth_variance']=np.array(variances)
    # A feature intervention is a simulator intervention only because this mechanism is declared.
    xi=x[125].copy();xi[0]+=1;arrays['intervention_effect']=np.tanh(.7*xi+1.1*p0@xi)-np.tanh(.7*x[125]+1.1*p0@x[125])
    stats.update(seconds=time.perf_counter()-start,seed_data=78,train_graphs=100,val_graphs=25,test_graphs=25,steps=1600,hidden=12)
    np.savez(out/'data.npz',**arrays);np.savez(out/'weights.npz',**weights);np.savez(out/'predictions.npz',**preds)
    (out/'results.json').write_text(json.dumps(stats,indent=2));return stats
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--output',default='outputs');args=q.parse_args();print(json.dumps(run(args.output),indent=2))
