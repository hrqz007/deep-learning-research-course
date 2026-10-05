"""Independent NumPy full BPTT with frozen padded states, token-mean BCE.
All public numerical inputs are finite float64 CPU-sized arrays, not arbitrary data.
"""
import numpy as np
NAMES=('U','W','b','v','c')

def validate(p,x,lengths,y=None):
    if not isinstance(x,np.ndarray) or x.dtype != np.float64 or x.ndim != 3:
        raise ValueError('x must be a float64 [N,T,D] NumPy array')
    n,t,d=x.shape
    if not (1<=n<=4096 and 1<=t<=128 and 1<=d<=64): raise ValueError('shape outside supported domain')
    if not np.isfinite(x).all() or np.max(np.abs(x))>10: raise ValueError('x must be finite and |x| <= 10')
    if not isinstance(lengths,np.ndarray) or lengths.shape!=(n,) or lengths.dtype.kind not in 'iu' or np.any(lengths<1) or np.any(lengths>t):
        raise ValueError('lengths must be integer [N] in [1,T]')
    if set(p)!=set(NAMES): raise ValueError('parameters must be U,W,b,v,c')
    h=p['b'].size
    shapes={'U':(h,d),'W':(h,h),'b':(h,),'v':(h,),'c':()}
    if not 1<=h<=64: raise ValueError('hidden size outside [1,64]')
    for k,s in shapes.items():
        a=p[k]
        if not isinstance(a,np.ndarray) or a.dtype!=np.float64 or a.shape!=s or not np.isfinite(a).all() or np.max(np.abs(a))>100:
            raise ValueError(f'{k} needs finite float64 shape {s} and |parameter| <= 100')
    if y is not None and (not isinstance(y,np.ndarray) or y.dtype!=np.float64 or y.shape!=(n,t) or not np.isin(y,[0.,1.]).all()):
        raise ValueError('y must be float64 binary [N,T], including padding placeholders')

def sigmoid(a):
    # sigmoid(a) = exp(-logaddexp(0,-a)), stable for either sign.
    return np.exp(-np.logaddexp(np.float64(0),-a))

def full_bptt(p,x,lengths,y,last_only=False):
    validate(p,x,lengths,y)
    if not isinstance(last_only,bool): raise ValueError("last_only must be bool")
    n,t,d=x.shape; hdim=p['b'].size
    mask=np.arange(t)[None,:]<lengths[:,None]
    h=np.zeros((n,t+1,hdim),dtype=np.float64)
    cand=np.zeros((n,t,hdim),dtype=np.float64)
    pre=np.zeros_like(cand)
    for k in range(t):
        pre[:,k]=x[:,k]@p['U'].T+h[:,k]@p['W'].T+p['b']
        cand[:,k]=np.tanh(pre[:,k])
        h[:,k+1]=np.where(mask[:,k,None],cand[:,k],h[:,k])
    z=h[:,1:]@p['v']+p['c']; prob=sigmoid(z)
    loss_mask=(np.arange(t)[None,:]==(lengths-1)[:,None]) if last_only else mask
    norm=int(loss_mask.sum())
    terms=np.logaddexp(np.float64(0),z)-y*z
    loss=float((terms*loss_mask).sum()/norm)
    dz=(prob-y)*loss_mask/norm
    grads={k:np.zeros_like(p[k]) for k in NAMES}
    contributions={k:np.zeros((n,t)+p[k].shape,dtype=np.float64) for k in NAMES}
    dh_next=np.zeros((n,hdim),dtype=np.float64)
    delta=np.zeros_like(cand); dh=np.zeros_like(cand); direct=np.zeros_like(cand); future=np.zeros_like(cand)
    dx=np.zeros_like(x)
    for k in range(t-1,-1,-1):
        direct[:,k]=dz[:,k,None]*p['v']; future[:,k]=dh_next
        dh[:,k]=direct[:,k]+dh_next
        delta[:,k]=dh[:,k]*(1-cand[:,k]**2)*mask[:,k,None]
        for i in range(n):
            contributions['U'][i,k]=np.outer(delta[i,k],x[i,k])
            contributions['W'][i,k]=np.outer(delta[i,k],h[i,k])
            contributions['b'][i,k]=delta[i,k]
            contributions['v'][i,k]=dz[i,k]*h[i,k+1]
            contributions['c'][i,k]=dz[i,k]
        dx[:,k]=delta[:,k]@p['U']
        dh_next=delta[:,k]@p['W']+dh[:,k]*(~mask[:,k,None])
    for key in NAMES: grads[key]=np.asarray(contributions[key].sum(axis=(0,1)),dtype=np.float64)
    return {'loss':loss,'h':h,'a':pre,'candidate':cand,'z':z,'prob':prob,'mask':mask,'terms':terms,'dz':dz,'dh':dh,'direct':direct,'future':future,'delta':delta,'dx':dx,'dh0':dh_next,'grads':grads,'contributions':contributions,'normalizer':norm,'loss_mask':loss_mask}

def fixture():
    p={'U':np.array([[.4]],dtype=np.float64),'W':np.array([[.6]],dtype=np.float64),'b':np.array([.1],dtype=np.float64),'v':np.array([.7],dtype=np.float64),'c':np.array(-.2,dtype=np.float64)}
    x=np.array([[[1.],[0.],[-1.]],[[-1.],[.5],[0.]]],dtype=np.float64)
    y=np.array([[1.,0.,1.],[0.,1.,0.]],dtype=np.float64)
    lengths=np.array([3,2],dtype=np.int64)
    return p,x,lengths,y

def scalar_fixture_reference(p,x,lengths,y):
    """Scalar-only third forward/loss route (math functions, no array dot)."""
    import math
    losses=[]; traces=[]
    for i in range(len(lengths)):
        h=0.; row=[]
        for t in range(int(lengths[i])):
            h=math.tanh(float(p['U'][0,0])*float(x[i,t,0])+float(p['W'][0,0])*h+float(p['b'][0]))
            z=float(p['v'][0])*h+float(p['c'])
            losses.append(max(z,0.)+math.log1p(math.exp(-abs(z)))-float(y[i,t])*z)
            row.append(h)
        traces.append(row)
    return sum(losses)/len(losses),traces

def finite_difference(p,x,lengths,y,eps=1e-6):
    if not (np.isfinite(eps) and 1e-8<=eps<=1e-3): raise ValueError('eps outside [1e-8,1e-3]')
    out={}
    for key in NAMES:
        g=np.zeros_like(p[key])
        for ix in np.ndindex(p[key].shape):
            plus={k:v.copy() for k,v in p.items()};minus={k:v.copy() for k,v in p.items()}
            plus[key][ix]+=eps;minus[key][ix]-=eps
            g[ix]=(full_bptt(plus,x,lengths,y)['loss']-full_bptt(minus,x,lengths,y)['loss'])/(2*eps)
        out[key]=g
    return out
