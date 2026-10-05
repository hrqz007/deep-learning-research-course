"""Framework-independent residual-block reference; deliberately small CPU arrays."""
from fractions import Fraction as Q
import numpy as np


def array4(x, name):
    x = np.asarray(x)
    if x.ndim != 4 or min(x.shape) < 1 or x.dtype.kind != 'f' or not np.isfinite(x).all():
        raise ValueError(f'{name}: require nonempty finite real floating NCHW/OIHW array')
    return x.astype(np.float64, copy=False)


def integer(x, name, minimum):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def conv(x, w, stride=1, padding=0):
    x, w = array4(x,'x'), array4(w,'w')
    integer(stride,'stride',1); integer(padding,'padding',0)
    if x.shape[1] != w.shape[1]: raise ValueError('input-channel mismatch')
    n,ci,h,ww=x.shape; co,_,kh,kw=w.shape
    oh=(h+2*padding-kh)//stride+1; ow=(ww+2*padding-kw)//stride+1
    if min(oh,ow)<1: raise ValueError('kernel has no valid output position')
    xp=np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    y=np.zeros((n,co,oh,ow))
    for b in range(n):
        for o in range(co):
            for i in range(oh):
                for j in range(ow):
                    y[b,o,i,j]=np.sum(xp[b,:,i*stride:i*stride+kh,j*stride:j*stride+kw]*w[o])
    if not np.isfinite(y).all(): raise ValueError("float64 convolution overflow; reduce input/weight scale")
    return y


def conv_backward(x,w,g,stride=1,padding=0):
    y=conv(x,w,stride,padding); g=array4(g,'upstream')
    if y.shape != g.shape: raise ValueError('upstream shape must equal output exactly')
    x=array4(x,'x');w=array4(w,'w')
    xp=np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    dxp=np.zeros_like(xp);dw=np.zeros_like(w);kh,kw=w.shape[2:]
    for b in range(y.shape[0]):
        for o in range(y.shape[1]):
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    sl=(b,slice(None),slice(i*stride,i*stride+kh),slice(j*stride,j*stride+kw))
                    dw[o]+=g[b,o,i,j]*xp[sl]
                    dxp[sl]+=g[b,o,i,j]*w[o]
    dx=dxp if padding==0 else dxp[:,:,padding:-padding,padding:-padding]
    if not np.isfinite(dx).all() or not np.isfinite(dw).all(): raise ValueError("float64 backward overflow")
    return dx,dw


def residual(x,w1,w2,p=None,stride=1,upstream=None):
    """No BN, no biases, no final gate; 3x3 residual and optional 1x1 projection.
    Exact addition only. Reference domain: stride 1 or 2, groups/dilation 1.
    """
    x=array4(x,'x');w1=array4(w1,'w1');w2=array4(w2,'w2')
    integer(stride,'stride',1)
    if stride not in (1,2): raise ValueError('reference supports stride 1 or 2 only')
    if w1.shape[2:]!=(3,3) or w2.shape[2:]!=(3,3): raise ValueError('residual kernels must be 3x3')
    if w2.shape[0]!=w1.shape[0] or w2.shape[1]!=w1.shape[0]: raise ValueError('residual channel mismatch')
    z=conv(x,w1,stride,1);a=np.maximum(z,0);f=conv(a,w2,1,1)
    if p is not None:
        p=array4(p,'projection')
        if p.shape!=(w1.shape[0],x.shape[1],1,1): raise ValueError('projection must be Cout,Cin,1,1')
    s=x if p is None else conv(x,p,stride,0)
    if f.shape!=s.shape: raise ValueError('shortcut shape mismatch; broadcasting is forbidden')
    y=f+s
    if not np.isfinite(y).all(): raise ValueError("residual addition overflow")
    if upstream is None: return y
    g=array4(upstream,'upstream')
    if g.shape!=y.shape: raise ValueError('upstream shape mismatch')
    da,dw2=conv_backward(a,w2,g,1,1)
    dz=da*(z>0)
    dx_f,dw1=conv_backward(x,w1,dz,stride,1)
    if p is None: dx_s=g;dp=None
    else: dx_s,dp=conv_backward(x,p,g,stride,0)
    dx=dx_s+dx_f
    if not np.isfinite(dx).all(): raise ValueError('residual input-gradient addition overflow')
    return dict(y=y,z=z,dx=dx,dx_skip=dx_s,dx_residual=dx_f,dw1=dw1,dw2=dw2,dp=dp)


def hand_round(theta):
    """All arithmetic exact; two samples, two spatial positions, six parameters."""
    w,b,v,c,a,d=theta; X=[[Q(-1),Q(1)],[Q(1),Q(2)]]; Y=[Q(0),Q(1)]
    rows=[]; grads=[]
    for x,t in zip(X,Y):
        z=[w*q+b for q in x];h=[max(Q(0),q) for q in z]
        f=[v*q+c for q in h];s=[q+r for q,r in zip(x,f)];m=sum(s)/2
        pred=a*m+d;e=pred-t;u=e/2;gm=u*a;gs=gm/2
        dz=[gs*v*int(q>0) for q in z]
        grad=[sum(dz[j]*x[j] for j in range(2)),sum(dz),gs*sum(h),2*gs,u*m,u]
        grads.append(grad)
        rows.append(dict(x=x,target=t,z=z,h=h,f=f,s=s,m=m,pred=pred,error=e,loss=e*e/2,
                         weighted_loss=e*e/4,u=u,gm=gm,gs=gs,dz=dz,gradient=grad,
                         dx_skip=[gs,gs],dx_residual=[q*w for q in dz],dx=[gs+q*w for q in dz]))
    gradient=[sum(g[j] for g in grads) for j in range(6)]
    return dict(theta=theta,rows=rows,loss=sum(r['weighted_loss'] for r in rows),gradient=gradient,
                next_theta=[t-Q(1,10)*g for t,g in zip(theta,gradient)])


def hand_ledger():
    first=hand_round([Q(1,2),Q(1,4),Q(2,5),Q(-1,10),Q(4,5),Q(1,10)])
    second=hand_round(first['next_theta']);third=hand_round(second['next_theta'])
    def numeric(v):
        if isinstance(v,Q): return float(v)
        if isinstance(v,list): return [numeric(q) for q in v]
        if isinstance(v,dict): return {k:numeric(q) for k,q in v.items()}
        return v
    return dict(rounds=numeric([first,second,third]),exact_initial_gradient=[str(q) for q in first['gradient']],
                exact_initial_loss=str(first['loss']),parameter_order=['w','b','v','c','a','d'])
