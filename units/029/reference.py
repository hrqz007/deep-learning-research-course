"""Independent NumPy matrix reference and scalar-loop finite differences.

Same row-sample equations and fixed data as unit027, self-contained implementation.
No torch import, no autograd call, no cross-unit import. Restricted teaching domain.
"""
import math
import numpy as np


def real(value, name, ndim=None):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
        raise ValueError(name + ': real numeric values required')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        b = a.astype(np.float64, copy=True)
    if not b.size or not np.isfinite(b).all() or np.any((a != 0) & (b == 0)) or np.max(np.abs(b)) > 10:
        raise ValueError(name + ': finite magnitude <=10, no nonzero-to-zero conversion required')
    if ndim is not None and b.ndim != ndim:
        raise ValueError(name + ': invalid rank')
    return b


def validate(X, y, layers, activation='tanh', reduction='mean'):
    if activation not in ('tanh', 'relu', 'linear') or reduction not in ('mean', 'sum'):
        raise ValueError('invalid activation/reduction')
    X = real(X, 'X', 2)
    if not 1 <= len(X) <= 128 or not 1 <= X.shape[1] <= 16:
        raise ValueError('invalid batch/input size')
    if not isinstance(layers, (list, tuple)) or not 1 <= len(layers) <= 5:
        raise ValueError('1..5 layers required')
    params=[]; width=X.shape[1]
    for layer in layers:
        if not isinstance(layer,dict) or set(layer) != {'W','b'}:
            raise ValueError('each layer requires W,b')
        W,b=real(layer['W'],'W',2),real(layer['b'],'b',1)
        if W.shape != (width,b.size) or not 1 <= b.size <= 16:
            raise ValueError('incompatible layer shapes')
        params.append({'W':W,'b':b}); width=b.size
    yy=np.asarray(y)
    if width < 2 or yy.dtype.kind not in 'iu' or yy.shape != (len(X),) or np.any(yy<0) or np.any(yy>=width) or any(isinstance(v,(bool,np.bool_)) for v in np.asarray(y,dtype=object).flat):
        raise ValueError('integer class indices of shape (n,) required')
    return X, yy.astype(np.int64), params


def numpy_reference(X, y, layers, activation='tanh', reduction='mean'):
    X,y,layers=validate(X,y,layers,activation,reduction)
    values={'H0':X}; H=[X]; Z=[]
    for j,p in enumerate(layers,1):
        z=H[-1] @ p['W'] + p['b']; Z.append(z); values[f'Z{j}']=z
        if not np.isfinite(z).all() or np.max(np.abs(z)) > 1e6: raise ValueError('affine output exceeds domain')
        if j < len(layers):
            h=np.tanh(z) if activation=='tanh' else np.maximum(z,0) if activation=='relu' else z.copy()
            H.append(h); values[f'H{j}']=h
    shift=Z[-1]-Z[-1].max(axis=1,keepdims=True)
    e=np.exp(shift); rest=e.copy(); rest[np.arange(len(y)),np.argmax(Z[-1],axis=1)]=0
    r=rest.sum(axis=1); p=e/(1+r[:,None]); losses=-shift[np.arange(len(y)),y]+np.log1p(r)
    values['p']=p;values['losses']=losses;values['loss']=np.asarray(losses.mean() if reduction=='mean' else losses.sum())
    d=p.copy(); d[np.arange(len(y)),y]=0;d[np.arange(len(y)),y]=-d.sum(axis=1)
    if reduction=='mean':d/=len(y)
    for j in range(len(layers)-1,-1,-1):
        values[f'dZ{j+1}']=d.copy(); values[f'dW{j+1}']=H[j].T@d;values[f'db{j+1}']=d.sum(axis=0)
        up=d@layers[j]['W'].T; values[f'dH{j}']=up
        if j:
            slope=1-H[j]**2 if activation=='tanh' else (Z[j-1]>0) if activation=='relu' else np.ones_like(up)
            d=up*slope
    if any(not np.isfinite(a).all() for a in values.values()):raise ValueError('nonfinite reference output')
    return values


def scalar_loss(X,y,layers,activation='tanh',reduction='mean'):
    """Python scalar loops, independent of NumPy matrix forward/backward."""
    X,y,layers=validate(X,y,layers,activation,reduction)
    losses=[]
    for row,target in zip(X,y):
        h=row.tolist()
        for j,layer in enumerate(layers):
            z=[math.fsum(a*float(layer['W'][i,k]) for i,a in enumerate(h))+float(layer['b'][k]) for k in range(len(layer['b']))]
            if any(not math.isfinite(a) or abs(a)>1e6 for a in z):raise ValueError('scalar affine output exceeds domain')
            h=([math.tanh(a) for a in z] if activation=='tanh' else [max(0,a) for a in z] if activation=='relu' else z) if j < len(layers)-1 else z
        m=max(h); e=[math.exp(a-m) for a in h]; e[h.index(m)]=0
        losses.append((m-h[int(target)])+math.log1p(math.fsum(e)))
    return math.fsum(losses)/(len(y) if reduction=='mean' else 1)


def finite_differences(X,y,layers,activation='tanh',h=1e-5):
    X,y,layers=validate(X,y,layers,activation,'mean')
    if isinstance(h,bool) or not isinstance(h,(float,int)) or not math.isfinite(h) or not 1e-8 <= h <= 1e-2:raise ValueError('h outside [1e-8,1e-2]')
    base=numpy_reference(X,y,layers,activation); rows=[]
    for name,a in [('H0',X)]+[(f'{key}{j+1}',p[key]) for j,p in enumerate(layers) for key in ('W','b')]:
        for idx in np.ndindex(a.shape):
            step=h*max(1,abs(a[idx])); pp=[{k:v.copy() for k,v in p.items()} for p in layers];pm=[{k:v.copy() for k,v in p.items()} for p in layers]; xp=X.copy();xm=X.copy()
            if name=='H0': xp[idx]+=step;xm[idx]-=step
            else:pp[int(name[1:])-1][name[0]][idx]+=step;pm[int(name[1:])-1][name[0]][idx]-=step
            numeric=(scalar_loss(xp,y,pp,activation)-scalar_loss(xm,y,pm,activation))/(2*step)
            ref=float(base['d'+name][idx]);rows.append({'tensor':'d'+name,'index':','.join(map(str,idx)),'reference':ref,'finite_difference':numeric,'absolute_error':abs(ref-numeric)})
    return rows
