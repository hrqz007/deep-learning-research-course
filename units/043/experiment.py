"""DL043: explicit NCHW cross-correlation, gradients and bounded symmetry tests.

Offline CPU teaching implementation. This is intentionally not a fast Conv2d.
Public API supports real finite nonempty arrays, groups=1, symmetric zero padding,
positive integer stride/dilation pairs; outputs and padded arrays <= 1e6 elements,
and <= 2e7 multiply-add sites. Unsupported modes raise rather than silently change.
"""
from pathlib import Path
from fractions import Fraction as F
from numbers import Integral
import argparse, hashlib, json, os, platform, tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
MAX_ELEMENTS=1_000_000
MAX_MULTIPLIES=20_000_000

def pair(value, name, minimum):
    if isinstance(value,Integral) and not isinstance(value,(bool,np.bool_)):
        value=(int(value),int(value))
    if not isinstance(value,(tuple,list)) or len(value)!=2:
        raise ValueError(f'{name} must be an integer or pair of integers')
    if any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) or v<minimum for v in value):
        raise ValueError(f'{name} values must be integers >= {minimum}')
    return tuple(int(v) for v in value)

def array(value,name,ndim):
    a=np.asarray(value)
    if a.ndim!=ndim or any(s==0 for s in a.shape): raise ValueError(f'{name}: expected nonempty {ndim}D array')
    if a.size>MAX_ELEMENTS: raise ValueError(f'{name}: teaching size limit exceeded')
    if a.dtype.kind not in 'fiu': raise ValueError(f'{name}: real numeric values required (not bool/complex/object)')
    a=a.astype(np.float64,copy=False)
    if not np.isfinite(a).all(): raise ValueError(f'{name}: all values must be finite float64')
    return a

def checked(x,k,b,stride,padding,dilation):
    x=array(x,'x',4);k=array(k,'kernel',4);b=array(b,'bias',1)
    s=pair(stride,'stride',1);p=pair(padding,'padding',0);d=pair(dilation,'dilation',1)
    n,ci,h,w=x.shape;co,ki,kh,kw=k.shape
    if ci!=ki or b.shape!=(co,):raise ValueError('input channels or bias shape mismatch')
    eh,ew=d[0]*(kh-1)+1,d[1]*(kw-1)+1
    oh=(h+2*p[0]-eh)//s[0]+1;ow=(w+2*p[1]-ew)//s[1]+1
    if oh<=0 or ow<=0:raise ValueError('effective kernel does not fit padded input')
    if n*ci*(h+2*p[0])*(w+2*p[1])>MAX_ELEMENTS or n*co*oh*ow>MAX_ELEMENTS:
        raise ValueError('padded/output teaching size limit exceeded')
    if n*co*oh*ow*ci*kh*kw>MAX_MULTIPLIES:raise ValueError('teaching operation limit exceeded')
    return x,k,b,s,p,d,(n,co,oh,ow)

def conv_forward(x,k,b,*,stride=1,padding=0,dilation=1):
    x,k,b,s,p,d,shape=checked(x,k,b,stride,padding,dilation)
    xp=np.pad(x,((0,0),(0,0),(p[0],p[0]),(p[1],p[1])))
    y=np.empty(shape,dtype=np.float64);kh,kw=k.shape[-2:]
    with np.errstate(over='raise',invalid='raise'):
        for n in range(shape[0]):
            for o in range(shape[1]):
                for r in range(shape[2]):
                    for c in range(shape[3]):
                        patch=xp[n,:,r*s[0]:r*s[0]+d[0]*(kh-1)+1:d[0],c*s[1]:c*s[1]+d[1]*(kw-1)+1:d[1]]
                        y[n,o,r,c]=np.sum(patch*k[o])+b[o]
    if not np.isfinite(y).all():raise FloatingPointError('non-finite forward output')
    return y

def conv_backward(x,k,b,upstream,*,stride=1,padding=0,dilation=1):
    x,k,b,s,p,d,shape=checked(x,k,b,stride,padding,dilation)
    u=array(upstream,'upstream',4)
    if u.shape!=shape:raise ValueError('upstream shape must exactly equal forward output')
    xp=np.pad(x,((0,0),(0,0),(p[0],p[0]),(p[1],p[1])))
    dxp=np.zeros_like(xp);dk=np.zeros_like(k);db=np.zeros_like(b);kh,kw=k.shape[-2:]
    with np.errstate(over='raise',invalid='raise'):
        for n in range(shape[0]):
            for o in range(shape[1]):
                for r in range(shape[2]):
                    for c in range(shape[3]):
                        rs=slice(r*s[0],r*s[0]+d[0]*(kh-1)+1,d[0]);cs=slice(c*s[1],c*s[1]+d[1]*(kw-1)+1,d[1])
                        g=u[n,o,r,c]
                        dk[o]+=g*xp[n,:,rs,cs]
                        dxp[n,:,rs,cs]+=g*k[o]
                        db[o]+=g
    dx=dxp[:,:,p[0]:p[0]+x.shape[2],p[1]:p[1]+x.shape[3]]
    if any(not np.isfinite(a).all() for a in (dx,dk,db)):raise FloatingPointError('non-finite gradient')
    return dx,dk,db

def hand_ledger():
    """Independent exact rational arithmetic, not conv_forward/backward."""
    xs=[[[1,0,2],[0,1,0],[2,0,1]],[[0,1,0],[1,0,1],[0,1,0]]]
    target=[F(1,2),F(1)];theta=[F(1,2),F(-1,4),F(1,4),F(1,2),F(1,10)]
    records=[]
    for step in range(3):
        samples=[];total=[F(0)]*5
        for x,t in zip(xs,target):
            z=[];h=[];patches=[]
            for r in range(2):
                for c in range(2):
                    v=[F(x[r][c]),F(x[r][c+1]),F(x[r+1][c]),F(x[r+1][c+1])]
                    patches.append(v);z.append(sum(a*b for a,b in zip(v,theta[:4]))+theta[4]);h.append(max(F(0),z[-1]))
            pred=sum(h)/4;res=pred-t;loss=res*res/2;u=[res/F(8) if a>0 else F(0) for a in z]
            contrib=[sum(g*v[j] for g,v in zip(u,patches)) for j in range(4)]+[sum(u)]
            total=[a+b for a,b in zip(total,contrib)]
            dx=[[F(0)]*3 for _ in range(3)]
            for i,g in enumerate(u):
                r,c=divmod(i,2)
                for j,w in enumerate(theta[:4]):
                    a,b=divmod(j,2);dx[r+a][c+b]+=g*w
            samples.append(dict(z=z,h=h,pred=pred,residual=res,loss=loss,upstream=u,patches=patches,contribution=contrib,dx=dx))
        records.append(dict(step=step,theta=theta.copy(),samples=samples,loss=sum(s['loss'] for s in samples)/2,gradient=total))
        if step<2:theta=[v-F(1,5)*g for v,g in zip(theta,total)]
    def encode(v):
        if isinstance(v,F):return {'exact':str(v),'float':float(v)}
        if isinstance(v,list):return [encode(x) for x in v]
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
        return v
    return encode(records)

def shift_zero(x,dy,dx):
    """Integer shift on last axes, zero-fill/crop, positive = down/right."""
    a=np.asarray(x);h,w=a.shape[-2:]
    if not isinstance(dy,Integral) or not isinstance(dx,Integral):raise ValueError('shift must be integer')
    out=np.zeros_like(a)
    r0,r1=max(0,dy),min(h,h+dy);c0,c1=max(0,dx),min(w,w+dx)
    if r1>r0 and c1>c0:out[...,r0:r1,c0:c1]=a[...,r0-dy:r1-dy,c0-dx:c1-dx]
    return out

def circular_corr(x,k,stride=1):
    """Separate full-period reference: odd one-channel kernel, no bias."""
    h,w=x.shape;kh,kw=k.shape
    y=np.zeros_like(x,dtype=float)
    for a in range(kh):
        for b in range(kw):y+=k[a,b]*np.roll(x,(kh//2-a,kw//2-b),(0,1))
    return y[::stride,::stride]

def symmetry_audit(data):
    x=np.array(data['symmetry_input'],float);k=np.array(data['symmetry_kernel'],float)
    def zero(v,s=1):return conv_forward(v[None,None],k[None,None],np.zeros(1),padding=1,stride=s)[0,0]
    rows=[];arrays={}
    def record(name,a,b,mask=None,note=''):
        diff=np.abs(a-b); selected=diff if mask is None else diff[mask]
        rows.append(dict(case=name,max_abs=float(selected.max()),rmse=float(np.sqrt(np.mean(selected**2))),coordinates=int(selected.size),note=note))
        arrays[name]={'left':a.tolist(),'right':b.tolist(),'difference':diff.tolist()}
    y=zero(x);sx=shift_zero(x,0,1);left=zero(sx);right=shift_zero(y,0,1)
    record('zero_stride1_full',left,right,note='finite zero crop; failure at boundary retained')
    mask=np.zeros_like(x,dtype=bool);mask[1:-1,2:-1]=True
    record('zero_stride1_interior',left,right,mask,note='both receptive fields wholly inside observed image')
    cy=circular_corr(x,k);csx=np.roll(x,1,axis=1)
    record('circular_stride1',circular_corr(csx,k),np.roll(cy,1,axis=1),note='periodic domain; exact integer translation action')
    record('circular_stride2_shift2',circular_corr(np.roll(x,2,axis=1),k,2),np.roll(circular_corr(x,k,2),1,axis=1),note='even width; 2 input columns = 1 output column')
    # One input-pixel shift is not an integer output-pixel shift at stride 2.
    record('circular_stride2_shift1_vs_shift0',circular_corr(csx,k,2),circular_corr(x,k,2),note='diagnostic only: no integer output shift represents half a pixel')
    record('circular_stride2_shift1_vs_shift1',circular_corr(csx,k,2),np.roll(circular_corr(x,k,2),1,axis=1),note='diagnostic; also not the correct group action')
    relu=lambda v:np.maximum(v,0)
    record('circular_relu',relu(circular_corr(csx,k)),np.roll(relu(cy),1,axis=1),note='pointwise shared ReLU preserves circular equivariance')
    weights=np.arange(x.size,dtype=float).reshape(x.shape)/x.size
    heads=dict(circular_gap=[float(relu(cy).mean()),float(relu(circular_corr(csx,k)).mean())],
               zero_gap=[float(relu(y).mean()),float(relu(left).mean())],
               circular_position_weighted=[float(np.sum(weights*relu(cy))),float(np.sum(weights*relu(circular_corr(csx,k))))])
    return dict(rows=rows,heads=heads,arrays=arrays)

def fit_kernel(data):
    """Fixed offline supervised system identification, no hyperparameter selection."""
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x=np.array(data['train_x'],float);target=np.array(data['train_y'],float)
    test_x=np.array(data['test_x'],float);test_y=np.array(data['test_y'],float)
    k=np.zeros((1,1,3,3));b=np.zeros(1);eta=0.1;steps=80;trace=[]
    xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(target,dtype=torch.float64)
    kt=torch.zeros((1,1,3,3),dtype=torch.float64,requires_grad=True);bt=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    max_gap=0.
    for step in range(steps+1):
        pred=conv_forward(x,k,b);r=pred-target;loss=float(np.mean(r*r)/2)
        tp=torch.nn.functional.conv2d(xt,kt,bt);tl=((tp-yt)**2).mean()/2
        max_gap=max(max_gap,float(np.max(np.abs(pred-tp.detach().numpy()))))
        trace.append(dict(step=step,loss=loss,kernel=k.ravel().tolist(),bias=float(b[0])))
        if step==steps:break
        dx,gk,gb=conv_backward(x,k,b,r/r.size)
        tl.backward()
        max_gap=max(max_gap,float(np.max(np.abs(gk-kt.grad.numpy()))),float(np.max(np.abs(gb-bt.grad.numpy()))))
        k=k-eta*gk;b=b-eta*gb
        with torch.no_grad():kt-=eta*kt.grad;bt-=eta*bt.grad
        kt.grad=None;bt.grad=None
    pred=conv_forward(test_x,k,b);baseline=np.full_like(test_y,target.mean())
    # Independent closed-form patch design and stable least-squares reference.
    rows=[]
    for image in x[:,0]:
        for r in range(4):
            for c in range(4):rows.append(np.r_[image[r:r+3,c:c+3].ravel(),1.])
    design=np.array(rows);coef,res,rank,svals=np.linalg.lstsq(design,target.ravel(),rcond=None)
    return dict(protocol={'train_images':len(x),'test_images':len(test_x),'image_size':[6,6],'valid_outputs_per_image':16,'steps':steps,'learning_rate':eta,'parameters':10,'optimizer':'full-batch SGD, all output pixels mean half-MSE','seed_train':4301,'seed_test':4302,'selection':'none; fixed protocol','noise_std':0.05},
        train_half_mse=trace[-1]['loss'],test_half_mse=float(np.mean((pred-test_y)**2)/2),constant_test_half_mse=float(np.mean((baseline-test_y)**2)/2),
        true_kernel=data['true_kernel'],learned_kernel=k[0,0].tolist(),bias=float(b[0]),trace=trace,
        torch_max_abs_gap=max_gap,least_squares_parameters=coef.tolist(),least_squares_rank=int(rank),
        least_squares_train_half_mse=float(np.mean((design@coef-target.ravel())**2)/2),
        gd_to_least_squares_max_abs=float(np.max(np.abs(np.r_[k.ravel(),b]-coef))),
        test_prediction=pred.tolist(),test_target=test_y.tolist())

def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def run(output=None):
    data=json.loads((ROOT/'data/experiment_data.json').read_text())
    expected=json.loads((ROOT/'data/data_manifest.json').read_text())['experiment_data_sha256']
    actual=hashlib.sha256((ROOT/'data/experiment_data.json').read_bytes()).hexdigest()
    if actual!=expected:raise ValueError('data hash mismatch; do not silently change the fixed protocol')
    results={'hand':hand_ledger(),'symmetry':symmetry_audit(data),'learning':fit_kernel(data)}
    results['environment']={'python':platform.python_version(),'numpy':np.__version__,'torch':__import__('torch').__version__,'device':'CPU float64','data_sha256':actual}
    if output is not None:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        payload=json_bytes(results)
        with tempfile.NamedTemporaryFile(dir=out,delete=False) as stream:stream.write(payload);tmp=Path(stream.name)
        os.replace(tmp,out/'results.json')
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs');args=parser.parse_args()
    r=run(args.output)
    print(json.dumps({'hand_losses':[s['loss']['float'] for s in r['hand']],'symmetry':r['symmetry']['rows'],'learning':{k:v for k,v in r['learning'].items() if k not in ['trace','test_prediction','test_target']}},ensure_ascii=False,indent=2))
