"""Numerical and behavioral checks. No Python assert: active under python -O."""
import json,tempfile,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from reference import fixture,full_bptt,finite_difference,scalar_fixture_reference,NAMES
from generate_data import make_split
from experiment import DT,DEV,tensor,params_from_numpy,numpy_params,forward,token_loss,init_params,step,input_probe,ROOT

def close(a,b,msg,atol=2e-9,rtol=2e-8):
    if not np.allclose(a,b,atol=atol,rtol=rtol): raise RuntimeError(msg+': max error '+str(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
def require(v,msg):
    if not v:raise RuntimeError(msg)
def rejected(fn,msg):
    try:fn()
    except (ValueError,TypeError):return
    raise RuntimeError('failed to reject: '+msg)
def torch_grad(p,x,l,y):
    tp=params_from_numpy(p);xx=tensor(x,True);tl=torch.tensor(l,dtype=torch.int64,device=DEV)
    z,h=forward(tp,xx,tl);loss=token_loss(z,tensor(y),tl);loss.backward()
    return float(loss.detach()),h.detach().numpy(),{k:tp[k].grad.numpy() for k in NAMES},xx.grad.numpy()

def main():
    checks=[];p,x,l,y=fixture();ref=full_bptt(p,x,l,y)
    sl,sh=scalar_fixture_reference(p,x,l,y);close(ref['loss'],sl,'scalar forward loss',1e-14,0)
    for i in range(2):close(ref['h'][i,1:l[i]+1,0],sh[i],'scalar states',1e-14,0)
    checks.append('independent math-scalar forward and BCE')
    rng=np.random.default_rng(503)
    for dims in [(2,3,1,1),(3,4,2,3)]:
        n,t,d,h=dims
        if dims==(2,3,1,1):pp,xx,ll,yy=p,x,l,y
        else:
            pp=numpy_params(init_params(22,d,h,.8));xx=rng.uniform(-.4,.4,size=(n,t,d)).astype(np.float64);ll=np.array([4,2,1],dtype=np.int64);yy=rng.integers(0,2,size=(n,t)).astype(np.float64)
        rr=full_bptt(pp,xx,ll,yy);loss,hs,gr,dx=torch_grad(pp,xx,ll,yy)
        close(loss,rr['loss'],'loss');close(hs,rr['h'][:,1:],'states');close(dx,rr['dx'],'input gradient')
        for k in NAMES:close(gr[k],rr['grads'][k],k+' autograd')
        for eps in [1e-4,1e-5,1e-6]:
            fd=finite_difference(pp,xx,ll,yy,eps)
            for k in NAMES:close(fd[k],rr['grads'][k],k+' finite diff',3e-8,3e-7)
        # Input finite differences, including all padded entries.
        fd_x=np.zeros_like(xx)
        for ix in np.ndindex(xx.shape):
            xp=xx.copy();xm=xx.copy();xp[ix]+=1e-6;xm[ix]-=1e-6
            fd_x[ix]=(full_bptt(pp,xp,ll,yy)['loss']-full_bptt(pp,xm,ll,yy)['loss'])/2e-6
        close(fd_x,rr['dx'],'input finite difference',2e-9,2e-7)
        # Sample-local loss is mean over its own valid tokens; weight it back.
        for k in NAMES:
            acc=np.zeros_like(pp[k])
            for i in range(n):acc+=full_bptt(pp,xx[i:i+1],ll[i:i+1],yy[i:i+1])['grads'][k]*ll[i]/ll.sum()
            close(acc,rr['grads'][k],'weighted sample sum '+k)
    checks.append('scalar and matrix NumPy BPTT vs autograd and central finite differences at three epsilons; every parameter and input')
    # Time-untied W gradients must sum to tied W gradient.
    tp=params_from_numpy(p);ws=[tensor(p['W'],True) for _ in range(3)];h=torch.zeros((2,1),dtype=DT,device=DEV);zs=[]
    for t in range(3):
        h=torch.where(torch.tensor(t<l,dtype=torch.bool,device=DEV)[:,None],torch.tanh(tensor(x[:,t])@tp['U'].T+h@ws[t].T+tp['b']),h);zs.append(h@tp['v']+tp['c'])
    token_loss(torch.stack(zs,dim=1),tensor(y),torch.tensor(l,dtype=torch.int64,device=DEV)).backward()
    close(sum(w.grad.numpy() for w in ws),ref['grads']['W'],'untied-to-tied sum');checks.append('shared recurrent weight equals sum of independently untied temporal copies')
    # Library nn.RNN mapping: merge its two biases into our single b.
    tp=numpy_params(init_params(29,2,3,.8));xx=rng.uniform(-.3,.3,size=(3,4,2)).astype(np.float64);ll=np.array([4,2,1],dtype=np.int64)
    cell=torch.nn.RNN(2,3,batch_first=True,nonlinearity='tanh',dtype=DT,device=DEV)
    with torch.no_grad():
        cell.weight_ih_l0.copy_(tensor(tp['U']));cell.weight_hh_l0.copy_(tensor(tp['W']));cell.bias_ih_l0.copy_(tensor(tp['b']));cell.bias_hh_l0.zero_()
    _,hs=forward(params_from_numpy(tp),tensor(xx),torch.tensor(ll,dtype=torch.int64,device=DEV))
    for i,L in enumerate(ll):
        expected,_=cell(tensor(xx[i:i+1,:L]))
        close(expected.detach().numpy(),hs[i:i+1,:L].detach().numpy(),'official nn.RNN forward mapping')
    checks.append('torch.nn.RNN official cell mapping with explicit dtype/device and merged bias')
    # Padded inputs and labels can change without effect, but valid values remain bounded.
    xp=x.copy();yp=y.copy();xp[1,2,0]=9.;yp[1,2]=1.
    rr=full_bptt(p,xp,l,yp);close(rr['loss'],ref['loss'],'padding loss')
    for k in NAMES:close(rr['grads'][k],ref['grads'][k],'padding gradient '+k)
    close(rr['dx'][1,2],0.,'padding input gradient');checks.append('state freeze and loss mask make padding inert')
    tp=params_from_numpy(p);z,_=forward(tp,tensor(x),torch.tensor(l,dtype=torch.int64,device=DEV))
    for i in range(2):
        zz,_=forward(tp,tensor(x[i:i+1]),torch.tensor(l[i:i+1],dtype=torch.int64,device=DEV));close(zz.detach().numpy(),z[i:i+1].detach().numpy(),'independent state reset')
    perm=[1,0];zz,_=forward(tp,tensor(x[perm]),torch.tensor(l[perm],dtype=torch.int64,device=DEV));close(zz.detach().numpy(),z[perm].detach().numpy(),'batch permutation')
    checks.append('batch vs independent sequences and batch permutation invariance')
    xx,yy=make_split(8,8,5101);tp=init_params(7);ll=torch.full((8,),9,dtype=torch.int64,device=DEV)
    zf,_=forward(tp,tensor(xx),ll);zt,_=forward(tp,tensor(xx),ll,4);close(zf.detach().numpy(),zt.detach().numpy(),'detach forward equality',0,0)
    gfull=input_probe(tp,tensor(xx));gtr=input_probe(tp,tensor(xx),4)
    require(max(gfull[:8])>1e-10,'full has early gradient');close(gtr[:8],np.zeros(8),'truncated exact zero');require(gtr[8]>0,'last block retained');checks.append('detach identical forward values; zero early gradient across fixed block boundary')
    # Real training step vs independent NumPy terminal-loss SGD and global clipping.
    for clip in [None,.001]:
        tp=init_params(19,3,3,.8);pp=numpy_params(tp);yt=np.broadcast_to(yy[:,None],(8,9)).copy()
        rr=full_bptt(pp,xx,np.full(8,9,dtype=np.int64),yt,last_only=True)
        gn=np.sqrt(sum(float(np.sum(v*v)) for v in rr['grads'].values()))
        scale=1. if clip is None else min(1.,clip/(gn+1e-6))
        expected={k:np.asarray(pp[k]-.3*scale*rr['grads'][k],dtype=np.float64) for k in NAMES}
        m=step(tp,tensor(xx),tensor(yy),ll,.3,clip=clip)
        close(m['loss_before_update'],rr['loss'],'step objective');close(m['grad_norm_before'],gn,'step norm')
        for k in NAMES:close(numpy_params(tp)[k],expected[k],'synchronous SGD '+k,2e-12,2e-11)
        z,_=forward(tp,tensor(xx),ll);rnew=full_bptt(expected,xx,np.full(8,9,dtype=np.int64),yt,last_only=True);close(z.detach().numpy(),rnew['z'],'updated next forward')
    checks.append('production SGD entry vs independent NumPy terminal loss, clipping, synchronous update and next forward')
    # Rejected domains remain rejected with -O.
    for bad in [x.astype(np.float32),np.full_like(x,np.nan),np.full_like(x,11.)]:rejected(lambda bad=bad:full_bptt(p,bad,l,y),'unsupported x')
    for bad in [np.array([0,2]),np.array([4,2]),l.astype(np.float64)]:rejected(lambda bad=bad:full_bptt(p,x,bad,y),'invalid lengths')
    for bad in [0,-1,1.5,True,4]:rejected(lambda bad=bad:forward(params_from_numpy(p),tensor(x),torch.tensor(l,dtype=torch.int64,device=DEV),bad),'truncate')
    rejected(lambda:make_split(3,2,1),'odd count');rejected(lambda:make_split(4,128,1),'sequence cap')
    rejected(lambda:step(init_params(1),tensor(xx),tensor(yy),ll,0),'nonpositive lr')
    rejected(lambda:step(init_params(1),tensor(xx),tensor(yy),ll,.3,clip=0),'nonpositive clip')
    rejected(lambda:token_loss(tensor(np.zeros((2,3))),tensor(y),torch.tensor([0,2],dtype=torch.int64,device=DEV)),'loss zero length')
    rejected(lambda:token_loss(tensor(np.full((2,3),np.inf)),tensor(y),torch.tensor(l,dtype=torch.int64,device=DEV)),'nonfinite logits')
    # Upstream differentiable embeddings are non-leaf tensors and remain supported.
    leaf=tensor(x,True);embedded=leaf*2.
    zz,_=forward(params_from_numpy(p),embedded,torch.tensor(l,dtype=torch.int64,device=DEV));token_loss(zz,tensor(y),torch.tensor(l,dtype=torch.int64,device=DEV)).backward()
    require(leaf.grad is not None and torch.isfinite(leaf.grad).all(),'upstream embedding chain')
    checks.append('explicit input bounds and invalid dtype/length/truncation/rate checks active under -O')
    # Data identity and disjoint generated noise rows, no target in final signal channel.
    data=np.load(ROOT/'data/delayed_sign.npz',allow_pickle=False);manifest=json.loads((ROOT/'data/manifest.json').read_text())
    import hashlib
    prot=json.loads((ROOT/'protocol.json').read_text())
    for d in prot['delays']:
        seen=[]
        for s,n in prot['counts'].items():
            a,b=make_split(n,d,prot['data_seeds'][s]);close(a,data[f'd{d}_{s}_x'],'data reproducibility',0,0);close(b,data[f'd{d}_{s}_y'],'label reproducibility',0,0)
            require(b.sum()==n/2,'balanced labels');seen.append(set(v.tobytes() for v in a))
        require(not seen[0]&seen[1] and not seen[0]&seen[2] and not seen[1]&seen[2],'no exact cross-split sequence duplicate')
    for key,sha in manifest['array_sha256'].items():require(hashlib.sha256(data[key].tobytes()).hexdigest()==sha,'data hash')
    checks.append('data regeneration, balance, distinct rows across splits and array hashes')
    print(json.dumps({'status':'author checks completed','optimized_python':not __debug__,'checks':checks},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
