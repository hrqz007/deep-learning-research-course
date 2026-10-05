"""Author's executable numerical checks. Uses explicit exceptions under python -O."""
import copy,json,math,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from reference import conv,conv_backward,residual,hand_ledger
from experiment import SmallCNN,budget,load_data

def close(a,b,name,tol=1e-9):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    if not np.isfinite(error) or error>tol:raise RuntimeError(f'{name}: max error {error} > {tol}')
    return error

def rejects(fn,name):
    try:fn()
    except (ValueError,TypeError):return
    raise RuntimeError(name+' should reject invalid input')

def finite_difference(fn,a,eps=1e-6):
    result=np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old=a[idx];a[idx]=old+eps;hi=fn();a[idx]=old-eps;lo=fn();a[idx]=old
        result[idx]=(hi-lo)/(2*eps)
    return result

def run_checks():
    torch.set_num_threads(1);rng=np.random.default_rng(4410);metrics={}
    for ci,co,h,stride in [(2,2,4,1),(2,3,5,2),(1,2,4,2)]:
        x=rng.normal(size=(1,ci,h,h));w1=rng.normal(0,.2,(co,ci,3,3));w2=rng.normal(0,.2,(co,co,3,3))
        p=None if ci==co and stride==1 else rng.normal(0,.2,(co,ci,1,1))
        out=residual(x,w1,w2,p,stride);g=rng.normal(size=out.shape);ref=residual(x,w1,w2,p,stride,g)
        xx=torch.tensor(x,requires_grad=True);a=torch.tensor(w1,requires_grad=True);b=torch.tensor(w2,requires_grad=True)
        pp=None if p is None else torch.tensor(p,requires_grad=True)
        shortcut=xx if pp is None else F.conv2d(xx,pp,stride=stride)
        yy=shortcut+F.conv2d(torch.relu(F.conv2d(xx,a,stride=stride,padding=1)),b,padding=1)
        (yy*torch.tensor(g)).sum().backward();key=f'c{ci}_{co}_h{h}_s{stride}'
        errors=[close(ref['y'],yy.detach().numpy(),key+' forward'),close(ref['dx'],xx.grad.numpy(),key+' dx'),
                close(ref['dw1'],a.grad.numpy(),key+' dw1'),close(ref['dw2'],b.grad.numpy(),key+' dw2')]
        if p is not None:errors.append(close(ref['dp'],pp.grad.numpy(),key+' dp'))
        # Every entry checked, not one cherry-picked coordinate; perturb away from ReLU boundaries.
        if np.min(np.abs(ref['z']))<1e-5:raise RuntimeError('finite difference would approach a ReLU boundary')
        for value,derivative in [(x,ref['dx']),(w1,ref['dw1']),(w2,ref['dw2'])]+([] if p is None else [(p,ref['dp'])]):
            fd=finite_difference(lambda:float(np.sum(residual(x,w1,w2,p,stride)*g)),value)
            errors.append(close(fd,derivative,key+' central difference',1e-7))
        metrics[key]=max(errors)
    # Exact hand derivation: six-parameter gradients and input path sums, all three forwards.
    ledger=hand_ledger()
    for i,round_ in enumerate(ledger['rounds']):
        theta=torch.tensor(round_['theta'],dtype=torch.float64,requires_grad=True)
        x=torch.tensor([[-1.,1.],[1.,2.]],dtype=torch.float64,requires_grad=True);target=torch.tensor([0.,1.])
        w,b,v,c,a,d=theta;pred=a*(x+v*torch.relu(w*x+b)+c).mean(1)+d
        loss=((pred-target)**2).mean()/2;loss.backward()
        metrics['hand_round_'+str(i)]=max(close(theta.grad.numpy(),round_['gradient'],'all hand gradients'),
            close(x.grad.numpy(),[r['dx'] for r in round_['rows']],'hand path sum'),close(float(loss.detach()),round_['loss'],'hand loss'))
    close(ledger['rounds'][0]['gradient'],[.1408,.0976,.237,.272,.4895,.34],'independent literal hand gradient')
    # F(x)=-x is a valid counterexample to a guaranteed nonzero skip gradient.
    x=torch.tensor([2.],requires_grad=True);(x-x).sum().backward();close(x.grad.numpy(),[0.],'cancellation counterexample')
    for blocks in (2,6):
        torch.manual_seed(11);plain=SmallCNN(blocks,False)
        torch.manual_seed(11);skip=SmallCNN(blocks,True)
        if sum(p.numel() for p in plain.parameters())!=budget(blocks,False)['parameters']:raise RuntimeError('parameter formula wrong')
        for (a,b) in zip(plain.state_dict().values(),skip.state_dict().values()):close(a.numpy(),b.numpy(),'paired initial state')
        x,y=load_data('train');before={k:v.clone() for k,v in skip.state_dict().items() if 'running' in k or 'num_batches' in k}
        skip.eval();skip(x[:2]);after=skip.state_dict()
        for k,v in before.items():close(v.numpy(),after[k].numpy(),'eval BN state invariant')
    z=np.zeros((1,1,3,3));w=np.zeros((1,1,3,3))
    rejects(lambda:conv(z,w,True),'boolean stride');rejects(lambda:conv(z,w,0),'zero stride')
    rejects(lambda:conv(z,w,-1),'negative stride');rejects(lambda:conv(z,w,1,-1),'negative padding')
    rejects(lambda:conv(np.ones((1,1,3,3),dtype=int),w),'integer tensor')
    rejects(lambda:conv(z,np.ones((1,2,3,3))),'channels')
    rejects(lambda:conv(np.full_like(z,np.nan),w),'NaN')
    rejects(lambda:conv(z,np.ones((1,1,9,9))),'oversized kernel')
    with np.errstate(over='ignore',invalid='ignore'):
        rejects(lambda:conv(np.full_like(z,1e308),np.full_like(w,1e308)),'finite input causing arithmetic overflow')
    rejects(lambda:conv_backward(z,w,np.zeros((1,1,2,2))),'upstream shape')
    rejects(lambda:residual(z,np.zeros((2,1,3,3)),np.zeros((2,2,3,3))),'broadcast shortcut')
    rejects(lambda:residual(z,w,w,stride=3),'unsupported stride')
    rejects(lambda:SmallCNN(3,False),'unsupported depth')
    rejects(lambda:SmallCNN(2,False)(torch.zeros((1,1,12,12),dtype=torch.float64)),'dtype')
    return dict(numerical_max_errors=metrics,scope='author numerical checks, not independent QA',optimization_disabled=not __debug__)
if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
