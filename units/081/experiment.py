"""NumPy PINN for -u''=pi^2 sin(pi*x), u(0)=u(1)=0.
All w,b,v,c parameters are genuinely optimized, not fixed random features.
"""
from pathlib import Path
import os,time,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np

def field(theta,x):
    w,b,v,c=[np.asarray(t,dtype=float) for t in theta]
    if w.ndim!=1 or b.shape!=w.shape or v.shape!=w.shape or c.shape!=(1,):
        raise ValueError("theta must have w,b,v vectors of equal length and c shape (1,)")
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or not all(np.isfinite(t).all() for t in [w,b,v,c]):
        raise ValueError("finite one-dimensional x and parameters required")
    x=x.reshape(-1,1)
    h=np.tanh(x*w+b); s=1-h*h; q=-2*h*s
    return h@v+c[0], (s*w)@v, (q*w*w)@v

def loss_grad(theta,x,lam=20.):
    w,b,v,c=theta; X=x[:,None]; h=np.tanh(X*w+b); s=1-h*h
    q=-2*h*s; qp=-2+8*h*h-6*h**4
    r=-(q*w*w)@v-np.pi**2*np.sin(np.pi*x)
    weights=2*r/len(x)
    gw=np.sum(weights[:,None]*(-v*(2*w*q+w*w*qp*X)),axis=0)
    gb=np.sum(weights[:,None]*(-v*w*w*qp),axis=0)
    gv=np.sum(weights[:,None]*(-w*w*q),axis=0); gc=np.zeros(1)
    xb=np.array([0.,1.])[:,None]; hb=np.tanh(xb*w+b); sb=1-hb*hb
    ub=hb@v+c[0]; bw=lam*ub
    gw+=np.sum(bw[:,None]*sb*v*xb,axis=0)
    gb+=np.sum(bw[:,None]*sb*v,axis=0)
    gv+=hb.T@bw;gc[0]=bw.sum()
    return float(np.mean(r*r)+lam*np.mean(ub*ub)),[gw,gb,gv,gc]

def train(seed=81,n=65,steps=12000):
    if not isinstance(n,int) or n<1 or not isinstance(steps,int) or steps<1:
        raise ValueError("positive integer n and steps required")
    rng=np.random.default_rng(seed); m=20
    theta=[rng.normal(0,1.5,m),rng.normal(0,.4,m),rng.normal(0,.2,m),np.zeros(1)]
    mom=[np.zeros_like(t) for t in theta]; var=[t.copy() for t in mom]
    x=(np.arange(n)+.5)/n;history=[];start=time.perf_counter()
    for step in range(1,steps+1):
        loss,grads=loss_grad(theta,x)
        lr=.003 if step<9000 else .0005
        for j,g in enumerate(grads):
            mom[j]=.9*mom[j]+.1*g;var[j]=.999*var[j]+.001*g*g
            theta[j]-=lr*(mom[j]/(1-.9**step))/(np.sqrt(var[j]/(1-.999**step))+1e-8)
        if step==1 or step%200==0:history.append([step,loss])
    return theta,np.array(history),time.perf_counter()-start,x

def finite_difference(n):
    # Thomas algorithm: linear cost, no artificial dense-matrix disadvantage.
    if not isinstance(n,int) or n<1:raise ValueError("n must be a positive integer")
    h=1/(n+1);x=np.linspace(0,1,n+2);d=np.full(n,2.);rhs=h*h*np.pi**2*np.sin(np.pi*x[1:-1])
    for i in range(1,n):
        factor=-1/d[i-1];d[i]+=factor;rhs[i]-=factor*rhs[i-1]
    u=np.zeros(n+2);u[-2]=rhs[-1]/d[-1]
    for i in range(n-2,-1,-1):u[i+1]=(rhs[i]+u[i+2])/d[i]
    return x,u

def run():
    (ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    grid=np.linspace(0,1,1001);truth=np.sin(np.pi*grid);rows=[];result={}
    for label,seed,n in [('pinn',81,65),('pinn_seed82',82,65),('pinn_sparse',81,5)]:
        theta,hist,seconds,x=train(seed,n);u,du,ddu=field(theta,grid)
        r=-ddu-np.pi**2*truth;trainr=-field(theta,x)[2]-np.pi**2*np.sin(np.pi*x)
        result[label]={'relative_l2':float(np.linalg.norm(u-truth)/np.linalg.norm(truth)),
            'max_error':float(np.max(abs(u-truth))),'boundary_max':float(max(abs(u[0]),abs(u[-1]))),
            'heldout_residual_rmse':float(np.sqrt(np.mean(r*r))),
            'train_residual_rmse':float(np.sqrt(np.mean(trainr*trainr))),
            'flux_balance_abs':float(abs((-du[-1]+du[0])-2*np.pi)),
            'training_seconds':seconds,'parameters':61,'collocation_points':n,'steps':12000}
        np.savez(ROOT/'outputs'/f'{label}_weights.npz',w=theta[0],b=theta[1],v=theta[2],c=theta[3])
        np.savetxt(ROOT/'outputs'/f'{label}_history.csv',hist,delimiter=',',header='step,loss',comments='')
        rows.append(u)
    for n in [31,63,127]:
        start=time.perf_counter();x,u=finite_difference(n);elapsed=time.perf_counter()-start
        pred=np.interp(grid,x,u);h=x[1]-x[0]
        # Conservative boundary face fluxes of the discrete finite-volume balance.
        net=(u[-2]+u[1])/h;source=h*np.sum(np.pi**2*np.sin(np.pi*x[1:-1]))
        result[f'fd_{n}']={'relative_l2':float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)),
            'max_error':float(np.max(abs(pred-truth))),'boundary_max':0.,
            'flux_balance_discrete_abs':float(abs(net-source)),
            'continuum_flux_error':float(abs(net-2*np.pi)),'solve_seconds':elapsed,'unknowns':n}
        if n==63: rows.append(pred)
    np.savetxt(ROOT/'data/evaluation.csv',np.column_stack([grid,truth]+rows),delimiter=',',header='x,truth,pinn,pinn_seed82,pinn_sparse,fd63',comments='')
    (ROOT/'outputs/metrics.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
