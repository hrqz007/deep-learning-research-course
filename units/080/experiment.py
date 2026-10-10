"""S3: ODE states, discrete tangent/reverse derivatives, parameter recovery.
Time: seconds. State: arbitrary concentration units. k: reciprocal seconds.
"""
from pathlib import Path
import argparse,json,time
import numpy as np

def validate(k,y0,T,n):
    if not np.isfinite([k,y0,T]).all() or k<0 or T<=0 or isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1:
        raise ValueError('finite k>=0, y0, T>0 and integer n>=1 required')

def exact(k,y0,t):return y0*np.exp(-k*np.asarray(t))

def euler(k=1.,y0=2.,T=2.,n=20):
    validate(k,y0,T,n);dt=T/n;y=np.empty(n+1);s=np.empty(n+1);y[0]=y0;s[0]=0
    for j in range(n):
        # s is d y_j / d k, propagated through the same discrete arithmetic.
        y[j+1]=y[j]-dt*k*y[j];s[j+1]=(1-dt*k)*s[j]-dt*y[j]
    return np.linspace(0,T,n+1),y,s

class Value:
    """Minimal scalar reverse-mode autodiff: store local edges, then backpropagate.
    This teaching engine supports only addition and multiplication, exactly the
    operations needed by Euler here. It is not a general tensor framework.
    """
    def __init__(self,value,parents=()):
        self.value=float(value);self.parents=parents;self.grad=0.
    @staticmethod
    def wrap(value):return value if isinstance(value,Value) else Value(value)
    def __add__(self,other):
        other=self.wrap(other)
        return Value(self.value+other.value,((self,1.),(other,1.)))
    __radd__=__add__
    def __mul__(self,other):
        other=self.wrap(other)
        return Value(self.value*other.value,((self,other.value),(other,self.value)))
    __rmul__=__mul__
    def __neg__(self):return self*(-1.)
    def __sub__(self,other):return self+(-self.wrap(other))
    def __rsub__(self,other):return self.wrap(other)+(-self)
    def backward(self):
        # Each node is visited once; shared parameters receive summed gradients.
        order=[];seen=set()
        def visit(node):
            if id(node) in seen:return
            seen.add(id(node))
            for parent,_ in node.parents:visit(parent)
            order.append(node)
        visit(self)
        for node in order:node.grad=0.
        self.grad=1.
        for node in reversed(order):
            for parent,local_derivative in node.parents:
                parent.grad+=node.grad*local_derivative

def autodiff_euler(k=1.,y0=2.,T=2.,n=20):
    """Run ordinary Euler arithmetic with Value nodes, then differentiate it."""
    validate(k,y0,T,n);parameter=Value(k);state=Value(y0);dt=T/n
    for _ in range(n):state=state-dt*parameter*state
    state.backward()
    return state.value,parameter.grad

def reverse_gradient(k,y0,T,n):
    t,y,s=euler(k,y0,T,n);dt=T/n;adj=1.;gradient=0.
    for j in range(n-1,-1,-1):
        gradient+=adj*(-dt*y[j]);adj*=1-dt*k
    return gradient

def rk4(k,y0,T,n):
    validate(k,y0,T,n);dt=T/n;y=np.empty(n+1);y[0]=y0
    for j in range(n):
        a=-k*y[j];b=-k*(y[j]+dt*a/2);c=-k*(y[j]+dt*b/2);d=-k*(y[j]+dt*c)
        y[j+1]=y[j]+dt*(a+2*b+2*c+d)/6
    return np.linspace(0,T,n+1),y

def fit(observed,n=20,T=2.,y0=2.,steps=400):
    k=.3;history=[]
    for _ in range(steps):
        _,y,s=euler(k,y0,T,n);err=y[-1]-observed;grad=err*s[-1];k=max(0.,k-.3*grad);history.append([k,.5*err**2])
    return k,np.array(history)

def heat_explicit(u0,alpha,dx,dt,steps):
    u=np.asarray(u0,float).copy()
    if u.ndim!=1 or len(u)<3 or not np.isfinite(u).all() or alpha<0 or dx<=0 or dt<=0 or steps<0:raise ValueError('invalid heat setup')
    mu=alpha*dt/(dx*dx)
    if mu>.5:raise ValueError('explicit heat step violates mu<=1/2')
    u[0]=u[-1]=0
    for _ in range(steps):
        v=u.copy();v[1:-1]=u[1:-1]+mu*(u[:-2]-2*u[1:-1]+u[2:]);u=v
    return u

def run(output='outputs'):
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True);k,y0,T=1.,2.,2.;ns=np.array([4,8,16,32,64,128]);rows=[];arrays={};observed=float(exact(k,y0,T));fits=[]
    for n0 in ns:
        n=int(n0);t,y,s=euler(k,y0,T,n);tr,yr=rk4(k,y0,T,n);fd=(euler(k+1e-6,y0,T,n)[1][-1]-euler(k-1e-6,y0,T,n)[1][-1])/2e-6
        rows.append([n,T/n,abs(y[-1]-observed),abs(s[-1]+T*observed),abs(yr[-1]-observed),abs(s[-1]-fd),abs(s[-1]-reverse_gradient(k,y0,T,n))]);kh,h=fit(observed,n,T,y0);fits.append([n,kh]);arrays['fit_history_'+str(n)]=h
    t,y,s=euler(k,y0,T,16);arrays.update(t=t,euler_y=y,exact_y=exact(k,y0,t),sensitivity=s,scan=np.array(rows),fit_scan=np.array(fits))
    unstable=[]
    for dt in [.5,1.5,2.5]:
        tt,yy,_=euler(1,2,dt*12,12);unstable.append(yy)
    arrays['stability_trajectories']=np.array(unstable)
    xx=np.linspace(0,1,41);dx=xx[1]-xx[0];alpha=.1;dt=.4*dx*dx/alpha;steps=100;u0=np.sin(np.pi*xx);u=heat_explicit(u0,alpha,dx,dt,steps);u_true=np.exp(-alpha*np.pi**2*dt*steps)*u0
    arrays.update(heat_x=xx,heat_u=u,heat_exact=u_true)
    result={'k_true':k,'y0':y0,'T':T,'observed_terminal':observed,'scan_columns':['n','dt','euler_state_error','euler_gradient_error','rk4_state_error','finite_difference_error','reverse_error'],'scan':rows,'fit_scan':fits,'heat_mu':.4,'heat_max_error':float(np.max(np.abs(u-u_true))),'heat_boundary_error':float(max(abs(u[0]),abs(u[-1]))),'seconds':time.perf_counter()-start}
    np.savez(out/'data.npz',**arrays);np.savez(out/'weights.npz',estimated_k=np.array(fits)[:,1]);np.savez(out/'predictions.npz',ode=y,heat=u);(out/'results.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');print(json.dumps(run(p.parse_args().output),indent=2))
