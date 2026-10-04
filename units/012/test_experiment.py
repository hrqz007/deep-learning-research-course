"""Exact rational index oracle, finite differences, shape and range failures."""
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re
import numpy as np
import experiment as lab


def fraction_oracle(X, p, Y):
    X=[[F(str(x)) for x in r] for r in X]
    p={k:[[F(str(x)) for x in r] for r in v] for k,v in p.items()}
    Y=[[F(str(x)) for x in r] for r in Y]
    n,d,h,c=len(X),len(X[0]),len(p['W1'][0]),len(Y[0])
    out={k:[[F(0) for _ in r] for r in v] for k,v in {**p,'X':X}.items()}
    total=F(0)
    for i in range(n):
        a=[sum(X[i][j]*p['W1'][j][k] for j in range(d))+p['b1'][0][k] for k in range(h)]
        for o in range(c):
            pred=sum(a[k]**2*p['W2'][k][o] for k in range(h))+p['b2'][0][o]
            e=pred-Y[i][o]
            total+=e**2/(2*n)
            out['b2'][0][o]+=e/n
            for k in range(h):
                out['W2'][k][o]+=e*a[k]**2/n
                out['b1'][0][k]+=e*2*a[k]*p['W2'][k][o]/n
                for j in range(d):
                    out['W1'][j][k]+=e*2*a[k]*p['W2'][k][o]*X[i][j]/n
                    out['X'][i][j]+=e*2*a[k]*p['W2'][k][o]*p['W1'][j][k]/n
    return float(total),{k:np.array(v,dtype=float) for k,v in out.items()}


def reject(call):
    try:call()
    except (TypeError,ValueError,ArithmeticError):return
    raise AssertionError('Invalid input accepted')


def tests():
    cfg,X,p,Y=lab.load_config();groups=[]
    g=lab.gradients(X,p,Y)
    assert lab.loss(X,p,Y)==29.140625
    expected={'W1':[[-7.375,-50.75],[1.375,59.75]],'b1':[[3.375,56.75]],'W2':[[-3.34375],[-34.71875]],'b2':[[-5.875]],'X':[[-5,2],[-48.375,56.4375]]}
    for k,v in expected.items():np.testing.assert_array_equal(g[k],v)
    np.testing.assert_array_equal(lab.parameter_jacobian(X,p),[[4,-6,8,-12,4,-6,4,2.25,1],[1,10,-1,-10,-1,-10,.25,6.25,1]])
    groups.append('complete exact hand calculation and explicit 2 by 9 Jacobian')
    rng=np.random.default_rng(12012)
    for fixture in range(100):
        n,d,h,c=(3,2,3,2) if fixture%2 else (2,3,2,1)
        a=lambda shape:rng.integers(-4,5,size=shape)/4
        xx=a((n,d));yy=a((n,c));pp={'W1':a((d,h)),'b1':a((1,h)),'W2':a((h,c)),'b2':a((1,c))}
        val,oracle=fraction_oracle(xx,pp,yy);gg=lab.gradients(xx,pp,yy)
        assert abs(lab.loss(xx,pp,yy)-val)<1e-12
        for k in oracle:np.testing.assert_allclose(gg[k],oracle[k],rtol=1e-13,atol=1e-13)
        dirs={k:a(pp[k].shape) for k in lab.PARAMETERS};weights=a((n,c));J=lab.parameter_jacobian(xx,pp)
        push=lab.jvp(xx,pp,dirs);pull=lab.vjp(xx,pp,weights)
        np.testing.assert_allclose(push.ravel(),J @ lab.flatten(dirs),rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(lab.flatten(pull),weights.ravel() @ J,rtol=1e-13,atol=1e-13)
        assert abs(np.sum(weights*push)-lab.flatten(pull)@lab.flatten(dirs))<1e-12
    groups.append('100 Fraction loop oracles with rectangular dimensions; explicit Jacobian JVP VJP identities')
    num=lab.central_gradients(X,p,Y,1e-5)
    for k in num:np.testing.assert_allclose(num[k],g[k],atol=2e-8,rtol=0)
    groups.append('all nine parameters and four input coordinates independently differenced')
    X2=np.concatenate([X,X]);Y2=np.concatenate([Y,Y]);g2=lab.gradients(X2,p,Y2)
    assert lab.loss(X2,p,Y2)==lab.loss(X,p,Y)
    for k in lab.PARAMETERS:np.testing.assert_array_equal(g2[k],g[k])
    np.testing.assert_array_equal(g2['X'],np.concatenate([g['X']/2,g['X']/2]))
    perm=[1,0];gp=lab.gradients(X[perm],p,Y[perm])
    for k in lab.PARAMETERS:np.testing.assert_array_equal(gp[k],g[k])
    groups.append('batch duplication mean-loss invariance and permutation equivariance')
    failures=[]
    for bad in [[],[1,2],[[[1,2]]],[[True,2],[1,2]],[[np.array(True),2],[1,2]],[[1+0j,2],[1,2]], [['1','2'],['3','4']],[[np.nan,2],[1,2]],[[np.inf,2],[1,2]]]:
        failures.append(lambda bad=bad:lab.forward(bad,p))
    for name,bad in [('b1',[0,.5]),('b1',[[0],[.5]]),('W1',[[1,2,3]]),('W2',[[1,2]]),('b2',[[1,2]])]:
        failures.append(lambda name=name,bad=bad:lab.forward(X,{**p,name:bad}))
    failures += [lambda:lab.forward(X,{**p,'extra':[[1]]}),lambda:lab.loss(X,p,[1,-1]),lambda:lab.loss(X,p,[[1,-1]]),lambda:lab.vjp(X,p,[[1,2]]),lambda:lab.jvp(X,p,{k:[[1]] for k in lab.PARAMETERS})]
    for h in [True,0,-1,np.nan,np.inf,[.1],1e-17,1e308]:failures.append(lambda h=h:lab.central_gradients(X,p,Y,h))
    failures += [lambda:lab.forward([[1e308,1e308]],p),lambda:lab.forward([[1e200,0]],p),lambda:lab.vjp(X,p,[[1e308],[1e308]]),lambda:lab.matrix(np.array([[np.longdouble('1e400')]]),'longdouble')]
    for f in failures:reject(f)
    before={k:v.copy() for k,v in p.items()};saved=X.copy();lab.central_gradients(X,p,Y,1e-4)
    for k in p:np.testing.assert_array_equal(p[k],before[k])
    np.testing.assert_array_equal(X,saved)
    groups.append(f'{len(failures)} type shape range step rejection cases; original inputs unchanged')
    with TemporaryDirectory() as temp:
        a,b=Path(temp)/'a',Path(temp)/'b';lab.run(a);lab.run(b)
        for f in a.iterdir():assert f.read_bytes()==(b/f.name).read_bytes()
    groups.append('two full runs produce byte-identical three-file outputs')
    snippets=0;ns={}
    for name in ['lecture.md','lab.md','answers.md']:
        if not (lab.BASE/name).exists():continue
        for code in re.findall(r'```python\n(.*?)```',(lab.BASE/name).read_text(),re.S):exec(code,ns);snippets+=1
    groups.append(f'{snippets} document Python snippets execute in order')
    return dict(status='passed',groups=len(groups),details=groups,fraction_fixtures=100,rejected_cases=len(failures),python_snippets=snippets)

if __name__=='__main__':print(json.dumps(tests(),ensure_ascii=False,indent=2))
