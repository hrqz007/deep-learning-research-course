"""Independent Decimal/Fraction checks and explicit dtype failure tests."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
import json, math, re
import numpy as np
import experiment as lab


def oracle(values):
    with localcontext() as c:
        c.prec=110
        z=[D.from_float(float(x)) for x in values]
        # Independent direct exponentiation is safe in Decimal for test ranges.
        e=[v.exp() for v in z];total=sum(e)
        return np.array([float(v/total) for v in e]),float(total.ln())


def reject(call):
    try:call()
    except (ValueError,TypeError,ArithmeticError):return
    raise AssertionError('Invalid input was accepted')


def tests():
    groups=[];rng=np.random.default_rng(14014)
    for dtype in ['float32','float64']:
        dt=lab.dtype_type(dtype);info=lab.format_info(dtype)
        assert info['bits']==(32 if dtype=='float32' else 64)
        assert F(info['eps'])==F(1,2**(23 if dtype=='float32' else 52))
        assert info['spacing_at_2']==2*info['spacing_at_1']
        assert info['one_plus_half_eps_equals_one']
    groups.append('exact format eps spacing and tie-to-even checks')
    for i in range(100):
        values=rng.integers(-160,161,size=int(rng.integers(2,9)))/4
        for dtype in ['float32','float64']:
            z=lab.vector(values,dtype);p,l=oracle(z);got=lab.stable_distribution(z,dtype)
            tol=8e-7 if dtype=='float32' else 2e-14
            np.testing.assert_allclose(got['probabilities'],p,rtol=tol,atol=tol)
            assert abs(float(got['logsumexp'])-l)<(5e-6 if dtype=='float32' else 2e-14)
            assert abs(float(got['probabilities'].sum())-1)<tol
            np.testing.assert_allclose(lab.stable_distribution(z+1000,dtype)['probabilities'],got['probabilities'],rtol=tol,atol=tol)
    groups.append('100 random vectors times two dtypes against independent 110-digit Decimal and exact shifts')
    for dtype in ['float32','float64']:
        for z in [[1000,1001,999],[-1000,-999,-1001]]:
            assert not np.isfinite(lab.naive_softmax(z,dtype)).all()
            s=lab.stable_distribution(z,dtype);expected,_=oracle(z)
            np.testing.assert_allclose(s['probabilities'],expected,atol=1e-7)
        t=lab.stable_distribution([0,-200],dtype)
        assert np.isfinite(t['log_probabilities']).all()
        assert abs(float(t['log_probabilities'][1])+200)<1e-6
        assert t['zero_tail_terms']==(1 if dtype=='float32' else 0)
    groups.append('overflow all-zero underflow and finite log-tail cases')
    np.testing.assert_array_equal(lab.stable_distribution([1e8,1e8+1],'float32')['probabilities'],[.5,.5])
    p,_=oracle([0,1]);np.testing.assert_allclose(lab.stable_distribution([1e8,1e8+1],'float64')['probabilities'],p,atol=1e-15)
    groups.append('input casting loses score difference before stable formula')
    for dtype in ['float32','float64']:
        for x in [0,-1,-.5,1e-4,1e-8,1e-12,1e-16,1e-100]:
            r=lab.cancellation(x,dtype)
            with localcontext() as c:
                c.prec=150;actual=D.from_float(r['stored_x']);ref=actual/((1+actual).sqrt()+1)
            assert abs(r['rationalized']-float(ref)) <= (2e-7 if dtype=='float32' else 3e-16)*abs(float(ref)) + 1e-323
    groups.append('rationalized sqrt Decimal reference including zero and domain endpoint')
    for i in range(41):
        x=(i-20)/10
        for dtype,h,tol in [('float64',1e-5,1e-9),('float32',.01,2e-4)]:
            got=lab.central_exp(x,h,dtype)
            with localcontext() as c:
                c.prec=110;expected=float(D.from_float(got['stored_x']).exp())
            assert got['reference']==expected
            assert abs(got['value']-expected)<tol
    for x in [F(-2),F(0),F(7,10)]:
        for h in [F(1,2),F(1,10),F(1,100)]:
            central=((x+h)**3-(x-h)**3)/(2*h)
            assert central-3*x*x==h*h
    groups.append('82 independent exp derivative references and nine exact cubic difference identities')
    failures=[]
    for bad in [[],[[1,2]],[True,2],[np.array(True),2],[1+0j,2],['1','2'],[np.nan,2],[np.inf,2]]:
        failures.append(lambda bad=bad:lab.stable_distribution(bad))
    failures += [lambda:lab.stable_distribution([1,2],'float16'),lambda:lab.stable_distribution([1e100],'float32'),lambda:lab.stable_distribution([-np.finfo(float).max,np.finfo(float).max]),lambda:lab.cancellation(-1.1),lambda:lab.cancellation(True)]
    for h in [0,-1,True,np.nan,np.inf,[.1],1e-17,1e308]:failures.append(lambda h=h:lab.central_exp(.7,h))
    failures += [lambda:lab.central_exp(1000,.1),lambda:lab.central_exp(.7,1e-100,'float32')]
    for f in failures:reject(f)
    z=np.array([1000.,1001.,999.]);saved=z.copy();lab.stable_distribution(z);np.testing.assert_array_equal(z,saved)
    groups.append(f'{len(failures)} invalid shape type domain range and step cases; source array unchanged')
    with TemporaryDirectory() as tmp:
        a,b=Path(tmp)/'a',Path(tmp)/'b';r=lab.run(a);lab.run(b)
        assert r['associativity']=={'left':1.,'right':0.,'fsum':1.}
        for f in a.iterdir():assert f.read_bytes()==(b/f.name).read_bytes()
    groups.append('associativity counterexample and byte-identical repeated complete runs')
    ns={};count=0
    for name in ['lecture.md','lab.md','answers.md']:
        if not (lab.BASE/name).exists():continue
        for code in re.findall(r'```python\n(.*?)```',(lab.BASE/name).read_text(),re.S):exec(code,ns);count+=1
    groups.append(f'{count} document Python snippets execute in order')
    return {'status':'passed','groups':len(groups),'details':groups,'random_vector_fixtures':100,'dtype_cases':200,'rejected_cases':len(failures),'python_snippets':count}
if __name__=='__main__':print(json.dumps(tests(),ensure_ascii=False,indent=2))
