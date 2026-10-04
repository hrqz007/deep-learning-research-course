"""Known geometries, explicit zero cases and independent rational projections."""
from fractions import Fraction as F
from tempfile import TemporaryDirectory
from pathlib import Path
import random
import json
import math
import numpy as np
import experiment as lab


def run_tests():
    u=np.array([2.,1.]);v=np.array([1.,3.]);groups=[]
    assert lab.dot(u,v)==5
    assert abs(lab.norm2(u)-math.sqrt(5))<1e-12
    assert abs(lab.distance(u,v)-math.sqrt(5))<1e-12
    assert abs(lab.cosine(u,v)-1/math.sqrt(2))<1e-12
    assert abs(lab.angle_degrees(u,v)-45)<1e-12
    coefficient,p,r=lab.project_to_direction(u,v)
    assert coefficient==.5
    np.testing.assert_allclose(p,[.5,1.5],rtol=0,atol=1e-12)
    np.testing.assert_allclose(r,[1.5,-.5],rtol=0,atol=1e-12)
    assert lab.dot(r,v)==0
    groups.append("complete hand-calculated 2D dot/norm/distance/angle/projection")
    a=np.array([1.,2.,-1.]);b=np.array([2.,0.,3.])
    c,p,r=lab.project_to_direction(a,b)
    assert c==-1/13 and lab.dot(a,b)==-1
    np.testing.assert_allclose(p,[-2/13,0,-3/13],rtol=0,atol=1e-12)
    np.testing.assert_allclose(r,[15/13,2,-10/13],rtol=0,atol=1e-12)
    assert abs(lab.distance(a,b)-math.sqrt(21))<1e-12
    groups.append("3D negative projection coefficient and exact rational coordinates")
    rng=random.Random(31008)
    for dimension in [2,3,5,7]:
        for repeat in range(25):
            first=[rng.randint(-10,10) for _ in range(dimension)]
            second=[rng.randint(-10,10) for _ in range(dimension)]
            if not any(second):second[0]=1
            uu=np.array(first,dtype=np.float64);vv=np.array(second,dtype=np.float64)
            true_dot=sum(F(x)*F(y) for x,y in zip(first,second))
            true_den=sum(F(y)**2 for y in second)
            true_coeff=true_dot/true_den
            coeff,proj,res=lab.project_to_direction(uu,vv)
            assert lab.dot(uu,vv)==float(true_dot)
            assert abs(coeff-float(true_coeff))<1e-12
            np.testing.assert_allclose(proj,[float(true_coeff*y) for y in second],rtol=0,atol=1e-12)
            assert abs(lab.dot(res,vv))<1e-10
            assert abs(lab.dot(uu,uu)-lab.dot(proj,proj)-lab.dot(res,res))<1e-10
            for candidate in [-2,-1,0,1,2]:assert lab.norm2(uu-candidate*vv)+1e-12>=lab.norm2(res)
    groups.append("100 independent Fraction dot/projection fixtures and nearest-line checks")
    assert abs(lab.cosine(u,2*u)-1)<1e-12 and lab.distance(u,2*u)>0
    assert abs(lab.cosine(2*u,v)-lab.cosine(u,v))<1e-12
    assert abs(lab.cosine(-u,v)+lab.cosine(u,v))<1e-12
    assert lab.cosine(np.array([1.,0.]),np.array([0.,2.]))==0
    assert lab.cosine(np.array([1.,1.]),np.array([1.,-1.]))==0
    assert abs(lab.cosine(np.array([1.,2.]),np.array([1.,-2.]))+.6)<1e-12
    groups.append("same direction differs in distance; scaling and translation counterexamples")
    zero=np.zeros(2)
    assert lab.dot(zero,u)==0 and lab.norm2(zero)==0
    for first,second in [(zero,u),(u,zero)]:
        try:lab.cosine(first,second)
        except ValueError:pass
        else:raise AssertionError("Zero-vector cosine accepted")
    try:lab.project_to_direction(u,zero)
    except ValueError:pass
    else:raise AssertionError("Zero direction accepted")
    for bad in [np.array([]),np.zeros((1,2)),np.array([np.nan,1]),np.array([np.inf,1])]:
        try:lab.norm2(bad)
        except ValueError:pass
        else:raise AssertionError("Invalid shape or nonfinite vector accepted")
    try:lab.dot(u,a)
    except ValueError:pass
    else:raise AssertionError("Mismatched dimensions accepted")
    groups.append("zero-vector distinctions, shape mismatch and nonfinite failures")
    try:lab.project_to_direction(np.array([1e150]),np.array([1e-160]))
    except ArithmeticError:pass
    else:raise AssertionError("Overflowing projection coefficient returned nonfinite values")
    # Some platforms implement longdouble with the same range as float64.
    with np.errstate(over="ignore",invalid="ignore"):
        wide=np.array([np.longdouble("1e400")])
    try:lab.vector(wide)
    except (ArithmeticError,ValueError):pass
    else:raise AssertionError("A nonfinite float64 conversion escaped validation")
    groups.append("projection coefficient overflow and post-conversion nonfinite values rejected")
    basis=lab.standard_basis(100)
    np.testing.assert_array_equal(basis@basis.T,np.eye(100))
    assert int(np.sum(np.all(basis[:,:2]==0,axis=1)))==98
    assert abs(lab.distance(basis[0],basis[99])-math.sqrt(2))<1e-12
    groups.append("100-dimensional coordinates and information loss in 2D display")
    with TemporaryDirectory() as tmp:
        first=Path(tmp)/"one";second=Path(tmp)/"two"
        assert lab.run(first)==lab.run(second)
        for p in first.iterdir():assert p.read_bytes()==(second/p.name).read_bytes()
    groups.append("repeated experiment produces identical outputs")
    return {"status":"passed","test_groups":groups}


if __name__=="__main__":
    print(json.dumps(run_tests(),indent=2))
