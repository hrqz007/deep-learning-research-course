"""Domain, exact-value and independently derived algebra checks."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import json
import random
import subprocess
import sys
import csv
import experiment as lab


def run_tests():
    groups=[]
    assert lab.solve_affine(2,1,7)=={"kind":"unique","value":3}
    assert lab.solve_affine(0,3,3)["kind"]=="all_real"
    assert lab.solve_affine(0,3,4)["kind"]=="no_solution"
    rng=random.Random(31007)
    for _ in range(100):
        a=rng.choice([-5,-3,-1,1,2,4]);b=rng.randint(-10,10);c=rng.randint(-10,10)
        expected=Fraction(c-b,a)
        actual=lab.solve_affine(a,b,c)["value"]
        assert abs(actual-float(expected))<1e-12
        assert abs(a*actual+b-c)<1e-12
    groups.append("three equation cases and 100 independent Fraction roots")
    for cm in [1,2,3]:assert lab.affine(cm)==lab.affine(10*cm,.2,1)
    assert lab.affine(30)==61 and lab.affine(30,.2,1)==7
    groups.append("unit-conversion invariance and incorrect-coefficient counterexample")
    for exponent,expected in zip([-2,-1,0,1,2,3],[.25,.5,1,2,4,8]):assert lab.power_two(exponent)==expected
    assert (-2)**2==4 and -2**2==-4
    groups.append("integer powers, negative exponents and Python precedence")
    for x,expected in [(.25,-2),(.5,-1),(1,0),(2,1),(4,2),(8,3)]:assert abs(lab.real_log(x)-expected)<1e-12
    for x,base in [(0,2),(-1,2),(1,1),(2,0),(2,-2),(float("nan"),2)]:
        try:lab.real_log(x,base)
        except ValueError:pass
        else:raise AssertionError("Invalid logarithm domain accepted")
    groups.append("logarithm known values and invalid domains")
    for x,y in [(.5,4),(4,8),(3,5),(10,.2)]:
        assert abs(lab.real_log(x*y)-(lab.real_log(x)+lab.real_log(y)))<1e-12
    assert lab.real_log(4+4)==3
    assert lab.real_log(4)+lab.real_log(4)==4
    groups.append("product-log identity and sum-log counterexample")
    A=lab.positive_loss_summary([1,9]);B=lab.positive_loss_summary([4,4])
    assert A["mean"]>B["mean"] and A["log2_of_mean"]>B["log2_of_mean"]
    assert A["mean_of_log2"]<B["mean_of_log2"]
    assert abs(A["mean_of_log2"]-math.log2(3))<1e-12
    groups.append("ranking reversal only for per-item log before averaging")
    assert lab.affine(lab.square(2))==9 and lab.square(lab.affine(2))==25
    for x in [0,.5,1]:
        try:lab.real_log(x-1)
        except ValueError:pass
        else:raise AssertionError("Invalid composition domain accepted")
    assert lab.real_log(2-1)==0
    assert [lab.rectifier(x) for x in [-2,0,3]]==[0,0,3]
    groups.append("composition order, composition domain and piecewise boundary")
    assert sum(v+1 for v in [2,4,6])==15 and sum([2,4,6])+1==13
    with TemporaryDirectory() as tmp:
        report=lab.run(Path(tmp)/"one")
        assert report==lab.run(Path(tmp)/"two")
        for p in (Path(tmp)/"one").iterdir():assert p.read_bytes()==(Path(tmp)/"two"/p.name).read_bytes()
    groups.append("summation expansion and byte-identical repeat outputs")
    for function,args,error in [
        (lab.affine,("3",),TypeError), (lab.affine,(True,),TypeError),
        (lab.square,(float("inf"),),ValueError),
        (lab.power_two,(1024,),ArithmeticError),
        (lab.power_two,(-1075,),ArithmeticError),
        (lab.affine,(1e308,2,1),ArithmeticError),
        (lab.square,(1e308,),ArithmeticError),
        (lab.solve_affine,(1e-308,0,1e308),ArithmeticError),
        (lab.solve_affine,(1e308,0,1e-308),ArithmeticError),
        (lab.positive_loss_summary,([],),ValueError),
        (lab.positive_loss_summary,([1,0],),ValueError),
        (lab.positive_loss_summary,([1,-1],),ValueError),
        (lab.positive_loss_summary,([1,float("nan")],),ValueError)]:
        try: function(*args)
        except error: pass
        else: raise AssertionError(f"Expected {error.__name__} for {function.__name__}")
    groups.append("type, empty-data, nonfinite, overflow and underflow failures")
    for x,r in [(2,3),(4,.5),(.5,-2)]:
        assert abs(lab.real_log(x**r)-r*lab.real_log(x)) < 1e-12
    assert abs(lab.real_log(8,.5)+3) < 1e-12
    assert lab.real_log(2,.5) > lab.real_log(4,.5)
    assert abs(lab.real_log(8,10)-math.log10(8)) < 1e-12
    groups.append("power and change-of-base laws; decreasing logarithm for base below one")
    unit=Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        child=subprocess.run([sys.executable,str(unit/"experiment.py")],cwd=tmp,capture_output=True,text=True,check=True)
        report=json.loads(child.stdout)
        assert report["objective_winners"]=={"mean":"B","log2_of_mean":"B","mean_of_log2":"A"}
    with (unit/"outputs/function_values.csv").open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==10
    for row in rows:
        if float(row["x"])<=0:
            assert row["log2"]=="" and row["log2_status"]=="outside_real_domain"
        else: assert row["log2_status"]=="defined"
    groups.append("fresh process from unrelated directory and explicit undefined-value CSV status")
    return {"status":"passed","test_groups":groups}


if __name__=="__main__":
    print(json.dumps(run_tests(),indent=2))
