"""Analytic, Fraction and high-precision checks for a small calculus lab."""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import json
import csv
import shutil
import subprocess
import sys
import experiment as lab


def run_tests():
    groups=[]
    assert abs(lab.forward_difference(lab.square,2,.1)-4.1)<1e-12
    assert abs(lab.central_difference(lab.square,2,.1)-4)<1e-12
    for x in [-2,-.5,0,1,3]:
        h=.125
        assert abs(lab.forward_difference(lab.cube,x,h)-(3*x*x+3*x*h+h*h))<1e-12
        assert abs(lab.central_difference(lab.cube,x,h)-(3*x*x+h*h))<1e-12
    groups.append("exact square and cube difference expansions")
    for name,function,x in [("square",lab.square,1.3),("cube",lab.cube,.7),("exp",lab.exponential,.7),("log",lab.logarithm,1.3)]:
        expected=lab.analytic_derivative(name,x)
        assert abs(lab.central_difference(function,x,1e-5)-expected)<1e-8
    with localcontext() as ctx:
        ctx.prec=70
        expected=Decimal.from_float(.7).exp()
        assert abs(Decimal.from_float(lab.analytic_derivative("exp",.7))-expected)<Decimal('5e-16')
        for denominator in [2,4,8,16,32,64]:
            x=F(1,denominator)
            assert abs(lab.analytic_derivative("log",float(x))-float(1/x))<1e-12
    groups.append("smooth derivative comparisons and 70-digit Decimal exponential reference")
    assert lab.central_difference(abs,0,.001)==0
    assert (abs(.001)-abs(0))/.001==1 and (abs(0)-abs(-.001))/.001==-1
    try:lab.analytic_derivative("abs",0)
    except ValueError:pass
    else:raise AssertionError("Cusp incorrectly declared differentiable")
    groups.append("cusp counterexample distinguishes a difference formula from a derivative")
    for bad in [0,-.1,float("nan"),float("inf")]:
        try:lab.central_difference(lab.square,1,bad)
        except ValueError:pass
        else:raise AssertionError("Invalid step accepted")
    try:lab.central_difference(lab.exponential,.7,1e-18)
    except ArithmeticError:pass
    else:raise AssertionError("Unresolvable floating-point step accepted")
    try:lab.central_difference(lab.logarithm,.1,.2)
    except ValueError:pass
    else:raise AssertionError("Difference crossed logarithm domain without rejection")
    calls=[]
    def scaled_identity(t):
        calls.append(t)
        return 1e-308*t
    try:lab.central_difference(scaled_identity,0.0,1e308)
    except ArithmeticError:pass
    else:raise AssertionError("Overflowed central denominator silently accepted")
    assert calls==[], "Range rejection must occur before evaluating the callback"
    with TemporaryDirectory() as tmp:
        copied=Path(tmp)/"experiment.py"
        shutil.copyfile(Path(lab.__file__),copied)
        data=Path(tmp)/"data";data.mkdir()
        config=json.loads((Path(lab.__file__).parent/"data/experiment_config.json").read_text())
        config["difference_steps"]=[1000.0,1e-17]
        (data/"experiment_config.json").write_text(json.dumps(config))
        subprocess.run([sys.executable,str(copied)],check=True,stdout=subprocess.DEVNULL)
        with (Path(tmp)/"outputs/difference_scan.csv").open(newline="") as handle:
            rows=list(csv.DictReader(handle))
        assert [row["status"] for row in rows]==["numeric_range_error","unresolvable_input_step"]
        assert all(row["forward"]=="" and row["central"]=="" for row in rows)
    groups.append("step positivity, resolvability, denominator, domain and correctly classified overflow checks")
    for n in [1,2,4,8,16,32]:
        left=sum(2*F(2,n)*k*F(2,n) for k in range(n))
        right=sum(2*F(2,n)*(k+1)*F(2,n) for k in range(n))
        midpoint=sum((F(2,n)*(F(k)+F(1,2)))**2*F(2,n) for k in range(n))
        assert lab.integrate(lab.flow,0,2,n,"left")==float(left)==4-4/n
        assert lab.integrate(lab.flow,0,2,n,"right")==float(right)==4+4/n
        assert lab.integrate(lab.flow,0,2,n,"midpoint")==4
        assert lab.integrate(lab.flow,0,2,n,"trapezoid")==4
        assert abs(lab.integrate(lab.square,0,2,n,"midpoint")-float(midpoint))<1e-12
    assert lab.integrate(lab.square,0,2,4,"midpoint")==2.625
    assert lab.integrate(lab.square,0,2,4,"trapezoid")==2.75
    assert lab.integrate(lab.flow,2,0,4,"midpoint")==-4
    assert lab.integrate(lab.square,1,1,4)==0
    groups.append("independent Fraction rectangle sums, trapezoids and oriented bounds")
    for n in [0,-1,1.5,True]:
        try:lab.integrate(lab.square,0,2,n)
        except ValueError:pass
        else:raise AssertionError("Invalid panel count accepted")
    assert abs(lab.integrate(lambda t:t,-1,1,16,"midpoint"))<1e-12
    assert lab.integrate(abs,-1,1,16,"midpoint")==1
    groups.append("panel-count failures and signed versus absolute area")
    for step in [F(1,10),F(1,2),F(1),F(11,10)]:
        row=lab.parameter_step(0,float(step));new_w=6*step;new_loss=(new_w-3)**2
        assert abs(row["new_w"]-float(new_w))<1e-12
        assert abs(row["new_loss"]-float(new_loss))<1e-12
    assert lab.parameter_step(0,1.1)["new_loss"]>9
    groups.append("exact parameter-step arithmetic and finite-step overshoot")
    with TemporaryDirectory() as tmp:
        first=Path(tmp)/"one";second=Path(tmp)/"two"
        assert lab.run(first)==lab.run(second)
        for p in first.iterdir():assert p.read_bytes()==(second/p.name).read_bytes()
    groups.append("complete repeated run is byte-identical")
    return {"status":"passed","test_groups":groups}


if __name__=="__main__":
    print(json.dumps(run_tests(),indent=2))
