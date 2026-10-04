"""Single-variable derivatives, finite differences and quadrature; stdlib only."""
from pathlib import Path
import csv
import json
import math
import platform
import sys


class UnresolvableStepError(ArithmeticError):
    """The requested shift equals the original floating-point input."""


def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise TypeError("A finite real scalar is required")
    if not math.isfinite(value):
        raise ValueError("Input must be finite")
    return value


def finite_result(value):
    if not math.isfinite(value):
        raise ArithmeticError("Calculation exceeded the supported numeric range")
    return value


def square(x):
    finite(x);return finite_result(x*x)


def cube(x):
    finite(x);return finite_result(x*x*x)


def exponential(x):
    finite(x);return finite_result(math.exp(x))


def logarithm(x):
    finite(x)
    if x<=0:raise ValueError("Real natural logarithm requires x>0")
    return math.log(x)


def flow(t):
    finite(t);return finite_result(2*t)


def analytic_derivative(name,x):
    finite(x)
    if name=="square":return finite_result(2*x)
    if name=="cube":return finite_result(3*x*x)
    if name=="exp":return exponential(x)
    if name=="log":
        if x<=0:raise ValueError("Logarithm derivative requires x>0")
        return finite_result(1/x)
    if name=="abs":
        if x==0:raise ValueError("Absolute value has no two-sided derivative at zero")
        return 1 if x>0 else -1
    raise ValueError("Unknown derivative example")


def checked_step(x,h):
    finite(x);finite(h)
    if h<=0:raise ValueError("This experiment requires positive step size")
    if not math.isfinite(x+h) or not math.isfinite(x-h):
        raise ArithmeticError("Shifted input exceeded the numeric range")
    if x+h==x or x-h==x:
        raise UnresolvableStepError("Step is not distinguishable at this floating-point input")


def forward_difference(function,x,h):
    checked_step(x,h)
    return finite_result((function(x+h)-function(x))/h)


def central_difference(function,x,h):
    checked_step(x,h)
    denominator=2*h
    if not math.isfinite(denominator):
        raise ArithmeticError("Central-difference denominator exceeded the numeric range")
    return finite_result((function(x+h)-function(x-h))/denominator)


def integrate(function,a,b,n,method="midpoint"):
    finite(a);finite(b)
    if type(n) is not int or n<1:
        raise ValueError("Panel count must be a positive integer")
    if method not in ["left","right","midpoint","trapezoid"]:
        raise ValueError("Unknown integration method")
    if a==b:return 0.0
    if a>b:return -integrate(function,b,a,n,method)
    width=(b-a)/n
    if not math.isfinite(width) or width==0 or a+width==a:
        raise ArithmeticError("Panel width is not usable at this floating-point scale")
    terms=[]
    if method=="trapezoid":
        terms.append(finite_result(function(a))/2)
        for k in range(1,n):terms.append(finite_result(function(a+k*width)))
        terms.append(finite_result(function(b))/2)
    else:
        offset={"left":0,"right":1,"midpoint":.5}[method]
        for k in range(n):terms.append(finite_result(function(a+(k+offset)*width)))
    return finite_result(width*math.fsum(terms))


def parameter_step(w,step):
    finite(w);finite(step)
    if step<=0:raise ValueError("Use a positive step size")
    derivative=2*(w-3)
    next_w=finite_result(w-step*derivative)
    old_loss=finite_result((w-3)**2);new_loss=finite_result((next_w-3)**2)
    return {"step":step,"old_w":w,"derivative":derivative,"new_w":next_w,"old_loss":old_loss,"new_loss":new_loss}


def run(output_dir=None):
    base=Path(__file__).resolve().parent
    output=Path(output_dir) if output_dir is not None else base/"outputs"
    config=json.loads((base/"data/experiment_config.json").read_text(encoding="utf-8"))
    x=config["difference_x"]
    truth=analytic_derivative("exp",x)
    scan=[]
    for h in config["difference_steps"]:
        try:
            fwd=forward_difference(exponential,x,h);ctr=central_difference(exponential,x,h)
            row={"h":h,"forward":fwd,"central":ctr,"forward_error":abs(fwd-truth),"central_error":abs(ctr-truth),"status":"computed"}
        except UnresolvableStepError:
            row={"h":h,"forward":None,"central":None,"forward_error":None,"central_error":None,"status":"unresolvable_input_step"}
        except ArithmeticError:
            row={"h":h,"forward":None,"central":None,"forward_error":None,"central_error":None,"status":"numeric_range_error"}
        scan.append(row)
    integration=[]
    for n in config["panel_counts"]:
        integration.append({"panels":n,"linear_left":integrate(flow,0,2,n,"left"),"linear_right":integrate(flow,0,2,n,"right"),
                            "linear_midpoint":integrate(flow,0,2,n,"midpoint"),"linear_trapezoid":integrate(flow,0,2,n,"trapezoid"),
                            "square_midpoint":integrate(square,0,2,n,"midpoint"),"square_trapezoid":integrate(square,0,2,n,"trapezoid")})
    h=.001
    cusp={"x":0,"h":h,"right_quotient":(abs(h)-abs(0))/h,"left_quotient":(abs(0)-abs(-h))/h,
          "central_difference":central_difference(abs,0,h),"two_sided_derivative":"undefined"}
    local=[{"delta":d,"true_square":square(2+d),"tangent_approximation":4+4*d,"exact_error_formula":d*d} for d in [.1,.5]]
    report={"difference_function":"exp","difference_x":x,"analytic_derivative":truth,"cusp":cusp,"local_square_approximation":local,
            "integration_exact":{"linear_0_to_2":4,"square_0_to_2":8/3,"signed_t_minus1_to1":0,"absolute_area":1},
            "parameter_steps":[parameter_step(0,step) for step in [.1,.5,1,1.1]]}
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [("difference_scan.csv",scan),("integration_scan.csv",integration)]:
        with (output/name).open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"environment.json").write_text(json.dumps({"python":sys.version.split()[0],"platform":platform.system(),"dependencies":"standard library"},indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    print(json.dumps(run(),indent=2,allow_nan=False))
