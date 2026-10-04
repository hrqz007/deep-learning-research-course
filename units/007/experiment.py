"""Scalar algebra, domains and order-of-operations examples; stdlib only."""
from pathlib import Path
import csv
import json
import math
import platform
import sys


def real_number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise TypeError("A real numerical input is required")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ArithmeticError("Input is outside the supported floating-point range") from exc
    if not finite:
        raise ValueError("The computational example requires finite inputs")
    return value


def finite_result(value):
    """Reject nonfinite results; this is not a universal roundoff detector."""
    if not math.isfinite(value):
        raise ArithmeticError("Result is outside the supported floating-point range")
    return value


def solve_affine(a,b,c):
    for value in [a,b,c]:real_number(value)
    if a==0:
        return {"kind":"all_real" if b==c else "no_solution","value":None}
    result=(c-b)/a
    finite_result(result)
    if result == 0 and c != b:
        raise ArithmeticError("Nonzero solution rounded to zero")
    return {"kind":"unique","value":result}


def affine(x,a=2,b=1):
    for value in [x,a,b]:real_number(value)
    return finite_result(a*x+b)


def square(x):
    real_number(x)
    return finite_result(x*x)


def power_two(x):
    real_number(x)
    try:
        result=math.pow(2,x)
    except OverflowError as exc:
        raise ArithmeticError("Mathematical 2**x exists; floating-point range exceeded") from exc
    if result==0 or not math.isfinite(result):
        raise ArithmeticError("Mathematical 2**x is positive; floating-point range exceeded")
    return result


def real_log(x,base=2):
    real_number(x);real_number(base)
    if x<=0 or base<=0 or base==1:
        raise ValueError("Real logarithm requires x>0, base>0 and base!=1")
    return finite_result(math.log(x)/math.log(base))


def rectifier(x):
    real_number(x)
    if x<0:return 0
    return x


def positive_loss_summary(values):
    if not values:raise ValueError("Need at least one positive loss")
    for value in values:
        real_number(value)
        if value<=0:raise ValueError("This logarithmic comparison requires positive losses")
    mean=finite_result(sum(values)/len(values))
    mean_logs=sum(real_log(v,2) for v in values)/len(values)
    return {"mean":mean,"log2_of_mean":real_log(mean,2),"mean_of_log2":mean_logs}


def run(output_dir=None):
    base=Path(__file__).resolve().parent
    output=Path(output_dir) if output_dir is not None else base/"outputs"
    output.mkdir(parents=True,exist_ok=True)
    with (base/"data/input_grid.csv").open(newline="",encoding="utf-8") as handle:
        xs=[float(row["x"]) for row in csv.DictReader(handle)]
    rows=[]
    for x in xs:
        row={"x":x,"affine_2x_plus_1":affine(x),"square":square(x),"power_2":power_two(x),"rectifier":rectifier(x),
             "log2":real_log(x) if x>0 else None,"log2_status":"defined" if x>0 else "outside_real_domain"}
        rows.append(row)
    with (output/"function_values.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    losses={"A":positive_loss_summary([1,9]),"B":positive_loss_summary([4,4])}
    solutions=[solve_affine(2,1,7),solve_affine(0,3,3),solve_affine(0,3,4)]
    unit_rows=[]
    for cm in [1,2,3]:
        mm=10*cm
        unit_rows.append({"cm":cm,"mm":mm,"old_prediction":affine(cm,2,1),"wrong_old_coefficient":affine(mm,2,1),"converted_prediction":affine(mm,.2,1)})
    winners={key:min(losses,key=lambda name:losses[name][key]) for key in ["mean","log2_of_mean","mean_of_log2"]}
    report={"equation_solutions":solutions,"unit_conversion":unit_rows,"loss_objectives":losses,
            "objective_winners":winners,
            "composition_at_2":{"f_of_g":affine(square(2)),"g_of_f":square(affine(2))},
            "summation":{"original_sum":sum([2,4,6]),"shift_each":sum(v+1 for v in [2,4,6]),"shift_once":sum([2,4,6])+1},
            "python_precedence":{"parenthesized_negative_base":(-2)**2,"unparenthesized_negative":-2**2}}
    (output/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (output/"environment.json").write_text(json.dumps({"python":sys.version.split()[0],"platform":platform.system(),"dependencies":"standard library"},indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
