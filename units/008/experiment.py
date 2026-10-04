"""Euclidean vector calculations with explicit shape and zero-vector rules."""
from pathlib import Path
import json
import math
import platform
import sys
import numpy as np


def vector(value):
    if not isinstance(value,np.ndarray):
        raise TypeError("Use an explicitly constructed NumPy vector")
    if value.ndim!=1 or value.size==0:
        raise ValueError("A nonempty one-dimensional shape (D,) is required")
    if not (np.issubdtype(value.dtype,np.integer) or np.issubdtype(value.dtype,np.floating)):
        raise TypeError("Use real integer or floating components")
    if not np.isfinite(value).all():
        raise ValueError("Components must be finite")
    with np.errstate(over="ignore",invalid="ignore"):
        converted=value.astype(np.float64,copy=True)
    if not np.isfinite(converted).all():
        raise ArithmeticError("Input cannot be represented as a finite float64 vector")
    return converted


def matched(u,v):
    u=vector(u);v=vector(v)
    if u.shape!=v.shape:
        raise ValueError("Corresponding vectors need exactly equal shapes")
    return u,v


def dot(u,v):
    u,v=matched(u,v)
    result=float(np.dot(u,v))
    if not math.isfinite(result):
        raise ArithmeticError("Dot product exceeded the supported numeric range")
    return result


def norm2(u):
    u=vector(u)
    result=float(np.linalg.norm(u))
    if not math.isfinite(result):
        raise ArithmeticError("Norm exceeded the supported numeric range")
    if result==0 and np.any(u!=0):
        raise ArithmeticError("Nonzero norm underflowed this float64 calculation")
    return result


def distance(u,v):
    u,v=matched(u,v)
    return norm2(u-v)


def cosine(u,v):
    u,v=matched(u,v)
    first=norm2(u);second=norm2(v)
    if first==0 or second==0:
        raise ValueError("Cosine is undefined for a zero vector")
    denominator=first*second
    if denominator==0 or not math.isfinite(denominator):
        raise ArithmeticError("This float64 product is outside the supported range")
    value=dot(u,v)/denominator
    if not math.isfinite(value) or abs(value)>1+1e-12:
        raise ArithmeticError("Invalid cosine beyond small rounding tolerance")
    return min(1.0,max(-1.0,value))


def angle_degrees(u,v):
    return math.degrees(math.acos(cosine(u,v)))


def project_to_direction(u,v):
    u,v=matched(u,v)
    denominator=dot(v,v)
    if denominator==0:
        if np.any(v!=0):
            raise ArithmeticError("Direction norm squared underflowed this calculation")
        raise ValueError("A zero vector does not define the requested direction")
    coefficient=dot(u,v)/denominator
    if not math.isfinite(coefficient):
        raise ArithmeticError("Projection coefficient exceeded the supported numeric range")
    with np.errstate(over="ignore",invalid="ignore"):
        projection=coefficient*v
        residual=u-projection
    if not np.isfinite(projection).all() or not np.isfinite(residual).all():
        raise ArithmeticError("Projection or residual exceeded the supported numeric range")
    return coefficient,projection,residual


def describe_pair(u,v):
    u,v=matched(u,v)
    coefficient,p,r=project_to_direction(u,v)
    return {"dimension":int(u.size),"u":u.tolist(),"v":v.tolist(),"elementwise_product":(u*v).tolist(),
            "sum":(u+v).tolist(),"difference":(u-v).tolist(),"dot":dot(u,v),
            "u_norm":norm2(u),"v_norm":norm2(v),"distance":distance(u,v),
            "cosine":cosine(u,v),"angle_degrees":angle_degrees(u,v),
            "projection_coefficient":coefficient,"projection":p.tolist(),"residual":r.tolist(),
            "residual_dot_direction":dot(r,v),"squared_length_decomposition":[dot(u,u),dot(p,p),dot(r,r)]}


def standard_basis(dimension):
    if type(dimension) is not int or dimension<1:
        raise ValueError("Dimension must be a positive integer")
    return np.eye(dimension,dtype=np.float64)


def run(output_dir=None):
    base=Path(__file__).resolve().parent
    output=Path(output_dir) if output_dir is not None else base/"outputs"
    source=json.loads((base/"data/vectors.json").read_text(encoding="utf-8"))
    pairs={}
    for name,case in source.items():
        pairs[name]=describe_pair(np.array(case["u"],dtype=np.float64),np.array(case["v"],dtype=np.float64))
    s=np.array([1.,1.]);t=np.array([1.,-1.]);scale=np.array([1.,2.])
    u=np.array([2.,1.]);v=np.array([1.,3.])
    invariances={"same_direction_cosine":cosine(u,2*u),"same_direction_distance":distance(u,2*u),
                 "original_pair_cosine":cosine(u,v),"scaled_u_cosine":cosine(2*u,v),
                 "negative_u_cosine":cosine(-u,v),
                 "translated_distance":distance(u-np.array([1.,1.]),v-np.array([1.,1.])),
                 "translated_cosine":cosine(u-np.array([1.,1.]),v-np.array([1.,1.])),
                 "axis_scale_before_cosine":cosine(s,t),"axis_scale_after_cosine":cosine(s*scale,t*scale)}
    basis=standard_basis(100)
    gram=basis@basis.T
    shown=basis[:,:2]
    high_dim={"shape":list(basis.shape),"gram_equals_identity":bool(np.array_equal(gram,np.eye(100))),
              "distinct_basis_distance":distance(basis[0],basis[1]),
              "zero_rows_after_two_coordinate_display":int(np.sum(np.all(shown==0,axis=1)))}
    a=np.array([1.,2.,-1.]);norms={"L1":float(np.abs(a).sum()),"L2":norm2(a),"Linf":float(np.abs(a).max()),"L2_squared":dot(a,a)}
    report={"pairs":pairs,"invariances_and_counterexamples":invariances,"three_component_norms":norms,"high_dimensional_basis":high_dim}
    output.mkdir(parents=True,exist_ok=True)
    (output/"vector_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (output/"environment.json").write_text(json.dumps({"python":sys.version.split()[0],"numpy":np.__version__,"platform":platform.system()},indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
