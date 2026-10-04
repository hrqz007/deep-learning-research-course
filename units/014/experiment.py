"""Finite precision lab with explicit float32/float64 and Decimal references."""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse
import csv
import json
import math
import platform
import sys
import numpy as np
BASE=Path(__file__).resolve().parent


def dtype_type(dtype):
    dt=np.dtype(dtype)
    if dt not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('Only float32 and float64 are supported')
    return dt.type


def reject_bool(value):
    if isinstance(value,(bool,np.bool_)) or (isinstance(value,np.ndarray) and value.dtype.kind=='b'):
        raise TypeError('Boolean inputs are unsupported')
    if isinstance(value,(list,tuple)):
        for child in value:reject_bool(child)


def vector(value,dtype):
    reject_bool(value);raw=np.asarray(value)
    if raw.ndim!=1 or raw.size==0:raise ValueError('Expected a nonempty one-dimensional vector')
    if raw.dtype.kind not in 'iuf':raise TypeError('Expected real numeric values')
    dt=dtype_type(dtype)
    with np.errstate(over='ignore',invalid='ignore'):a=raw.astype(dt,copy=True)
    if not np.isfinite(a).all():raise ValueError('Input is not finite after dtype conversion')
    return a


def scalar(value,dtype):
    raw=np.asarray(value)
    if raw.shape!=():raise ValueError('Expected one scalar')
    return vector([value],dtype)[0]


def checked(value):
    if not np.isfinite(value).all():raise ArithmeticError('Nonfinite intermediate or result')
    return value


def naive_softmax(value,dtype='float64'):
    """Deliberate baseline: caller must inspect nonfinite output."""
    z=vector(value,dtype);dt=dtype_type(dtype)
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        e=np.exp(z)
        return e/np.sum(e,dtype=dt)


def stable_distribution(value,dtype='float64'):
    z=vector(value,dtype);dt=dtype_type(dtype);m=np.max(z)
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        shifted=checked(z-m)
        exp_shifted=checked(np.exp(shifted))
        total=checked(np.sum(exp_shifted,dtype=dt))
        if total<=0:raise ArithmeticError('Normalization denominator must be positive')
        probabilities=checked(exp_shifted/total)
        log_total=checked(np.log(total))
        log_probabilities=checked(shifted-log_total)
        logsumexp=checked(m+log_total)
    return {'probabilities':probabilities,'log_probabilities':log_probabilities,
            'logsumexp':logsumexp,'shift':m,'zero_tail_terms':int(np.sum(exp_shifted==0))}


def decimal_distribution(actual_values):
    """Reference evaluates exact values of already rounded binary inputs."""
    with localcontext() as context:
        context.prec=90
        z=[Decimal.from_float(float(x)) for x in actual_values]
        m=max(z);e=[(x-m).exp() for x in z];s=sum(e)
        return [float(x/s) for x in e],float(m+s.ln())


def cancellation(value,dtype='float64'):
    dt=dtype_type(dtype);x=scalar(value,dt)
    if x < -1:raise ValueError('sqrt(1+x) requires rounded x >= -1')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        root=checked(np.sqrt(checked(dt(1)+x)))
        direct=checked(root-dt(1))
        rationalized=checked(x/checked(root+dt(1)))
    with localcontext() as context:
        context.prec=90
        exact=Decimal.from_float(float(x))
        reference=exact/((Decimal(1)+exact).sqrt()+Decimal(1))
    return {'stored_x':float(x),'direct':float(direct),'rationalized':float(rationalized),'reference':float(reference)}


def central_exp(x,h,dtype='float64'):
    dt=dtype_type(dtype);x=scalar(x,dt);h=scalar(h,dt)
    if h<=0:raise ValueError('h must stay positive after dtype conversion')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        denom=checked(dt(2)*h)
        plus=checked(x+h);minus=checked(x-h)
        if plus==x or minus==x:raise ArithmeticError('Unresolvable input step')
        top=checked(np.exp(plus));bottom=checked(np.exp(minus))
        result=checked(checked(top-bottom)/denom)
    with localcontext() as context:
        context.prec=90
        reference=float(Decimal.from_float(float(x)).exp())
    return {'value':float(result),'reference':reference,'absolute_error':abs(float(result)-reference),
            'stored_x':float(x),'stored_h':float(h),'actual_plus_step':float(plus)-float(x),'actual_minus_step':float(x)-float(minus)}


def format_info(dtype):
    dt=dtype_type(dtype);f=np.finfo(dt)
    return {'bits':f.bits,'eps':float(f.eps),'smallest_normal':float(f.smallest_normal),
            'smallest_subnormal':float(f.smallest_subnormal),'max':float(f.max),
            'spacing_at_1':float(np.spacing(dt(1))),'spacing_at_2':float(np.spacing(dt(2))),
            'one_plus_half_eps_equals_one':bool(dt(1)+dt(f.eps/2)==dt(1))}


def run(output_dir=None):
    config=json.loads((BASE/'data/config.json').read_text())
    report={'formats':{},'softmax':{},'cancellation':{},'cast_information_loss':{}}
    scans=[]
    for dtype in ['float32','float64']:
        report['formats'][dtype]=format_info(dtype)
        report['softmax'][dtype]={}
        for name,values in config['logit_cases'].items():
            actual=vector(values,dtype);naive=naive_softmax(actual,dtype);stable=stable_distribution(actual,dtype)
            ref,ref_lse=decimal_distribution(actual)
            report['softmax'][dtype][name]={'stored_logits':actual.tolist(),'naive_all_finite':bool(np.isfinite(naive).all()),
                'naive_probabilities':[float(v) if np.isfinite(v) else None for v in naive],
                'stable_probabilities':stable['probabilities'].tolist(),'stable_log_probabilities':stable['log_probabilities'].tolist(),
                'logsumexp':float(stable['logsumexp']),'zero_tail_terms':stable['zero_tail_terms'],
                'probability_sum':float(np.sum(stable['probabilities'],dtype=dtype_type(dtype))),
                'reference_probabilities':ref,'max_absolute_error':float(np.max(np.abs(stable['probabilities'].astype(float)-ref))),
                'logsumexp_absolute_error':abs(float(stable['logsumexp'])-ref_lse)}
        report['cancellation'][dtype]=[cancellation(x,dtype) for x in config['cancellation_inputs']]
        report['cast_information_loss'][dtype]=stable_distribution(config['rounded_logits'],dtype)['probabilities'].tolist()
        for h in config['difference_steps']:
            row={'dtype':dtype,'requested_h':h}
            try:row.update(central_exp(config['difference_x'],h,dtype),status='computed')
            except ArithmeticError as error:row.update(value=None,reference=None,absolute_error=None,stored_x=None,stored_h=None,actual_plus_step=None,actual_minus_step=None,status=str(error))
            scans.append(row)
    report['associativity']={'left':float((np.float64(1e16)+np.float64(-1e16))+np.float64(1)),
                             'right':float(np.float64(1e16)+(np.float64(-1e16)+np.float64(1))),
                             'fsum':math.fsum([1e16,-1e16,1])}
    output=Path(output_dir) if output_dir is not None else BASE/'outputs';output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (output/'difference_scan.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scans[0]));writer.writeheader();writer.writerows(scans)
    (output/'environment.json').write_text(json.dumps({'python':sys.version.split()[0],'numpy':np.__version__,'platform':platform.system(),'device':'CPU','network_used':False},indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    print(json.dumps(run(args.output),ensure_ascii=False,indent=2,allow_nan=False))
