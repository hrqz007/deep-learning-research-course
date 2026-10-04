"""Unit 023: a bounded, deterministic NumPy least-squares training loop.

All input validation, numerical work, and serialization precede output mutation.
Only NumPy plus Python's standard library are needed. No network or GPU.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math
from decimal import Decimal, InvalidOperation
import os
import platform
import tempfile
import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_CONFIG = {'learning_rate': 0.1, 'steps': 300, 'initial': [0., 0., 0.],
                  'gradient_point': [0.4, -0.2, 0.7], 'fd_step': 1e-5}
HEADER = ['sample_id', 'split', 'x1', 'x2', 'y']
SPLITS = ('train', 'validation', 'test')
OUTPUT_NAMES = ('summary.json', 'trajectory.csv', 'predictions.csv', 'gradient_check.csv')


def array(value, name, ndim):
    """Strict real float64 array, with explicit teaching resource/range bounds."""
    def has_bool(v):
        if isinstance(v, (bool, np.bool_)): return True
        if isinstance(v, (list, tuple)): return any(has_bool(z) for z in v)
        if isinstance(v, np.ndarray) and v.dtype.kind == 'b': return True
        return False
    if has_bool(value): raise ValueError(f'{name}: bool is not numeric data')
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.ndim != ndim or raw.size == 0:
        raise ValueError(f'{name}: require nonempty {ndim}D real numeric array; no bool/object/string')
    with np.errstate(over="ignore", under="ignore"):
        a = np.asarray(raw, dtype=np.float64)
    if np.any((raw != 0) & (a == 0)):
        raise ValueError(f"{name}: nonzero input underflows during float64 conversion")
    if a.size > 1_000_000 or not np.isfinite(a).all() or np.max(np.abs(a)) > 1e100:
        raise ValueError(f'{name}: nonfinite, range, or size limit')
    return a


def scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name}: real scalar required')
    v = float(value)
    if value != 0 and v == 0:
        raise ValueError(f"{name}: nonzero scalar underflows during float64 conversion")
    if not math.isfinite(v) or abs(v) > 1e100 or (positive and v <= 0):
        raise ValueError(f'{name}: invalid range')
    return v


def xy(X, y):
    X = array(X, 'X', 2); y = array(y, 'y', 1)
    if X.shape[0] != y.size or not 1 <= X.shape[0] <= 10000 or not 1 <= X.shape[1] <= 64:
        raise ValueError('X/y: incompatible shapes or teaching size limit')
    return X, y


def design(X):
    X = array(X, 'X', 2)
    if not 1 <= X.shape[0] <= 10000 or not 1 <= X.shape[1] <= 64:
        raise ValueError('X: teaching size limit')
    return np.column_stack((X, np.ones(X.shape[0])))


def parameters(theta, p):
    t = array(theta, 'theta', 1)
    if t.shape != (p,): raise ValueError('theta: wrong length')
    return t


def checked(value, name):
    a = np.asarray(value)
    if not np.isfinite(a).all(): raise ValueError(name + ': nonfinite calculation')
    return value


def loss_gradient(X, y, theta):
    """L=||A theta-y||^2/(2n), g=A.T@(A theta-y)/n; theta=(w,b)."""
    X, y = xy(X, y); A = design(X); t = parameters(theta, A.shape[1])
    with np.errstate(all='raise'):
        residual = checked(A @ t - y, 'residual')
        loss = float(checked(np.dot(residual, residual) / (2*y.size), 'loss'))
        grad = checked(A.T @ residual / y.size, 'gradient')
    # Explicitly catch subnormal dot-product loss that a BLAS may silently flush.
    if loss == 0.0 and np.any(residual != 0.0):
        raise ValueError('squared residual underflow')
    return loss, grad


def predict(X, theta):
    A = design(X); t = parameters(theta, A.shape[1])
    with np.errstate(all='raise'): return checked(A @ t, 'prediction')


def train(X, y, learning_rate=0.1, steps=300, initial=None):
    """Full batch descent. Row t records theta_t BEFORE its next update."""
    X, y = xy(X, y); eta = scalar(learning_rate, 'learning_rate', True)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 10000:
        raise ValueError('steps: integer in [0,10000] required')
    if (steps+1)*X.size > 10_000_000: raise ValueError('training budget exceeded')
    t = np.zeros(X.shape[1]+1) if initial is None else parameters(initial, X.shape[1]+1).copy()
    records=[]
    for step in range(steps+1):
        loss, grad = loss_gradient(X, y, t)
        norm = math.hypot(*map(float, grad)); checked(norm, 'gradient norm')
        records.append({'step': step, 'loss': loss, 'gradient_norm': norm, 'theta': t.tolist()})
        if step != steps:
            with np.errstate(all='raise'): t = checked(t - eta*grad, 'update')
    return t, records


def least_squares(X, y):
    X, y = xy(X, y); A = design(X)
    theta, _residuals, rank, singular = np.linalg.lstsq(A, y, rcond=None)
    checked(theta, 'lstsq theta'); checked(singular, 'singular values')
    loss, grad = loss_gradient(X, y, theta)  # Never infer zero loss from empty residuals.
    return theta, {'rank': int(rank), 'columns': A.shape[1], 'singular_values': singular.tolist(),
                   'loss': loss, 'gradient_norm': math.hypot(*map(float, grad))}


def gradient_check(X, y, theta, h=1e-5):
    X, y = xy(X, y); t = parameters(theta, X.shape[1]+1); h=scalar(h,'h',True)
    _, analytic = loss_gradient(X, y, t); records=[]
    denom=2*h
    if not math.isfinite(denom): raise ValueError('finite difference denominator overflow')
    for j in range(t.size):
        plus=t.copy(); minus=t.copy()
        with np.errstate(all='raise'):
            plus[j]+=h; minus[j]-=h
        if plus[j] == t[j] or minus[j] == t[j]: raise ValueError('unresolvable finite difference step')
        lp,_=loss_gradient(X,y,plus); lm,_=loss_gradient(X,y,minus)
        estimate=(lp-lm)/denom; checked(estimate,'finite difference')
        error=abs(estimate-float(analytic[j])); scaled=error/max(1.,abs(estimate),abs(float(analytic[j])))
        records.append({'coordinate':j,'analytic':float(analytic[j]),'finite_difference':estimate,
                        'absolute_error':error,'scaled_error':scaled})
    return records


def _unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('duplicate JSON key: '+key)
        out[key]=value
    return out


def checked_float_text(text):
    try:
        precise=Decimal(text)
        value=float(text)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise ValueError('invalid numeric text') from error
    if precise.is_finite() and precise != 0 and value == 0:
        raise ValueError('nonzero text underflows during float64 conversion')
    return value


def load_config(path):
    cfg=json.loads(Path(path).read_text(),object_pairs_hook=_unique,parse_float=checked_float_text)
    if not isinstance(cfg,dict) or set(cfg)!=set(DEFAULT_CONFIG): raise ValueError('exact config keys required')
    scalar(cfg['learning_rate'],'learning_rate',True); scalar(cfg['fd_step'],'fd_step',True)
    if type(cfg['steps']) is not int or not 0<=cfg['steps']<=10000: raise ValueError('steps range/type')
    parameters(cfg['initial'],3); parameters(cfg['gradient_point'],3)
    return cfg


def load_samples(path):
    rows=[]; seen=set()
    with Path(path).open(newline='') as f:
        reader=csv.reader(f,strict=True)
        if next(reader,None)!=HEADER: raise ValueError('CSV requires exact ordered, unique header: '+','.join(HEADER))
        for row in reader:
            if len(row)!=5: raise ValueError('CSV row width')
            sid,split,*raw=row
            if not sid or sid!=sid.strip() or sid in seen: raise ValueError('sample ID empty/duplicate/whitespace')
            if split not in SPLITS: raise ValueError('unknown split')
            values=[]
            for s in raw:
                if not s or s!=s.strip(): raise ValueError('blank/whitespace numeric field')
                values.append(scalar(checked_float_text(s),'CSV value'))
            rows.append({'sample_id':sid,'split':split,'x1':values[0],'x2':values[1],'y':values[2]});seen.add(sid)
            if len(rows)>10000: raise ValueError('CSV row budget')
    if any(not any(r['split']==s for r in rows) for s in SPLITS): raise ValueError('all three splits must be nonempty')
    return rows


def split_arrays(rows, split):
    selected=[r for r in rows if r['split']==split]
    return np.array([[r['x1'],r['x2']] for r in selected]),np.array([r['y'] for r in selected])


def compute(rows, cfg):
    """Only training rows may influence parameters, baseline, or gradient checks."""
    X,y=split_arrays(rows,'train')
    theta,trace=train(X,y,cfg['learning_rate'],cfg['steps'],cfg['initial'])
    reference,ref=least_squares(X,y)
    checks=gradient_check(X,y,cfg['gradient_point'],cfg['fd_step'])
    with np.errstate(all='raise'):
        baseline=float(checked(y.mean(),'baseline'))
        bound=float(checked(np.sum(design(X)**2)/len(y),'curvature bound'))
    baseline_theta=np.array([0.,0.,baseline]); scores={}; predictions=[]
    for split in SPLITS:
        XX,yy=split_arrays(rows,split)
        scores[split]={name:2*loss_gradient(XX,yy,param)[0] for name,param in
                       [('trained',theta),('reference',reference),('constant',baseline_theta)]}
        for row,pred,refpred in zip([r for r in rows if r['split']==split],predict(XX,theta),predict(XX,reference)):
            predictions.append({**row,'trained':float(pred),'reference':float(refpred),'constant':baseline})
    summary={'config':cfg,'counts':{s:sum(r['split']==s for r in rows) for s in SPLITS},
             'theta_order':['w1','w2','b'],'theta':theta.tolist(),'reference_theta':reference.tolist(),
             'reference':ref,'baseline':baseline,'mse':scores,'curvature_upper_bound':bound,
             'sufficient_step_upper_bound':2/bound,'initial_loss':trace[0]['loss'],
             'final_loss':trace[-1]['loss'],'final_gradient_norm':trace[-1]['gradient_norm'],
             'max_scaled_gradient_error':max(v['scaled_error'] for v in checks),
             'max_training_prediction_difference':float(np.max(np.abs(predict(X,theta)-predict(X,reference)))),
             'environment':{'python':platform.python_version(),'numpy':np.__version__},
             'scope':'Original deterministic synthetic split; no generalization guarantee; validation/test unused for fitting.'}
    return summary,trace,predictions,checks


def csv_text(columns, rows):
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,columns,lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue()


def serialize(result):
    summary,trace,preds,checks=result
    # Validate all nested numerical values, including CSV-only records, before any output.
    json.dumps(result,allow_nan=False)
    flat=[{'step':r['step'],'loss':r['loss'],'gradient_norm':r['gradient_norm'],
           'w1':r['theta'][0],'w2':r['theta'][1],'b':r['theta'][2]} for r in trace]
    return {'summary.json':json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',
            'trajectory.csv':csv_text(['step','loss','gradient_norm','w1','w2','b'],flat),
            'predictions.csv':csv_text(HEADER+['trained','reference','constant'],preds),
            'gradient_check.csv':csv_text(['coordinate','analytic','finite_difference','absolute_error','scaled_error'],checks)}


def reject_symlink_chain(path):
    path=Path(os.path.abspath(path))
    if any(p.is_symlink() for p in [path,*path.parents]): raise ValueError('symlink in output path')
    return path


def output_preflight(out, names, inputs=()):
    out=reject_symlink_chain(out)
    if out.exists() and not out.is_dir(): raise ValueError('output must be a directory')
    protected={Path(x).resolve() for x in inputs}
    for name in names:
        if Path(name).name!=name: raise ValueError('output name must be a basename')
        path=reject_symlink_chain(out/name)
        if path.resolve() in protected or (path.exists() and not path.is_file()):
            raise ValueError('output/input collision or non-file target')
    return out


def write_outputs(payload, out, inputs=()):
    encoded={name:value.encode('utf-8') for name,value in payload.items()}
    out=output_preflight(out,encoded,inputs)
    out.mkdir(parents=True,exist_ok=True)
    for name,value in encoded.items():
        fd,temp=tempfile.mkstemp(prefix='.'+name+'.',dir=out)
        try:
            with os.fdopen(fd,'wb') as f: f.write(value)
            os.replace(temp,out/name)
        finally:
            if os.path.exists(temp): os.unlink(temp)
    return out


def run(config=BASE/'data/config.json', samples=BASE/'data/samples.csv', output=BASE/'outputs'):
    # No mkdir, temporary files, or replacement until this entire block succeeds.
    cfg=load_config(config);rows=load_samples(samples);result=compute(rows,cfg);payload=serialize(result)
    write_outputs(payload,output,[config,samples,Path(__file__)])
    return result[0]


def teaching_payload(config=BASE/'data/config.json', samples=BASE/'data/samples.csv', results=BASE/'outputs'):
    """Fixed figures/notebook must not silently attach original captions to other runs."""
    cfg=load_config(config)
    if cfg!=DEFAULT_CONFIG: raise ValueError('fixed teaching artifact requires original config')
    raw=Path(samples).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TEACHING_DATA_SHA256:
        raise ValueError('fixed teaching artifact requires original sample file')
    result=compute(load_samples(samples),cfg);expected=serialize(result)
    # Environment version is informative, not a model assumption. A rebuild on another
    # version is allowed only if all numeric artifacts still agree at serialization level.
    for name,value in expected.items():
        actual=(Path(results)/name).read_text()
        if name=='summary.json':
            aa=json.loads(actual);ee=json.loads(value)
            aa.pop('environment',None);ee.pop('environment',None)
            if aa!=ee: raise ValueError('fixed teaching results mismatch: '+name)
        elif actual!=value: raise ValueError('fixed teaching results mismatch: '+name)
    return result


TEACHING_DATA_SHA256 = 'c1aee48bd722b484baea86078797064938724943f2433d1a628e7e53ff3c24ef'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE/'data/config.json')
    parser.add_argument('--samples',type=Path,default=BASE/'data/samples.csv')
    parser.add_argument('--output',type=Path,default=BASE/'outputs')
    args=parser.parse_args();answer=run(args.config,args.samples,args.output)
    print(json.dumps({'theta':answer['theta'],'mse':answer['mse'],
                      'gradient_error':answer['max_scaled_gradient_error']},ensure_ascii=False,allow_nan=False))
