"""Unit 027: NumPy batch backpropagation, CPU, synthetic data only.

Row-sample convention. A scalar mean or sum CE is the differentiated objective.
No training frameworks, autograd, downloads, or unit-026 imports.
"""
from pathlib import Path
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT_NAMES = ('summary.json', 'gradients.csv', 'sample_gradients.csv', 'fd_sweep.csv', 'tensors.json')


def real_array(value, name, ndim=None, bound=1e6):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise ValueError(f'{name}: real numeric values, not bool/string/object, required')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)) or np.any((raw != 0) & (result == 0)):
        raise ValueError(f'{name}: nonfinite or nonzero-to-zero conversion')
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f'{name}: expected {ndim} dimensions')
    if result.size == 0 or np.max(np.abs(result)) > bound:
        raise ValueError(f'{name}: empty or outside bound {bound}')
    return result


def scalar(value, name, low, high):
    a = real_array(value, name, ndim=0, bound=max(abs(low), abs(high)))
    v = float(a)
    if not low <= v <= high:
        raise ValueError(f'{name}: expected [{low}, {high}]')
    return v


def checked(a, name):
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{name}: nonfinite calculation')
    return a


def validate_params(params):
    if not isinstance(params, (list, tuple)) or not 1 <= len(params) <= 5:
        raise ValueError('params: 1..5 layers required')
    result = []
    previous = None
    for i, layer in enumerate(params):
        if not isinstance(layer, dict) or set(layer) != {'W', 'b'}:
            raise ValueError('each layer must contain exactly W and b')
        W = real_array(layer['W'], f'W{i+1}', 2, 10)
        b = real_array(layer['b'], f'b{i+1}', 1, 10)
        if not all(1 <= d <= 16 for d in W.shape) or b.shape != (W.shape[1],):
            raise ValueError('invalid layer dimensions or bias shape')
        if previous is not None and W.shape[0] != previous:
            raise ValueError('adjacent layer widths disagree')
        previous = W.shape[1]
        result.append({'W': W, 'b': b})
    if previous < 2:
        raise ValueError('output requires at least 2 classes')
    return result


def validate_labels(y, n, classes):
    a = np.asarray(y)
    if a.dtype.kind not in 'iu' or a.shape != (n,) or any(isinstance(x,(bool,np.bool_)) for x in np.asarray(y,dtype=object).flat):
        raise ValueError('labels must be integer vector (n,), not bool or one-hot')
    if np.any(a < 0) or np.any(a >= classes):
        raise ValueError('class index outside [0,C)')
    return a.astype(np.int64)


def forward(X, params, activation='tanh'):
    if activation not in ('tanh', 'relu', 'linear'):
        raise ValueError('activation must be tanh, relu or linear')
    layers = validate_params(params)
    X = real_array(X, 'X', 2, 10)
    if not 1 <= X.shape[0] <= 128 or X.shape[1] != layers[0]['W'].shape[0]:
        raise ValueError('invalid batch/input shape')
    H, Z = [X], []
    for j, p in enumerate(layers):
        z = checked(H[-1] @ p['W'] + p['b'], 'affine output')
        if np.max(np.abs(z)) > 1e6:
            raise ValueError('activation/logit magnitude exceeds 1e6')
        Z.append(z)
        if j < len(layers)-1:
            H.append(np.tanh(z) if activation == 'tanh' else np.maximum(z, 0) if activation == 'relu' else z.copy())
    return Z[-1], H, Z, layers


def cross_entropy(logits, y, reduction='mean'):
    z = real_array(logits, 'logits', 2, 1e6)
    n, c = z.shape
    if not (1 <= n <= 128 and 2 <= c <= 16) or reduction not in ('mean', 'sum'):
        raise ValueError('invalid logit shape or reduction')
    y = validate_labels(y, n, c)
    shifted = z - z.max(axis=1, keepdims=True)
    with np.errstate(under='ignore'):
        expz = np.exp(shifted)
    # Remove one exact exp(0)=1 before summation, preserving a tiny correct-class loss.
    maxima = np.argmax(z, axis=1)
    rest = expz.copy(); rest[np.arange(n), maxima] = 0
    logden = np.log1p(rest.sum(axis=1))
    logp = shifted - logden[:, None]
    p = expz / (1 + rest.sum(axis=1, keepdims=True))
    losses = -logp[np.arange(n), y]
    delta = p.copy()
    delta[np.arange(n), y] = 0
    delta[np.arange(n), y] = -delta.sum(axis=1)
    factor = 1/n if reduction == 'mean' else 1.0
    return float(losses.sum()*factor), delta*factor, losses, p


def loss_only(X, y, params, activation='tanh', reduction='mean'):
    return cross_entropy(forward(X, params, activation)[0], y, reduction)[0]


def loss_and_grad(X, y, params, activation='tanh', reduction='mean'):
    logits, H, Z, layers = forward(X, params, activation)
    loss, delta, losses, probabilities = cross_entropy(logits, y, reduction)
    gradients = [None]*len(layers)
    deltas = [None]*len(layers)
    for j in range(len(layers)-1, -1, -1):
        deltas[j] = delta.copy()
        gradients[j] = {'W': checked(H[j].T @ delta, 'dW'),
                        'b': checked(delta.sum(axis=0), 'db')}
        upstream = checked(delta @ layers[j]['W'].T, 'dH')
        if j:
            derivative = 1-H[j]**2 if activation == 'tanh' else (Z[j-1] > 0).astype(float) if activation == 'relu' else np.ones_like(H[j])
            delta = upstream*derivative
        else:
            dX = upstream
    return {'loss': loss, 'grads': gradients, 'dX': dX, 'losses': losses,
            'p': probabilities, 'H': H, 'Z': Z, 'delta': deltas}


def unbroadcast(gradient, original_shape):
    """Adjoint of broadcasting a nonempty original tensor to gradient.shape."""
    g = real_array(gradient, 'gradient', bound=1e12)
    if not isinstance(original_shape, tuple) or any(type(d) is not int or d < 1 for d in original_shape):
        raise ValueError('original_shape must be tuple of positive Python integers (or ())')
    if len(original_shape) > g.ndim:
        raise ValueError('target rank exceeds broadcast result rank')
    padded = (1,)*(g.ndim-len(original_shape)) + original_shape
    if any(a != b and a != 1 for a,b in zip(padded,g.shape)):
        raise ValueError('original shape cannot broadcast to gradient shape')
    axes = tuple(i for i,(a,b) in enumerate(zip(padded,g.shape)) if a == 1 and b != 1)
    return checked(g.sum(axis=axes,keepdims=True).reshape(original_shape), 'unbroadcast sum')


def coordinates(params):
    for j, layer in enumerate(params):
        for key in ('W','b'):
            for index in np.ndindex(layer[key].shape):
                yield j, key, index


def flat(params):
    return np.array([params[j][key][idx] for j,key,idx in coordinates(params)], dtype=float)


def finite_difference(X, y, params, activation='tanh', reduction='mean', h=1e-5):
    h = scalar(h, 'h', 1e-10, 1e-2)
    base = validate_params(params)
    analytic = loss_and_grad(X,y,base,activation,reduction)['grads']
    rows = []
    for j,key,index in coordinates(base):
        plus = [{k:v.copy() for k,v in p.items()} for p in base]
        minus = [{k:v.copy() for k,v in p.items()} for p in base]
        step = h * max(1.0,abs(base[j][key][index]))
        plus[j][key][index] += step; minus[j][key][index] -= step
        # Forward-only evaluations: no backprop result reused in the numerical oracle.
        numerical = (loss_only(X,y,plus,activation,reduction)-loss_only(X,y,minus,activation,reduction))/(2*step)
        actual = float(analytic[j][key][index]); absolute = abs(actual-numerical)
        relative = absolute/max(1e-8,abs(actual)+abs(numerical))
        passed = absolute <= 2e-8 + 2e-5*max(abs(actual),abs(numerical))
        rows.append({'parameter': f'{key}{j+1}[{",".join(map(str,index))}]', 'analytic':actual,
                     'numeric':float(numerical), 'absolute_error':absolute,'relative_error':relative,'pass':bool(passed)})
    return rows


def scalar_sample_oracle(X, y, params, activation='tanh'):
    """Independent Python-list loops, no matrix multiplication or loss_and_grad call.

    Returns individual gradients of each unnormalised sample loss.
    This is a separate scalar implementation, not unit 026's autodiff engine.
    """
    layers = validate_params(params)
    X = real_array(X,'X',2,10)
    if activation not in ('tanh', 'relu', 'linear') or not 1 <= len(X) <= 128 or X.shape[1] != layers[0]['W'].shape[0]:
        raise ValueError('invalid scalar-oracle activation or input shape')
    y = validate_labels(y,len(X),layers[-1]['b'].size)
    result=[]
    for x,label in zip(X,y):
        values=[x.tolist()]; pre=[]
        for j,p in enumerate(layers):
            z=[math.fsum(values[-1][r]*float(p['W'][r,k]) for r in range(len(values[-1])))+float(p['b'][k]) for k in range(len(p['b']))]
            pre.append(z)
            if j < len(layers)-1:
                values.append([math.tanh(v) if activation=='tanh' else max(v,0) if activation=='relu' else v for v in z])
        shifted=[v-max(pre[-1]) for v in pre[-1]]
        exps=[math.exp(v) for v in shifted]; denom=math.fsum(exps)
        d=[v/denom for v in exps]
        d[int(label)]=-math.fsum(d[k] for k in range(len(d)) if k != label)
        gradients=[None]*len(layers)
        for j in range(len(layers)-1,-1,-1):
            gradients[j]={'W':np.array([[a*v for v in d] for a in values[j]]),'b':np.array(d)}
            if j:
                up=[math.fsum(float(layers[j]['W'][r,k])*d[k] for k in range(len(d))) for r in range(len(values[j]))]
                d=[v*(1-values[j][r]**2 if activation=='tanh' else float(pre[j-1][r]>0) if activation=='relu' else 1) for r,v in enumerate(up)]
        result.append(flat(gradients))
    return np.array(result)


def read_number(text):
    try:
        d=Decimal(text); f=float(d)
    except (ValueError,InvalidOperation,OverflowError) as exc:
        raise ValueError('invalid numeric text') from exc
    if not d.is_finite() or not math.isfinite(f) or (d != 0 and f == 0):
        raise ValueError('nonfinite or nonzero-to-zero numeric text')
    return f


def no_duplicate_keys(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError('duplicate JSON key')
        result[k]=v
    return result


def load_inputs(data=HERE/'data/batch.csv', config=HERE/'data/config.json'):
    with Path(data).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['id','x1','x2','y']: raise ValueError('CSV header must be id,x1,x2,y')
        rows=list(reader)
    if not 1 <= len(rows) <= 128: raise ValueError('CSV requires 1..128 rows')
    ids=[]; X=[]; y=[]
    for row in rows:
        if set(row) != {'id','x1','x2','y'} or any(v is None for v in row.values()): raise ValueError('malformed CSV row')
        ident=row['id']
        if not ident or not ident.isascii() or not all(c.isalnum() or c in '-_' for c in ident) or ident in ids: raise ValueError('invalid/duplicate id')
        if not row['y'].isascii() or not row['y'].isdigit(): raise ValueError('label must be integer text')
        ids.append(ident); X.append([read_number(row['x1']),read_number(row['x2'])]); y.append(int(row['y']))
    cfg=json.loads(Path(config).read_text(encoding='utf-8'),parse_float=read_number,parse_constant=lambda s: (_ for _ in ()).throw(ValueError('invalid JSON constant')),object_pairs_hook=no_duplicate_keys)
    if not isinstance(cfg,dict) or set(cfg) != {'activation','layers','fd_h','learning_rate'}: raise ValueError('invalid config keys')
    params=validate_params(cfg['layers'])
    cfg['fd_h']=scalar(cfg['fd_h'],'fd_h',1e-10,1e-2)
    cfg['learning_rate']=scalar(cfg['learning_rate'],'learning_rate',1e-6,.5)
    X=real_array(X,'X',2,10); y=validate_labels(y,len(X),params[-1]['b'].size)
    forward(X,params,cfg['activation'])
    return ids,X,y,params,cfg


def csv_text(rows, fields):
    s=io.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    return s.getvalue()


def encode_json(value):
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def run(data=HERE/'data/batch.csv', config=HERE/'data/config.json', output=HERE/'outputs'):
    ids,X,y,params,cfg=load_inputs(data,config)
    act=cfg['activation']; n=len(X)
    mean=loss_and_grad(X,y,params,act,'mean'); summed=loss_and_grad(X,y,params,act,'sum')
    samples=scalar_sample_oracle(X,y,params,act)
    fd=finite_difference(X,y,params,act,'mean',cfg['fd_h'])
    # Split at min(2,n); for n=1 this is one microbatch, not an empty second batch.
    blocks=[np.arange(0,min(2,n))]+([np.arange(2,n)] if n>2 else [])
    micros=[flat(loss_and_grad(X[i],y[i],params,act,'mean')['grads']) for i in blocks]
    weighted=sum(len(i)/n*g for i,g in zip(blocks,micros))
    wrong=sum(micros)/len(micros)
    g=flat(mean['grads'])
    after=[{k:p[k]-cfg['learning_rate']*d[k] for k in ('W','b')} for p,d in zip(params,mean['grads'])]
    after_loss=loss_only(X,y,after,act,'mean')
    sweep=[]
    for h in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10):
        rows=finite_difference(X,y,params,act,'mean',h)
        sweep.append({'h':h,'max_absolute_error':max(r['absolute_error'] for r in rows),'max_relative_error':max(r['relative_error'] for r in rows),'failed_coordinates':sum(not r['pass'] for r in rows)})
    summary={'n':n,'widths':[params[0]['W'].shape[0]]+[len(p['b']) for p in params], 'activation':act,'parameter_count':len(g),
             'mean_loss':mean['loss'],'sum_loss':summed['loss'],'gradient_norm':float(np.linalg.norm(g)),
             'sum_vs_n_mean_max_error':float(np.max(np.abs(flat(summed['grads'])-n*g))),
             'samples_vs_mean_max_error':float(np.max(np.abs(samples.mean(axis=0)-g))),
             'microbatch_sizes':[len(i) for i in blocks], 'weighted_microbatch_max_error':float(np.max(np.abs(weighted-g))),
             'unweighted_microbatch_max_error':float(np.max(np.abs(wrong-g))),
             'fd_h':cfg['fd_h'],'fd_max_absolute_error':max(r['absolute_error'] for r in fd),
             'fd_max_relative_error':max(r['relative_error'] for r in fd),'fd_all_pass':all(r['pass'] for r in fd),
             'learning_rate':cfg['learning_rate'],'after_one_step_loss':after_loss,
             'scope':'synthetic fixed-batch derivative checks; no generalization or convergence claim'}
    sample_rows=[{'id':ident,'parameter':row['parameter'],'gradient':float(samples[i,j])} for i,ident in enumerate(ids) for j,row in enumerate(fd)]
    tensor={'X':X.tolist(),'y':y.tolist(),'H':[a.tolist() for a in mean['H']], 'Z':[a.tolist() for a in mean['Z']],
            'delta':[a.tolist() for a in mean['delta']],'probabilities':mean['p'].tolist(),'per_sample_loss':mean['losses'].tolist(),'dX':mean['dX'].tolist(),
            'gradients':[{k:v.tolist() for k,v in layer.items()} for layer in mean['grads']]}
    payload={'summary.json':encode_json(summary),'gradients.csv':csv_text(fd,list(fd[0])),
             'sample_gradients.csv':csv_text(sample_rows,['id','parameter','gradient']),
             'fd_sweep.csv':csv_text(sweep,list(sweep[0])),'tensors.json':encode_json(tensor)}
    # Input/calculation/serialization failure never changes final outputs. OS write failure
    # is not a multi-file transaction: replacements below are atomic only per file.
    destination=Path(output)
    with tempfile.TemporaryDirectory(prefix='dl027-') as tmp:
        for name,text in payload.items(): (Path(tmp)/name).write_text(text,encoding='utf-8')
        destination.mkdir(parents=True,exist_ok=True)
        for name in OUTPUT_NAMES:
            # copy staging data into a destination-local temp, so os.replace stays on one filesystem.
            with tempfile.NamedTemporaryFile(dir=destination,prefix='.027-',delete=False) as f:
                temp=Path(f.name); f.write((Path(tmp)/name).read_bytes())
            try: os.replace(temp,destination/name)
            finally: temp.unlink(missing_ok=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=HERE/'data/batch.csv')
    p.add_argument('--config',type=Path,default=HERE/'data/config.json')
    p.add_argument('--output',type=Path,default=HERE/'outputs')
    args=p.parse_args()
    try: result=run(args.data,args.config,args.output)
    except (ValueError,OSError,TypeError,OverflowError) as exc: p.exit(2,f'input/calculation error: {exc}\n')
    print(encode_json(result),end='')

if __name__=='__main__': main()
