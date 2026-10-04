"""025: transparent forward calculations and bounded, exhaustive candidate search.
Core APIs accept general numeric arrays. The reproducible teaching run deliberately
requires the exact supplied configuration and four truth-table rows; it never
silently combines changed inputs with fixed teaching figures or commentary.
"""
from pathlib import Path
from decimal import Decimal
import argparse, csv, hashlib, io, itertools, json, math, os
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = {"schema": 1, "affine_grid": [-2, -1, 0, 1, 2], "scales": [0, 1, 2, 4, 8], "threshold_logit": 0}
SAMPLE_TEXT = "id,x1,x2,y\na,0,0,0\nb,0,1,1\nc,1,0,1\nd,1,1,0\n"
OUTPUT_NAMES = ("summary.json", "forward.csv", "search.csv", "depth.csv")

def _no_bool(x):
    if isinstance(x, (bool, np.bool_)): raise ValueError("boolean is not a real-valued coordinate")
    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'b': raise ValueError("boolean array")
        if x.dtype.kind == 'O': raise ValueError("object array")
    elif isinstance(x, (list, tuple)):
        for v in x: _no_bool(v)

def real_array(value, name, ndim):
    _no_bool(value)
    try: a = np.asarray(value)
    except (TypeError, ValueError) as e: raise ValueError(name + ": rectangular numeric array required") from e
    if a.dtype.kind not in 'iuf' or a.ndim != ndim or a.size == 0 or a.size > 2_000_000:
        raise ValueError(name + ": nonempty real array with correct dimensions required")
    if not np.isfinite(a).all() or np.max(np.abs(a)) > 1e100:
        raise ValueError(name + ": finite magnitude at most 1e100 required")
    with np.errstate(under='ignore'):
        converted = a.astype(np.float64)
    if np.any((a != 0) & (converted == 0)):
        raise ValueError(name + ": a nonzero coordinate underflows during float64 conversion")
    return converted

def checked(a, name):
    if not np.isfinite(a).all(): raise ValueError(name + ": calculation overflowed")
    return a

def validate_layers(layers, input_width=None):
    if not isinstance(layers, (list, tuple)) or not 1 <= len(layers) <= 10: raise ValueError("1 to 10 layers required")
    ans=[]
    for k, pair in enumerate(layers):
        if not isinstance(pair, (list, tuple)) or len(pair)!=2: raise ValueError("each layer is (W,b)")
        W=real_array(pair[0], 'W', 2); b=real_array(pair[1], 'b', 1)
        if max(W.shape)>128 or W.shape[1]!=len(b) or (input_width is not None and W.shape[0]!=input_width):
            raise ValueError("incompatible layer shapes or width above 128")
        input_width=W.shape[1]; ans.append((W,b))
    return ans

def forward(X, layers, activation='relu'):
    """Rows are samples; every bias is (width,); final layer is always affine."""
    X=real_array(X,'X',2)
    if X.shape[0]>100_000: raise ValueError("at most 100000 samples")
    if activation not in ('relu','identity'): raise ValueError("activation must be relu or identity")
    layers=validate_layers(layers,X.shape[1]); h=X; trace=[]
    with np.errstate(over='ignore',invalid='ignore'):
        for k,(W,b) in enumerate(layers):
            z=checked(h@W+b,'affine forward')
            h=np.maximum(z,0) if activation=='relu' and k<len(layers)-1 else z.copy()
            trace.append({'z':z,'h':h})
    return h, trace

def parameter_count(widths):
    if not isinstance(widths,(list,tuple)) or not 2<=len(widths)<=11: raise ValueError("2 to 11 widths required")
    if any(type(v) is not int or not 1<=v<=128 for v in widths): raise ValueError("integer widths 1..128 required")
    return sum((a+1)*b for a,b in zip(widths,widths[1:]))

def collapse_affine(layers):
    layers=validate_layers(layers); W,b=(v.copy() for v in layers[0])
    with np.errstate(over='ignore',invalid='ignore'):
        for V,c in layers[1:]:
            W=checked(W@V,'collapsed W'); b=checked(b@V+c,'collapsed b')
    return W,b

def sigmoid(z):
    z=real_array(z,'logits',1); e=np.exp(-np.abs(z))
    return np.where(z>=0,1/(1+e),e/(1+e))

def binary_metrics(z,y):
    z=real_array(z,'logits',1); y=real_array(y,'labels',1)
    if z.shape!=y.shape or not np.isin(y,[0,1]).all(): raise ValueError("matching binary labels required")
    # Stable even when the correct-class probability rounds to 1.
    losses=(1-y)*np.maximum(z,0)+y*np.maximum(-z,0)+np.log1p(np.exp(-np.abs(z)))
    loss=float(np.mean(losses)); accuracy=float(np.mean((z>=0)==y))
    if not math.isfinite(loss): raise ValueError("nonfinite loss")
    return {'loss':loss,'accuracy':accuracy}

def xor_layers(scale=1):
    a=real_array([scale],'scale',1)
    if a[0]<0 or a[0]>100: raise ValueError("scale must lie in [0,100]")
    return [(np.ones((2,2)),np.array([0.,-1.])),(np.array([[2.],[-4.]])*a[0],np.array([-1.])*a[0])]

def triangle(t, depth=1):
    t=real_array(t,'t',1)
    if type(depth) is not int or not 1<=depth<=12 or np.any((t<0)|(t>1)): raise ValueError("t in [0,1] and integer depth 1..12 required")
    h=t.copy()
    for _ in range(depth): h=2*np.maximum(h,0)-4*np.maximum(h-.5,0)
    return h

def duplicate_safe(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise ValueError('duplicate JSON key: '+k)
        d[k]=v
    return d

def load_inputs(samples=ROOT/'data/samples.csv',config=ROOT/'data/config.json'):
    samples=Path(samples); config=Path(config)
    if samples.stat().st_size>10000 or config.stat().st_size>10000: raise ValueError('input too large')
    def invalid_constant(v): raise ValueError('nonfinite JSON constant: '+v)
    cfg=json.loads(config.read_text(encoding='utf-8'),object_pairs_hook=duplicate_safe,parse_constant=invalid_constant,parse_float=Decimal)
    if type(cfg) is not dict or set(cfg)!=set(DEFAULT_CONFIG): raise ValueError('exact config keys required')
    _no_bool(list(cfg.values()))
    if cfg!=DEFAULT_CONFIG: raise ValueError('fixed teaching run requires the supplied default config; use core APIs for exploration')
    # Decimal token comparison above precedes ANY float conversion. Even a tiny
    # nonzero token, or a near-integer that float would round, must not pass.
    # Equal decimal spellings (such as 0e-400) normalize to known safe types.
    cfg={key:list(value) if isinstance(value,list) else value for key,value in DEFAULT_CONFIG.items()}
    with samples.open(newline='',encoding='utf-8') as f: rows=list(csv.reader(f,strict=True))
    if not rows or rows[0]!=['id','x1','x2','y']: raise ValueError('exact unique CSV header required')
    if len(rows)!=5 or any(len(r)!=4 for r in rows): raise ValueError('exactly four complete rows required')
    if len({r[0] for r in rows[1:]})!=4: raise ValueError('duplicate sample ID')
    try: X=np.array([[float(r[1]),float(r[2])] for r in rows[1:]]); y=np.array([float(r[3]) for r in rows[1:]])
    except ValueError as e: raise ValueError('invalid numeric CSV field') from e
    real_array(X,'X',2); real_array(y,'y',1)
    # Byte guard also covers row order, labels, identifiers and decimal spellings.
    if samples.read_bytes()!=SAMPLE_TEXT.encode('utf-8'): raise ValueError('fixed teaching data bytes changed')
    return X,y,cfg,[r[0] for r in rows[1:]]

def calculate(X,y,cfg,ids):
    # Do not expose fixed result labels for a caller's unrelated data.
    if cfg!=DEFAULT_CONFIG or ids!=['a','b','c','d'] or not np.array_equal(X,[[0,0],[0,1],[1,0],[1,1]]) or not np.array_equal(y,[0,1,1,0]):
        raise ValueError('calculate requires the fixed teaching inputs')
    z,trace=forward(X,xor_layers()); zi,_=forward(X,xor_layers(),activation='identity')
    W,b=collapse_affine(xor_layers())
    searches=[]
    for w1,w2,bias in itertools.product(cfg['affine_grid'],repeat=3):
        m=binary_metrics(X@np.array([w1,w2])+bias,y)
        searches.append({'family':'affine','w1':w1,'w2':w2,'bias':bias,'scale':'',**m})
    best=min(searches,key=lambda r:r['loss'])
    for scale in cfg['scales']:
        q,_=forward(X,xor_layers(scale)); m=binary_metrics(q[:,0],y)
        searches.append({'family':'relu_scale','w1':'','w2':'','bias':'','scale':scale,**m})
    best_relu=min(searches[125:],key=lambda r:r['loss'])
    records=[]
    p=sigmoid(z[:,0]); p8=sigmoid(8*z[:,0])
    for i,name in enumerate(ids):
        records.append(dict(id=name,x1=float(X[i,0]),x2=float(X[i,1]),y=int(y[i]),a1=float(trace[0]['z'][i,0]),a2=float(trace[0]['z'][i,1]),h1=float(trace[0]['h'][i,0]),h2=float(trace[0]['h'][i,1]),logit=float(z[i,0]),probability=float(p[i]),prediction=int(z[i,0]>=0),identity_logit=float(zi[i,0]),scale8_probability=float(p8[i])))
    t=np.linspace(0,1,17); depths=[{'t':float(v),'depth1':float(a),'depth2':float(b),'depth3':float(c)} for v,a,b,c in zip(t,triangle(t,1),triangle(t,2),triangle(t,3))]
    summary={'schema':1,'sample_count':4,'parameter_count_2_2_1':parameter_count([2,2,1]),'parameter_count_2_3_2_1':parameter_count([2,3,2,1]),'relu_scale1':binary_metrics(z[:,0],y),'identity_same_parameters':binary_metrics(zi[:,0],y),'collapsed_weight':W.tolist(),'collapsed_bias':b.tolist(),'affine_candidate_count':125,'affine_best_loss':best,'affine_best_accuracy':max(r['accuracy'] for r in searches[:125]),'relu_candidate_count':5,'relu_best_loss':best_relu,'center_triangle_logit':1.0,'center_absolute_difference_logit':-1.0,'scope':'Synthetic four-corner truth table; candidate search, not unconstrained training or generalization evidence.'}
    return summary,records,searches,depths

def json_bytes(obj): return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')

def csv_bytes(rows):
    if not rows: raise ValueError('empty result')
    # Validate the full object before CSV's permissive string conversion.
    json_bytes(rows)
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
    return f.getvalue().encode('utf-8')

def prepare(samples=ROOT/'data/samples.csv',config=ROOT/'data/config.json'):
    inputs=load_inputs(samples,config); result=calculate(*inputs)
    payload={OUTPUT_NAMES[0]:json_bytes(result[0])}
    for name,rows in zip(OUTPUT_NAMES[1:],result[1:]): payload[name]=csv_bytes(rows)
    return inputs,result,payload

def verify_teaching_artifacts():
    """Call before ANY figure/Notebook display or write, including fixed claims."""
    inputs,result,payload=prepare()
    for name,data in payload.items():
        if (ROOT/'outputs'/name).read_bytes()!=data: raise ValueError('published result does not match current default calculation: '+name)
    return inputs,result,payload

def safe_destination(output, names, protected=()):
    out=Path(os.path.abspath(output))
    for p in (out,*out.parents):
        if p.is_symlink(): raise ValueError('output path cannot contain symlinks')
    if out.exists() and not out.is_dir(): raise ValueError('output must be a directory')
    resolved={Path(p).resolve() for p in protected}
    for name in names:
        p=out/name
        if p.is_symlink() or (p.exists() and not p.is_file()) or p.resolve() in resolved: raise ValueError('unsafe output target')
    return out

def run(output=ROOT/'outputs',samples=ROOT/'data/samples.csv',config=ROOT/'data/config.json'):
    inputs,result,payload=prepare(samples,config) # All input, numeric and serialization work precedes filesystem writes.
    out=safe_destination(output,payload,protected=(samples,config))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in payload.items(): (out/name).write_bytes(data)
    return result[0]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=ROOT/'outputs'); p.add_argument('--samples',type=Path,default=ROOT/'data/samples.csv'); p.add_argument('--config',type=Path,default=ROOT/'data/config.json'); a=p.parse_args()
    print(json.dumps(run(a.output,a.samples,a.config),ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__': main()
