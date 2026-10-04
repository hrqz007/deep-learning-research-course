"""029 actual PyTorch CPU autograd migration and diagnostic probes.

Fixed synthetic027 data. Never installs packages, downloads data, uses a GPU,
imports another unit, or changes global dtype. Plain tensor affine layers.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import warnings
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError('PyTorch is required; see README.md for an isolated CPU installation. No package was installed by this script.') from exc
from reference import validate, numpy_reference, finite_differences

HERE=Path(__file__).resolve().parent
INPUT_SHA={'batch.csv':'42807a0698a93257e6eb239ede3d1fca4c25487fabb21df777194de74ab1074d','config.json':'949093a235057e2a6ccc725dd703814d93c5f297bd50f5dee139f13ce2e3107f'}
OUTPUT_NAMES=('summary.json','comparisons.csv','finite-difference.csv','semantic-probes.json','tensors.json')


def load_inputs(data_dir=HERE/'data'):
    root=Path(data_dir)
    for name,digest in INPUT_SHA.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('fixed teaching input changed: '+name+'; explore modified inputs through the functions, not this fixed-report entry point')
    with (root/'batch.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    cfg=json.loads((root/'config.json').read_text())
    X,y,layers=validate([[float(r['x1']),float(r['x2'])] for r in rows],[int(r['y']) for r in rows],cfg['layers'])
    return X,y,layers,cfg


def leaf(a,dtype=torch.float64):
    return torch.tensor(a,dtype=dtype,device='cpu',requires_grad=True)


def make_leaves(X,layers,dtype=torch.float64):
    return leaf(X,dtype),[{k:leaf(v,dtype) for k,v in p.items()} for p in layers]


def forward_tensors(X,params,activation='tanh',cut=None,retain=False):
    H=[X];Z=[]
    for j,p in enumerate(params):
        z=H[-1]@p['W']+p['b'];Z.append(z)
        if not bool(torch.isfinite(z).all()) or float(z.detach().abs().max()) > 1e6:raise ValueError('affine output exceeds domain')
        if retain and z.requires_grad:z.retain_grad()
        if j<len(params)-1:
            h=torch.tanh(z) if activation=='tanh' else torch.relu(z) if activation=='relu' else z.clone()
            if j==0 and cut=='detach':h=h.detach()
            if j==0 and cut=='reconstruct':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    h=torch.tensor(h,dtype=h.dtype,device='cpu',requires_grad=True)
            if retain and h.requires_grad:h.retain_grad()
            H.append(h)
    return Z[-1],H,Z


def array(t):
    return t.detach().cpu().numpy().copy()


def torch_result(X,y,layers,activation='tanh',reduction='mean',dtype=torch.float64,cut=None):
    X,y,layers=validate(X,y,layers,activation,reduction)
    if dtype not in (torch.float64,torch.float32):raise ValueError('only CPU float64 or float32 supported')
    if cut not in (None,'detach','reconstruct'):raise ValueError('invalid diagnostic cut')
    tx,params=make_leaves(X,layers,dtype);ty=torch.tensor(y,dtype=torch.int64,device='cpu')
    logits,H,Z=forward_tensors(tx,params,activation,cut,True)
    losses=F.cross_entropy(logits,ty,reduction='none');loss=losses.mean() if reduction=='mean' else losses.sum()
    loss.backward()
    values={'H0':array(tx),'loss':array(loss),'losses':array(losses),'p':array(torch.softmax(logits,dim=1))}
    for j,z in enumerate(Z,1):values[f'Z{j}']=array(z);values[f'dZ{j}']=None if z.grad is None else array(z.grad)
    for j,h in enumerate(H):
        if j:values[f'H{j}']=array(h)
        values[f'dH{j}']=None if h.grad is None else array(h.grad)
    for j,p in enumerate(params,1):
        for k,t in p.items():values[f'd{k}{j}']=None if t.grad is None else array(t.grad)
    if any(a is not None and not np.isfinite(a).all() for a in values.values()):raise ValueError('nonfinite torch result')
    return values


def compare(reference,actual):
    if set(reference)!=set(actual):raise ValueError('comparison keys differ')
    rows=[]
    for name,ref in reference.items():
        a=actual[name]
        if a is None:raise ValueError('missing gradient '+name)
        if a.shape!=ref.shape:raise ValueError('comparison shapes differ '+name)
        for idx in np.ndindex(ref.shape):
            r,t=float(ref[idx]),float(a[idx]);err=abs(r-t)
            rows.append({'tensor':name,'index':','.join(map(str,idx)),'numpy':r,'torch':t,'absolute_error':err,'pass':bool(err<=2e-12+2e-10*max(abs(r),abs(t)))})
    return rows


def expect_runtime(action):
    try:action()
    except RuntimeError as e:return str(e).split('\n')[0]
    raise AssertionError('expected RuntimeError was not raised')


def semantic_probes(X,y,layers):
    report={}
    # Storage, graph history, and identity are separate questions.
    w=leaf([2.]);h=w*w;cl=h.clone();det=h.detach();dc=h.detach().clone()
    report['storage']={'clone_shares':cl.data_ptr()==h.data_ptr(),'clone_has_history':cl.grad_fn is not None,'detach_shares':det.data_ptr()==h.data_ptr(),'detach_requires_grad':det.requires_grad,'detached_clone_shares':dc.data_ptr()==h.data_ptr()}
    converted=w.to(torch.float32)
    report['leaf']={'input':w.is_leaf,'multiply':h.is_leaf,'float_conversion':converted.is_leaf,'same_dtype_returns_same_object':w.to(torch.float64) is w,'cast_backprop':float(torch.autograd.grad(converted.sum(),w)[0])}
    # Backward accumulation uses NEW forward graphs, unchanged leaf parameters.
    w=leaf(2.); (w*w).backward();first=float(w.grad);(w*w).backward();second=float(w.grad);w.grad=None;(w*w).backward();third=float(w.grad)
    report['accumulation']={'first':first,'uncleared_second':second,'after_clear':third}
    # Saved graph buffers are freed by the usual backward call.
    w=leaf(2.);l=w*w;l.backward();report['second_backward_error']=expect_runtime(l.backward)
    w=leaf([2.]);report['leaf_inplace_error']=expect_runtime(lambda:w.add_(1))
    w=leaf([.5]);h=torch.tanh(w);l=(h*h).sum();h.add_(1);report['saved_inplace_error']=expect_runtime(l.backward)
    w=leaf([.5]);h=torch.tanh(w);l=(h*h).sum();h.detach().add_(1);report['detached_alias_error']=expect_runtime(l.backward)
    w=leaf([2.]);l=(w*w).sum()
    with torch.no_grad():w.add_(1)
    report['no_grad_before_backward_error']=expect_runtime(l.backward)
    w=leaf(2.);q=w*w
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);rebuilt=torch.tensor(q)
    report['reconstructed_scalar_error']=expect_runtime(rebuilt.backward)
    # Foreign writes are deliberately isolated to this one diagnostic.
    a=np.array([2.],dtype=np.float64);t=torch.from_numpy(a).requires_grad_();l=(t*t).sum();a[0]=3;l.backward()
    safe_a=np.array([2.],dtype=np.float64);safe=leaf(safe_a);safe_l=(safe*safe).sum();safe_a[0]=3;safe_l.backward()
    report['numpy_alias']={'stored_forward_loss':float(l.detach()),'gradient_after_foreign_mutation':float(t.grad[0]),'expected_at_original_input':4.,'copied_tensor_gradient':float(safe.grad[0]),'scope':'observed CPU behavior; never mutate foreign aliases between forward and backward'}
    w=leaf(2.)
    with torch.no_grad():z=w*3;factory=torch.tensor(1.,requires_grad=True)
    report['no_grad']={'input_still_requires_grad':w.requires_grad,'result_requires_grad':z.requires_grad,'factory_requires_grad':factory.requires_grad}
    # p=1 makes train/eval contrast deterministic; this is not a training recommendation.
    module=torch.nn.Dropout(p=1.0);inp=leaf([1.,2.]);train=module(inp)+0
    train_grad=torch.autograd.grad(train.sum(),inp)[0]
    with torch.no_grad():train_ng=module(inp)+0
    module.eval();ev=module(inp)+0;ev.sum().backward()
    with torch.no_grad():
        raw_ng=module(inp);ev_ng=raw_ng+0
    report['eval']={'training_output':array(train).tolist(),'training_input_grad':array(train_grad).tolist(),'training_no_grad_requires_grad':train_ng.requires_grad,'eval_output':array(ev).tolist(),'eval_requires_grad':ev.requires_grad,'eval_input_grad':array(inp.grad).tolist(),'eval_no_grad_requires_grad':ev_ng.requires_grad,'raw_eval_returns_input_object':raw_ng is inp,'raw_eval_input_flag_unchanged':raw_ng.requires_grad}
    w=leaf([1.,2.]);vector=w*w;seed=torch.tensor([3.,-1.],dtype=torch.float64);g=torch.autograd.grad(vector,w,grad_outputs=seed)[0]
    report['vjp']={'seed':[3.,-1.],'gradient':array(g).tolist(),'leaf_grad_remains_none':w.grad is None}
    w=leaf(.5);f=.5*(w*w+w)**2;g=torch.autograd.grad(f,w,create_graph=True)[0];gg=torch.autograd.grad(g,w)[0]
    report['higher_order']={'first':float(g.detach()),'second':float(gg),'leaf_grad_remains_none':w.grad is None}
    full=torch_result(X,y,layers)
    report['cuts']={}
    for cut in ('detach','reconstruct'):
        broken=torch_result(X,y,layers,cut=cut)
        report['cuts'][cut]={'loss_difference':float(broken['loss']-full['loss']),'missing':[k for k,v in broken.items() if v is None],'last_weight_gradient_difference':float(np.max(np.abs(broken['dW3']-full['dW3'])))}
    # Same leaves across independent microbatch graphs; one common parameter state.
    grads={}
    for mode in ('weighted','unweighted'):
        tx,params=make_leaves(X,layers);ty=torch.tensor(y,dtype=torch.int64)
        for index in (slice(0,2),slice(2,None)):
            logits,_,_=forward_tensors(tx[index],params);n=len(ty[index]);loss=F.cross_entropy(logits,ty[index]);(loss*(n/len(y) if mode=='weighted' else .5)).backward()
        grads[mode]={f'd{k}{j}':array(t.grad) for j,p in enumerate(params,1) for k,t in p.items()}
    report['microbatch']={m:max(float(np.max(np.abs(v-full[k]))) for k,v in d.items()) for m,d in grads.items()}
    # Safe plain SGD update after backward, then a new forward computation.
    tx,params=make_leaves(X,layers);ty=torch.tensor(y,dtype=torch.int64);logits,_,_=forward_tensors(tx,params);before=F.cross_entropy(logits,ty);before.backward()
    with torch.no_grad():
        for p in params:
            for t in p.values():t.add_(t.grad,alpha=-.1);t.grad=None
    after=F.cross_entropy(forward_tensors(tx,params)[0],ty)
    report['update']={'before':float(before.detach()),'after':float(after.detach()),'parameters_still_leaf':all(t.is_leaf for p in params for t in p.values())}
    # Core integer/mixed precision diagnostics are actual framework errors.
    report['integer_requires_grad_error']=expect_runtime(lambda:torch.tensor([1],dtype=torch.int64,requires_grad=True))
    report['mixed_matmul_error']=expect_runtime(lambda:torch.ones((1,2),dtype=torch.float32)@torch.ones((2,1),dtype=torch.float64))
    return report


def encode(value):return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
def csv_text(rows):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows);return out.getvalue()


def run(data_dir=HERE/'data',output=HERE/'outputs'):
    X,y,layers,cfg=load_inputs(data_dir)
    ref=numpy_reference(X,y,layers);actual=torch_result(X,y,layers);rows=compare(ref,actual)
    if not all(r['pass'] for r in rows):raise ValueError('framework/reference comparison failed')
    fd=finite_differences(X,y,layers);probes=semantic_probes(X,y,layers)
    f32=torch_result(X,y,layers,dtype=torch.float32)
    summary={'torch_version':torch.__version__,'numpy_version':np.__version__,'device':'cpu','dtype':'float64','shape':[5,2,3,2,3],'parameter_count':26,'comparison_coordinates':len(rows),'comparison_all_pass':True,'comparison_max_abs':max(r['absolute_error'] for r in rows),'loss':float(actual['loss']),'fd_coordinates':len(fd),'fd_max_abs':max(r['absolute_error'] for r in fd),'float32_loss':float(f32['loss']),'float32_vs_float64_gradient_max_abs':max(float(np.max(np.abs(f32[k]-v))) for k,v in actual.items() if k.startswith('d')),'scope':'synthetic CPU numerical/semantic checks, not training quality or cross-platform bitwise guarantees'}
    payload={'summary.json':encode(summary),'comparisons.csv':csv_text(rows),'finite-difference.csv':csv_text(fd),'semantic-probes.json':encode(probes),'tensors.json':encode({k:v.tolist() for k,v in actual.items()})}
    # No output directory created until all input, computations and serialization succeed.
    # Final replacements are per-file atomic, not a multi-file OS crash transaction.
    dest=Path(output)
    with tempfile.TemporaryDirectory(prefix='dl029-') as staging:
        for name,text in payload.items():(Path(staging)/name).write_text(text,encoding='utf-8')
        dest.mkdir(parents=True,exist_ok=True)
        for name in OUTPUT_NAMES:
            with tempfile.NamedTemporaryFile(dir=dest,prefix='.029-',delete=False) as f:
                temp=Path(f.name);f.write((Path(staging)/name).read_bytes())
            try:os.replace(temp,dest/name)
            finally:temp.unlink(missing_ok=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=HERE/'data');p.add_argument('--output',type=Path,default=HERE/'outputs');a=p.parse_args()
    torch.set_num_threads(1)
    try:r=run(a.data_dir,a.output)
    except (ValueError,OSError,RuntimeError,TypeError) as e:p.exit(2,f'input/calculation error: {e}\n')
    print(encode(r),end='')
if __name__=='__main__':main()
