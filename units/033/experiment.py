"""033: CPU momentum, schedule units, complete backprop trace and exact resume.

Fixed CLI inputs are byte-guarded. Public helpers validate stated small domains.
No data download, GPU, service, hidden cross-unit import, or pickle loading.
"""
from pathlib import Path
import argparse, copy, csv, hashlib, io, json, math, os, sys, tempfile
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
INPUT_SHA256 = {'data/config.json': '2593197098ed324da99f782a1a45c38d39feebf37dd54da2bd80cb712cd6a9fd', 'data/hand-example.json': '59683cccabc78b9840bf5c10d14ac7313a3e3f7837449d74ede946ad55eaa1e8', 'data/samples.csv': '6b845cc6764a573f1736c4fda7c817e1bbef72e7d535b7104fb86c2d34e3e949'}  # Filled at authoring time; intentionally immutable in the fixed driver.
MODES = ('sgd_fixed', 'momentum_fixed', 'momentum_cosine', 'momentum_warmup_cosine')
DTYPE = torch.float64


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def integer(x, name, low, high):
    if type(x) is not int or not low <= x <= high:
        raise ValueError(f'{name} must be integer in [{low},{high}]')
    return x


def real(x, name, low, high):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise ValueError(name+' must be a finite real scalar')
    if not math.isfinite(float(x)) or not low <= float(x) <= high:
        raise ValueError(name+f' must be in [{low},{high}]')
    return float(x)


def array(x, name, shape=None):
    def has_bool(v):
        return isinstance(v,(bool,np.bool_)) or (isinstance(v,(list,tuple)) and any(has_bool(a) for a in v))
    if has_bool(x):raise ValueError(name+' must not contain booleans')
    raw = np.asarray(x)
    if raw.dtype.kind not in 'fiu' or raw.dtype.kind == 'b':
        raise ValueError(name+' must contain non-boolean real numbers')
    out = np.array(raw, dtype=np.float64, copy=True)
    if shape is not None and out.shape != shape:
        raise ValueError(name+f' shape must be {shape}')
    if not np.isfinite(out).all() or np.max(np.abs(out), initial=0) > 100:
        raise ValueError(name+' must be finite and bounded by 100')
    return out


def validate_batch(X, y):
    X = array(X, 'X')
    if X.ndim != 2 or X.shape[1] != 2 or not 1 <= len(X) <= 128:
        raise ValueError('X shape must be (n,2), 1<=n<=128')
    return X, array(y, 'y', (len(X), 1))


def read_inputs():
    for name, sha in INPUT_SHA256.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != sha:
            raise ValueError('fixed input changed: '+name)
    cfg = json.loads((HERE/'data/config.json').read_text())
    hand = json.loads((HERE/'data/hand-example.json').read_text())
    with (HERE/'data/samples.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    X, y = validate_batch([[float(r['x1']),float(r['x2'])] for r in rows], [[float(r['y'])] for r in rows])
    if [int(r['id']) for r in rows] != list(range(48)):
        raise ValueError('fixed IDs must be 0..47')
    return cfg, hand, X, y


def forward(theta, X):
    """Internal tensor expression. Public callers validate shape/range first."""
    Z = X @ theta[:4].reshape(2,2) + theta[4:6]
    H = torch.relu(Z)
    P = H @ theta[6:8,None] + theta[8]
    return Z, H, P


def manual_backward(theta, X, y):
    """Independent NumPy chain/path calculation, including per-sample contributions."""
    X,y = validate_batch(X,y); t = array(theta,'theta',(9,)); n = len(X)
    W=t[:4].reshape(2,2); b=t[4:6]; v=t[6:8]; c=t[8]
    Z=X@W+b; H=np.maximum(Z,0); P=H@v[:,None]+c; R=P-y
    dP=R/n; mask=(Z>0).astype(float); dH=dP*v; dZ=dH*mask
    contributions=np.concatenate([np.einsum('ni,nj->nij',X,dZ).reshape(n,4),dZ,dP*H,dP],axis=1)
    grad=contributions.sum(axis=0)
    return {'shapes':{'X':list(X.shape),'W':[2,2],'b':[2],'Z':list(Z.shape),'H':list(H.shape),'v':[2],'c':[], 'P':list(P.shape),'y':list(y.shape),'per_sample_parameter_contribution':[n,9]},
      'theta':t.tolist(),'X':X.tolist(),'y':y.tolist(),'Z':Z.tolist(),'H':H.tolist(),'P':P.tolist(),'residual':R.tolist(),'per_sample_half_squared_error':(R[:,0]**2/2).tolist(),'loss':float(np.mean(R**2)/2),
      'local_derivatives':{'dL_dsample_loss':[1/n]*n,'dsample_loss_dP':R.tolist(),'dP_dH':np.tile(v,(n,1)).tolist(),'dP_dv':H.tolist(),'dP_dc':[1.]*n,'dH_dZ':mask.tolist(),'dZ_dW_input':X.tolist(),'dZ_db':np.ones_like(Z).tolist()},
      'chain':{'dL_dP':dP.tolist(),'dL_dH':dH.tolist(),'dL_dZ':dZ.tolist(),'dL_dX':(dZ@W.T).tolist()},'per_sample_parameter_contribution':contributions.tolist(),'gradient':grad.tolist()}


def trace_steps(hand):
    required={'X','y','theta','learning_rates','momentum'}
    if type(hand) is not dict or set(hand)!=required: raise ValueError('hand schema')
    X,y=validate_batch(hand['X'],hand['y']); t=array(hand['theta'],'theta',(9,))
    mu=real(hand['momentum'],'momentum',0,.99)
    rates=hand['learning_rates']
    if type(rates) is not list or not 1<=len(rates)<=10:raise ValueError('1..10 learning rates')
    rates=[real(v,'learning_rate',1e-8,.2) for v in rates]
    p=torch.nn.Parameter(torch.tensor(t,dtype=DTYPE)); opt=torch.optim.SGD([p],lr=rates[0],momentum=mu)
    tx=torch.tensor(X,dtype=DTYPE);ty=torch.tensor(y,dtype=DTYPE)
    traces=[];buffer=np.zeros(9)
    for k,lr in enumerate(rates):
        m=manual_backward(p.detach().numpy(),X,y); opt.zero_grad(set_to_none=True)
        Z,H,P=forward(p,tx);L=(P-ty).square().mean()/2;L.backward()
        grad=p.grad.detach().numpy().copy()
        if not np.allclose(grad,m['gradient'],rtol=1e-13,atol=1e-14):raise ArithmeticError('manual gradient mismatch')
        previous=p.detach().numpy().copy();oldbuffer=buffer.copy(); buffer=mu*buffer+grad
        opt.param_groups[0]['lr']=lr;opt.step()
        if not np.allclose(p.detach().numpy(),previous-lr*buffer,rtol=0,atol=2e-16):raise ArithmeticError('momentum mismatch')
        next_forward=manual_backward(p.detach().numpy(),X,y)
        m.update({'update_index':k,'learning_rate':lr,'momentum':mu,'buffer_before':oldbuffer.tolist(),'buffer_after':buffer.tolist(),'autograd_gradient':grad.tolist(),'delta_theta':(-lr*buffer).tolist(),'theta_after':p.detach().tolist(),'next_forward':{key:next_forward[key] for key in ['Z','H','P','residual','loss']}})
        traces.append(m)
    return {'parameter_order':['W11','W12','W21','W22','b1','b2','v1','v2','c'],'relu_zero_derivative_convention':0,'same_batch_at_every_step':True,'steps':traces}


def schedule_lr(k, total, peak, floor, warmup=0):
    """LR used by zero-based optimizer update k; calls >= total clamp to floor.
    At warmup=0, update0=peak, update(total-1)=floor.
    At warmup>0, updates0..warmup-1 rise to peak; final update=floor.
    """
    integer(k,'k',0,1000000);integer(total,'total',2,10000)
    integer(warmup,'warmup',0,total-1);peak=real(peak,'peak',1e-8,.2);floor=real(floor,'floor',0,peak)
    if k>=total:return floor
    if warmup and k<warmup:return peak*(k+1)/warmup
    progress=(k-warmup+1)/(total-warmup) if warmup else k/(total-1)
    return floor+.5*(peak-floor)*(1+math.cos(math.pi*progress))


def quadratic_run(steps=80, lr=.045, momentum=.8, schedule='fixed'):
    integer(steps,'steps',2,1000);lr=real(lr,'lr',1e-8,.06);mu=real(momentum,'momentum',0,.99)
    if schedule not in ('fixed','cosine','warmup_cosine'):raise ValueError('unknown schedule')
    p=torch.nn.Parameter(torch.tensor([3.,1.],dtype=DTYPE));o=torch.optim.SGD([p],lr=lr,momentum=mu)
    rows=[{'update':0,'x':3.,'y':1.,'loss':24.5,'lr_used':None}]
    for k in range(steps):
        alpha=lr if schedule=='fixed' else schedule_lr(k,steps,lr,.002,min(12,steps-1) if schedule=='warmup_cosine' else 0)
        o.zero_grad(set_to_none=True);loss=(p[0]**2+40*p[1]**2)/2;loss.backward();o.param_groups[0]['lr']=alpha;o.step()
        q=p.detach().tolist();rows.append({'update':k+1,'x':q[0],'y':q[1],'loss':float((q[0]**2+40*q[1]**2)/2),'lr_used':alpha})
    if not all(math.isfinite(r['loss']) for r in rows):raise ArithmeticError('quadratic non-finite')
    return rows


def train_config(cfg, X, y):
    X,y=validate_batch(X,y)
    if type(cfg) is not dict:raise ValueError('config dict required')
    needed={'seed','epochs','microbatch_size','accumulation','peak_lr','min_lr','warmup_updates','momentum','initial_theta'}
    if not needed<=set(cfg):raise ValueError('config missing required fields')
    integer(cfg['seed'],'seed',0,2**31-1);integer(cfg['epochs'],'epochs',1,50)
    integer(cfg['microbatch_size'],'microbatch_size',1,len(X));integer(cfg['accumulation'],'accumulation',1,32)
    effective=cfg['microbatch_size']*cfg['accumulation']
    if len(X)%effective:raise ValueError('this experiment requires complete equal-size accumulation windows')
    total=cfg['epochs']*(len(X)//effective)
    integer(total,'total_updates',2,10000);real(cfg['peak_lr'],'peak_lr',1e-8,.2);real(cfg['min_lr'],'min_lr',0,cfg['peak_lr'])
    integer(cfg['warmup_updates'],'warmup_updates',0,total-1);real(cfg['momentum'],'momentum',0,.99);array(cfg['initial_theta'],'initial_theta',(9,))
    return X,y,effective,total


def canonical(obj):
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def seal(payload):
    return {'payload':payload,'sha256':hashlib.sha256(canonical(payload)).hexdigest()}


def validate_checkpoint(checkpoint, cfg, X, y, mode):
    """Small, self-generated JSON only. Checksum detects accidental changes, not an adversary."""
    X,y,effective,total=train_config(cfg,X,y)
    if type(checkpoint) is not dict or set(checkpoint)!={'payload','sha256'}:raise ValueError('checkpoint envelope')
    p=checkpoint['payload']
    if hashlib.sha256(canonical(p)).hexdigest()!=checkpoint['sha256']:raise ValueError('checkpoint checksum')
    expected={'format','torch_version','config','data_sha256','mode','completed_epochs','theta','momentum_buffer','optimizer_group','scheduler','shuffle_rng','history','orders'}
    if type(p) is not dict or set(p)!=expected:raise ValueError('checkpoint payload schema')
    if p['format']!='dl033-epoch-boundary-v1' or p['torch_version']!=torch.__version__:raise ValueError('checkpoint format/version')
    if canonical(p['config'])!=canonical(cfg) or p['mode']!=mode:raise ValueError('checkpoint config/mode mismatch')
    if p['data_sha256']!=hashlib.sha256(canonical({'X':X.tolist(),'y':y.tolist()})).hexdigest():raise ValueError('checkpoint data mismatch')
    e=integer(p['completed_epochs'],'completed_epochs',1,cfg['epochs']-1);k=e*len(X)//effective
    array(p['theta'],'checkpoint theta',(9,));array(p['momentum_buffer'],'checkpoint momentum',(9,))
    if type(p['orders']) is not list or len(p['orders'])!=e or any(type(r) is not list or sorted(r)!=list(range(len(X))) or any(type(v) is not int for v in r) for r in p['orders']):raise ValueError('checkpoint orders')
    history=p['history']
    if type(history) is not list or len(history)!=k+1:raise ValueError('checkpoint history length')
    fields={'update','epoch','samples_seen','loss','lr_used','lr_next'}
    for j,r in enumerate(history):
        if type(r) is not dict or set(r)!=fields or type(r['update']) is not int or r['update']!=j or r['samples_seen']!=j*effective:raise ValueError('checkpoint history indexing')
        integer(r['samples_seen'],'samples_seen',0,total*effective);integer(r['epoch'],'history epoch',0,e)
        if r['epoch']!=(0 if j==0 else (j-1)//(len(X)//effective)+1):raise ValueError('checkpoint history epoch')
        real(r['loss'],'history loss',0,1e6);real(r['lr_next'],'history lr_next',0,.2)
        if j:real(r['lr_used'],'history lr_used',0,.2)
        elif r['lr_used'] is not None:raise ValueError('initial lr_used')
    rng=p['shuffle_rng']
    if type(rng) is not list or len(rng)!=len(torch.Generator().get_state()) or any(type(v) is not int or not 0<=v<=255 for v in rng):raise ValueError('checkpoint RNG bytes')
    try:torch.Generator().set_state(torch.tensor(rng,dtype=torch.uint8))
    except RuntimeError as ex:raise ValueError('invalid RNG state') from ex
    # Build a pristine optimizer/scheduler to validate every known state key and constant.
    param=torch.nn.Parameter(torch.tensor(cfg['initial_theta'],dtype=DTYPE));opt,sch=make_optimizer(param,cfg,mode,total)
    group=copy.deepcopy(opt.state_dict()['param_groups'][0]);group['lr']=rate_for_mode(k,cfg,mode,total)
    if canonical(p['optimizer_group'])!=canonical(group):raise ValueError('checkpoint optimizer group/LR')
    ss=sch.state_dict();ss.update({'last_epoch':k,'_step_count':k+1,'_last_lr':[group['lr']]})
    if canonical(p['scheduler'])!=canonical(ss):raise ValueError('checkpoint scheduler counter/state')
    if history[-1]['lr_next']!=group['lr']:raise ValueError('checkpoint next LR mismatch')
    return copy.deepcopy(p)


def rate_for_mode(k,cfg,mode,total):
    return cfg['peak_lr'] if mode.endswith('fixed') else schedule_lr(k,total,cfg['peak_lr'],cfg['min_lr'],cfg['warmup_updates'] if mode=='momentum_warmup_cosine' else 0)


def make_optimizer(param,cfg,mode,total):
    if mode not in MODES:raise ValueError('unknown mode')
    opt=torch.optim.SGD([param],lr=cfg['peak_lr'],momentum=0 if mode=='sgd_fixed' else cfg['momentum'])
    # Functions are not saved by LambdaLR.state_dict; config + this definition are required.
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda k:rate_for_mode(k,cfg,mode,total)/cfg['peak_lr'])
    return opt,sch


def train_network(cfg,X,y,mode='momentum_warmup_cosine',stop_epoch=None,checkpoint=None,_omit=None):
    """Train to a full-epoch boundary; resume continues only the original remaining budget.
    _omit is an internal negative-control switch; ordinary callers leave it None.
    """
    X,y,effective,total=train_config(cfg,X,y)
    if mode not in MODES:raise ValueError('unknown mode')
    stop=cfg['epochs'] if stop_epoch is None else integer(stop_epoch,'stop_epoch',1,cfg['epochs'])
    if _omit not in (None,'optimizer','scheduler','rng'):raise ValueError('unknown omission')
    if _omit is not None and checkpoint is None:raise ValueError('omission requires checkpoint')
    restored=validate_checkpoint(checkpoint,cfg,X,y,mode) if checkpoint is not None else None
    p=torch.nn.Parameter(torch.tensor(cfg['initial_theta'],dtype=DTYPE));opt,sch=make_optimizer(p,cfg,mode,total)
    g=torch.Generator().manual_seed(cfg['seed']);tx=torch.tensor(X,dtype=DTYPE);ty=torch.tensor(y,dtype=DTYPE)
    epoch0=0;orders=[];history=[]
    if restored:
        epoch0=restored['completed_epochs']
        if stop<=epoch0:raise ValueError('resume requires remaining updates in original budget')
        with torch.no_grad():p.copy_(torch.tensor(restored['theta'],dtype=DTYPE))
        if _omit!='scheduler':sch.load_state_dict(copy.deepcopy(restored['scheduler']))
        if _omit!='optimizer':
            state={0:{'momentum_buffer':torch.tensor(restored['momentum_buffer'],dtype=DTYPE)}} if mode!='sgd_fixed' and cfg['momentum'] else {}
            opt.load_state_dict({'state':state,'param_groups':[copy.deepcopy(restored['optimizer_group'])]})
        else:opt.param_groups[0]['lr']=restored['optimizer_group']['lr']
        if _omit!='rng':g.set_state(torch.tensor(restored['shuffle_rng'],dtype=torch.uint8))
        orders=copy.deepcopy(restored['orders']);history=copy.deepcopy(restored['history'])
    else:
        with torch.no_grad():loss=float((forward(p,tx)[2]-ty).square().mean()/2)
        history=[{'update':0,'epoch':0,'samples_seen':0,'loss':loss,'lr_used':None,'lr_next':opt.param_groups[0]['lr']}]
    steps_per_epoch=len(X)//effective
    for epoch in range(epoch0,stop):
        order=torch.randperm(len(X),generator=g).tolist();orders.append(order)
        for start in range(0,len(X),effective):
            ids=order[start:start+effective];opt.zero_grad(set_to_none=True)
            # Each microbatch sums losses and divides by the full window sample count.
            for micro in range(0,effective,cfg['microbatch_size']):
                idx=ids[micro:micro+cfg['microbatch_size']]
                loss=(forward(p,tx[idx])[2]-ty[idx]).square().sum()/(2*len(ids));loss.backward()
            if not torch.isfinite(p.grad).all():raise ArithmeticError('non-finite gradient')
            alpha=opt.param_groups[0]['lr'];opt.step();sch.step()
            update=epoch*steps_per_epoch+start//effective+1
            with torch.no_grad():full_loss=float((forward(p,tx)[2]-ty).square().mean()/2)
            history.append({'update':update,'epoch':epoch+1,'samples_seen':update*effective,'loss':full_loss,'lr_used':alpha,'lr_next':opt.param_groups[0]['lr']})
    if not all(math.isfinite(r['loss']) for r in history) or not torch.isfinite(p).all():raise ArithmeticError('non-finite network result')
    buffer=opt.state.get(p,{}).get('momentum_buffer',torch.zeros_like(p)).detach().tolist()
    payload={'format':'dl033-epoch-boundary-v1','torch_version':torch.__version__,'config':copy.deepcopy(cfg),'data_sha256':hashlib.sha256(canonical({'X':X.tolist(),'y':y.tolist()})).hexdigest(),'mode':mode,'completed_epochs':stop,'theta':p.detach().tolist(),'momentum_buffer':buffer,'optimizer_group':copy.deepcopy(opt.state_dict()['param_groups'][0]),'scheduler':copy.deepcopy(sch.state_dict()),'shuffle_rng':g.get_state().tolist(),'history':history,'orders':orders}
    return seal(payload)


def build_outputs(cfg,hand,X,y,resume=None):
    trace=trace_steps(hand)
    quadratics={m:quadratic_run(cfg['quadratic_steps'],cfg['quadratic_peak_lr'],0 if m=='sgd_fixed' else cfg['momentum'],'fixed' if m.endswith('fixed') else 'warmup_cosine' if m=='momentum_warmup_cosine' else 'cosine') for m in MODES}
    runs={m:train_network(cfg,X,y,m) for m in MODES}
    cut=train_network(cfg,X,y,stop_epoch=cfg['checkpoint_epoch'])
    continued=train_network(cfg,X,y,checkpoint=resume or cut)
    full=runs['momentum_warmup_cosine']
    if canonical(continued)!=canonical(full):raise ArithmeticError('full resume is not exactly identical')
    faults={name:train_network(cfg,X,y,checkpoint=cut,_omit=name) for name in ['optimizer','scheduler','rng']}
    resume_report={'boundary':'after full epoch 4: optimizer.step then scheduler.step, before next permutation; next zero_grad discards old gradients','remaining_updates':48,'exact_all_state_equal':continued==full,'negative_controls':{name:{'max_parameter_gap':float(np.max(np.abs(np.array(obj['payload']['theta'])-np.array(full['payload']['theta'])))),'same_all_orders':obj['payload']['orders']==full['payload']['orders'],'lr_used_suffix': [r['lr_used'] for r in obj['payload']['history'][25:]],'loss_final':obj['payload']['history'][-1]['loss']} for name,obj in faults.items()}}
    total=72
    summary={'unit':'033','runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':torch.get_num_threads()},'budget':{'samples':len(X),'epochs':cfg['epochs'],'microbatch':cfg['microbatch_size'],'accumulation':cfg['accumulation'],'optimizer_updates':72,'sample_presentations':576},'network_final_loss':{m:runs[m]['payload']['history'][-1]['loss'] for m in MODES},'quadratic_final_loss':{m:quadratics[m][-1]['loss'] for m in MODES},'trace_losses':[r['loss'] for r in trace['steps']]+[trace['steps'][-1]['next_forward']['loss']],'resume':resume_report}
    schedule_rows=[{'update_index':k,**{m:rate_for_mode(k,cfg,m,total) for m in MODES},'wrong_microbatch_clock':rate_for_mode(2*k,cfg,'momentum_warmup_cosine',total)} for k in range(total)]
    data={'trace.json':trace,'quadratic.json':quadratics,'network.json':{m:obj['payload'] for m,obj in runs.items()},'checkpoint-epoch-04.json':cut,'final-state.json':full,'resume.json':resume_report,'summary.json':summary}
    blobs={name:canonical(obj) for name,obj in data.items()}
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(schedule_rows[0]));w.writeheader();w.writerows(schedule_rows);blobs['schedules.csv']=s.getvalue().encode()
    return blobs


def write_outputs(blobs,directory):
    """All validation/computation/serialization precedes this function.
    Each file replacement is atomic; a multi-file OS transaction is NOT promised.
    """
    out=Path(directory).absolute()
    if any(p.is_symlink() for p in [out,*out.parents]):raise ValueError('output path must not use symlinks')
    if out.resolve() in {HERE,HERE/'data',HERE/'figures'}:raise ValueError('output must be separate from source directories')
    if out.exists() and not out.is_dir():raise ValueError('output must be a directory')
    for name,blob in blobs.items():
        if Path(name).name!=name or type(blob) is not bytes:raise ValueError('invalid output mapping')
        target=out/name
        if target.is_symlink() or (target.exists() and not target.is_file()):raise ValueError('unsafe output target')
    out.mkdir(parents=True,exist_ok=True)
    staged=[]
    try:
        for name,blob in blobs.items():
            fd,tmp=tempfile.mkstemp(prefix='.033-',dir=out)
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            staged.append((tmp,out/name))
        for tmp,target in staged:os.replace(tmp,target)
    finally:
        for tmp,_ in staged:
            if os.path.exists(tmp):os.unlink(tmp)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'outputs');parser.add_argument('--resume-checkpoint',type=Path);args=parser.parse_args()
    cfg,hand,X,y=read_inputs();configure_cpu()
    checkpoint=None
    if args.resume_checkpoint:
        if args.resume_checkpoint.stat().st_size>2_000_000:raise ValueError('checkpoint exceeds bounded teaching format')
        checkpoint=json.loads(args.resume_checkpoint.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('non-finite JSON')))
        validate_checkpoint(checkpoint,cfg,X,y,'momentum_warmup_cosine')
    blobs=build_outputs(cfg,hand,X,y,checkpoint);write_outputs(blobs,args.output)
    print(json.dumps({'status':'passed','files':sorted(blobs),'output':str(args.output)},ensure_ascii=False))

if __name__=='__main__':main()
