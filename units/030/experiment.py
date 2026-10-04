"""DL030: a bounded, self-contained CPU training and epoch-resume experiment.

Only the bundled fixed synthetic inputs drive the command-line report. The
functions below are readable teaching code, not a general checkpoint service.
"""
from pathlib import Path
import argparse, copy, csv, hashlib, io, json, math, platform
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
INPUT_HASHES = {'config.json': '415dfe8ed8750b983b454ab38629ee36fb282f3e9449c955065445abc9eca238', 'samples.csv': 'e26f750b0e4c25f0dcdd53100a395a86b21f96d056b91365010da4cf68eb6c4c', 'split.json': '43298159b4808eb3b80fe841d347f26543cbb95212f657fa7a5ea86a1fd2d98f'}  # Filled from original teaching fixtures, not at runtime.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_files(data_dir):
    data_dir = Path(data_dir)
    for name, digest in INPUT_HASHES.items():
        require(hashlib.sha256((data_dir/name).read_bytes()).hexdigest() == digest,
                'Fixed teaching input differs: '+name)


def load_inputs(data_dir=ROOT/'data'):
    check_files(data_dir)
    p = Path(data_dir)
    cfg = json.loads((p/'config.json').read_text())
    split = json.loads((p/'split.json').read_text())
    with (p/'samples.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    x = torch.tensor([[float(r['x1']),float(r['x2'])] for r in rows],dtype=torch.float64)
    y = torch.tensor([[float(r['y'])] for r in rows],dtype=torch.float64)
    validate_data(x,y,split)
    return x,y,split,cfg


def validate_data(x,y,split):
    require(isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor), 'Expected tensors')
    require(x.device.type == y.device.type == 'cpu', 'CPU only')
    require(x.dtype == y.dtype == torch.float64, 'float64 only')
    require(x.ndim == 2 and x.shape[1] == 2 and y.shape == (len(x),1), 'Expected X(N,2), y(N,1)')
    require(3 <= len(x) <= 1000 and torch.isfinite(x).all().item() and torch.isfinite(y).all().item(), 'Finite bounded data')
    require(x.abs().max().item() <= 100 and y.abs().max().item() <= 100, 'Teaching range is [-100,100]')
    require(isinstance(split,dict) and set(split) == {'train','validation','test'}, 'Three explicit splits required')
    ids=[]
    for name in ('train','validation','test'):
        q=split[name]
        require(isinstance(q,list) and len(q)>0 and all(type(v) is int for v in q), 'Nonempty integer ID lists required')
        ids.extend(q)
    require(sorted(ids)==list(range(len(x))), 'Splits must partition all row IDs exactly once')


class Rows(Dataset):
    """Map-style dataset: local index -> copied feature, target and original ID."""
    def __init__(self,x,y,ids):
        self.x=x[ids].detach().clone();self.y=y[ids].detach().clone()
        self.ids=torch.tensor(ids,dtype=torch.int64)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,index):
        return self.x[index].clone(),self.y[index].clone(),self.ids[index].clone()


class SmallRegressor(nn.Module):
    def __init__(self,mean,scale,hidden=4,dropout=0.25):
        super().__init__()
        self.register_buffer('mean',mean.detach().clone())
        self.register_buffer('scale',scale.detach().clone())
        self.hidden=nn.Linear(2,hidden,dtype=torch.float64)
        self.activation=nn.Tanh()
        self.dropout=nn.Dropout(dropout)
        self.output=nn.Linear(hidden,1,dtype=torch.float64)
    def forward(self,x):
        z=(x-self.mean)/self.scale
        return self.output(self.dropout(self.activation(self.hidden(z))))


def clone_state(model):
    # state_dict() is shallow; detach().clone() makes a true numeric snapshot.
    return {k:v.detach().clone() for k,v in model.state_dict().items()}


def tensor_dict_equal(a,b):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)


def nested_equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def maximum_parameter_gap(a,b):
    return max((a[k]-b[k]).abs().max().item() for k in a)


def make_run(x,y,split,cfg):
    validate_data(x,y,split)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg['seed'])
    train=x[split['train']]
    mean=train.mean(0); scale=((train-mean).square().mean(0)).sqrt()
    require((scale>0).all().item(), 'Training feature has zero scale')
    model=SmallRegressor(mean,scale,cfg['hidden'],cfg['dropout'])
    opt=torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],foreach=False)
    generator=torch.Generator().manual_seed(cfg['loader_seed'])
    loaders={}
    for name in ('train','train_eval','validation','test'):
        g=generator if name=='train' else torch.Generator().manual_seed(900)
        loaders[name]=DataLoader(Rows(x,y,split['train' if name=='train_eval' else name]),batch_size=cfg['batch_size'],
            shuffle=name=='train',drop_last=False,num_workers=0,generator=g)
    return {'model':model,'optimizer':opt,'generator':generator,'loaders':loaders,
            'epoch':0,'history':[],'orders':[],'best':None,'best_loss':None,'best_epoch':None,
            'config':copy.deepcopy(cfg),'split':copy.deepcopy(split)}


def batch_mse(pred,target):
    require(pred.shape==target.shape and pred.ndim==2 and pred.shape[1]==1, 'Prediction/target shapes must both be (B,1)')
    loss=(pred-target).square().mean()
    require(torch.isfinite(loss).item(), 'Nonfinite loss')
    return loss


def evaluate(model,loader):
    # Observational contract: no parameter/buffer change, no stale-gradient change.
    before=clone_state(model)
    grads=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    modes=[(m,m.training) for m in model.modules()]
    numerator=0.0; count=0; batch_means=[];predictions=[]
    model.eval()
    try:
        with torch.no_grad():
            for x,y,ids in loader:
                pred=model(x);loss=batch_mse(pred,y)
                numerator+=loss.item()*len(y);count+=len(y);batch_means.append(loss.item())
                predictions.extend({'id':int(i),'prediction':float(p),'target':float(t)}
                                   for i,p,t in zip(ids,pred[:,0],y[:,0]))
    finally:
        # Preserve even mixed submodule modes, not merely the root flag.
        for module,was_training in modes: module.training=was_training
    require(count>0, 'Cannot evaluate empty loader')
    require(tensor_dict_equal(before,clone_state(model)), 'Evaluation mutated model state')
    require(all((old is None and p.grad is None) or (old is not None and p.grad is not None and torch.equal(old,p.grad))
                for old,p in zip(grads,model.parameters())), 'Evaluation mutated gradients')
    return {'mse':numerator/count,'count':count,'wrong_equal_batch_mean':sum(batch_means)/len(batch_means),'predictions':predictions}


def train_one_epoch(run):
    model=run['model'];opt=run['optimizer'];model.train()
    loss_sum=0.0;count=0;order=[];batch_sizes=[]
    for x,y,ids in run['loaders']['train']:
        opt.zero_grad(set_to_none=True)
        loss=batch_mse(model(x),y)
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Missing/nonfinite gradient')
        opt.step()
        loss_sum+=loss.item()*len(y);count+=len(y);order.extend(ids.tolist());batch_sizes.append(len(y))
    require(sorted(order)==sorted(run['split']['train']), 'Training must cover every train ID once')
    val=evaluate(model,run['loaders']['validation'])
    train_eval=evaluate(model,run['loaders']['train_eval'])
    # Evaluation loaders have separate generators and never advance train RNG.
    run['epoch']+=1
    row={'epoch':run['epoch'],'train_online_mse':loss_sum/count,
         'train_eval_mse':train_eval['mse'],'validation_mse':val['mse'],
         'validation_wrong_equal_batch_mean':val['wrong_equal_batch_mean'],
         'examples':count,'optimizer_steps':len(batch_sizes)}
    run['history'].append(row);run['orders'].append(order)
    if run['best_loss'] is None or val['mse']<run['best_loss']:
        run['best_loss']=val['mse'];run['best_epoch']=run['epoch'];run['best']=clone_state(model)
    return row


def train_until(run,epoch):
    require(type(epoch) is int and run['epoch']<=epoch<=run['config']['epochs'], 'Invalid epoch endpoint')
    while run['epoch']<epoch:train_one_epoch(run)
    return run


def checkpoint(run):
    require(run['epoch']>0, 'Save only after a complete epoch and validation')
    return {'format':1,'epoch':run['epoch'],'model':clone_state(run['model']),
        'optimizer':copy.deepcopy(run['optimizer'].state_dict()),
        'torch_rng':torch.get_rng_state().clone(),'loader_rng':run['generator'].get_state().clone(),
        'config':copy.deepcopy(run['config']),'split':copy.deepcopy(run['split']),
        'input_sha256':dict(INPUT_HASHES),'history':copy.deepcopy(run['history']),
        'orders':copy.deepcopy(run['orders']),'best':copy.deepcopy(run['best']),
        'best_loss':run['best_loss'],'best_epoch':run['best_epoch'],
        'runtime':{'torch':str(torch.__version__),'python':platform.python_version(),'dtype':'float64','device':'cpu'}}


def checkpoint_bytes(state):
    buffer=io.BytesIO();torch.save(state,buffer);return buffer.getvalue()


def load_checkpoint_bytes(content):
    # This is our own generated teaching artifact, not arbitrary internet input.
    return torch.load(io.BytesIO(content),map_location='cpu',weights_only=True)


def restore(state,x,y,split,cfg,omit=None):
    require(omit in (None,'optimizer','torch_rng','loader_rng'), 'Unknown ablation')
    require(type(state) is dict and state.get('format')==1, 'Unsupported checkpoint')
    require(nested_equal(state['config'],cfg) and nested_equal(state['split'],split), 'Configuration/split mismatch')
    require(state['input_sha256']==INPUT_HASHES, 'Data identity mismatch')
    require(state['runtime']=={'torch':str(torch.__version__),'python':platform.python_version(),'dtype':'float64','device':'cpu'}, 'Runtime mismatch')
    epoch=state['epoch'];require(type(epoch) is int and 1<=epoch<=cfg['epochs'], 'Invalid completed epoch')
    require(len(state['history'])==len(state['orders'])==epoch, 'Incomplete epoch history')
    require([r['epoch'] for r in state['history']]==list(range(1,epoch+1)), 'History epoch mismatch')
    require(all(sorted(o)==sorted(split['train']) for o in state['orders']), 'Incomplete sample order')
    require(type(state['best_epoch']) is int and 1<=state['best_epoch']<=epoch, 'Invalid best epoch')
    require(type(state['best_loss']) is float and math.isfinite(state['best_loss']), 'Invalid best loss')
    require(state['best_loss']==min(r['validation_mse'] for r in state['history']), 'Best loss/history mismatch')
    expected_best=min(range(epoch),key=lambda i:state['history'][i]['validation_mse'])+1
    require(state['best_epoch']==expected_best, 'Best epoch/history mismatch')
    run=make_run(x,y,split,cfg)
    reference=run['model'].state_dict()
    for key in ('model','best'):
        require(set(state[key])==set(reference), 'State keys mismatch')
        for name,t in state[key].items():
            require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==reference[name].dtype
                    and t.shape==reference[name].shape and torch.isfinite(t).all().item(), 'State tensor invalid')
        require(torch.equal(state[key]['mean'],reference['mean']) and torch.equal(state[key]['scale'],reference['scale']), 'Training normalization mismatch')
    for key in ('torch_rng','loader_rng'):
        rng=state[key]
        require(isinstance(rng,torch.Tensor) and rng.dtype==torch.uint8 and rng.ndim==1, 'Invalid RNG tensor')
        torch.Generator().set_state(rng)
    require(state['optimizer']['param_groups']==run['optimizer'].state_dict()['param_groups'], 'Optimizer configuration mismatch')
    os=state['optimizer']['state'];ps=list(run['model'].parameters())
    require(set(os)==set(range(len(ps))), 'Optimizer state IDs mismatch')
    for i,p in enumerate(ps):
        require(set(os[i])=={'momentum_buffer'}, 'Momentum state missing')
        t=os[i]['momentum_buffer']
        require(isinstance(t,torch.Tensor) and t.dtype==p.dtype and t.shape==p.shape and torch.isfinite(t).all().item(), 'Momentum tensor invalid')
    run['model'].load_state_dict(state['model'],strict=True)
    if omit!='optimizer':run['optimizer'].load_state_dict(copy.deepcopy(state['optimizer']))
    for key in ('epoch','history','orders','best','best_loss','best_epoch'):run[key]=copy.deepcopy(state[key])
    if omit!='loader_rng':run['generator'].set_state(state['loader_rng'])
    # Restore last, after all constructors that can consume the global RNG.
    if omit!='torch_rng':torch.set_rng_state(state['torch_rng'])
    return run


def state_to_json(state):
    if isinstance(state,torch.Tensor):return state.tolist()
    if isinstance(state,dict):return {str(k):state_to_json(v) for k,v in state.items()}
    if isinstance(state,(tuple,list)):return [state_to_json(v) for v in state]
    return state


def semantic_probes():
    class Unregistered(nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.tensor(1.,requires_grad=True);self.layers=[nn.Linear(1,1)]
        def forward(self,x):return self.layers[0](x)*self.w
    class Registered(nn.Module):
        def __init__(self):
            super().__init__();self.w=nn.Parameter(torch.tensor(1.));self.layers=nn.ModuleList([nn.Linear(1,1)])
    bad=Unregistered();bad(torch.ones(1,1)).sum().backward();good=Registered()
    m=nn.Linear(1,1,dtype=torch.float64);alias=m.state_dict();snapshot=clone_state(m)
    with torch.no_grad():m.weight.add_(1)
    p=nn.Parameter(torch.tensor(1.,dtype=torch.float64));opt=torch.optim.SGD([p],lr=.1,momentum=.5,foreach=False)
    seq=[]
    for g in (2.,3.):
        p.grad=torch.tensor(g,dtype=torch.float64);opt.step()
        seq.append({'parameter':p.item(),'buffer':opt.state[p]['momentum_buffer'].item()})
    wrong=nn.Linear(1,1,dtype=torch.float64);wrong.eval();o=torch.optim.SGD(wrong.parameters(),lr=.1)
    before=clone_state(wrong);o.zero_grad();wrong(torch.ones(1,1,dtype=torch.float64)).square().mean().backward();o.step()
    with torch.random.fork_rng():
        torch.manual_seed(20);d=nn.Dropout(.5);d.train()
        with torch.no_grad():a=d(torch.ones(64));b=d(torch.ones(64))
    return {'unregistered_parameter_names':[k for k,_ in bad.named_parameters()],
        'unregistered_w_has_gradient':bad.w.grad is not None,'registered_parameter_names':[k for k,_ in good.named_parameters()],
        'shallow_state_changed':not torch.equal(alias['weight'],snapshot['weight']),
        'clone_state_unchanged':torch.equal(snapshot['weight']+1,m.weight),
        'momentum_hand':seq,'eval_does_not_prevent_update':not tensor_dict_equal(before,clone_state(wrong)),
        'no_grad_train_dropout_changes_output':not torch.equal(a,b),
        'uneven_hand':{'batch_sizes':[2,3],'batch_means':[1.,4.],'correct':2.8,'wrong_equal':2.5}}


def first_step_trace(x,y,split,cfg):
    """All numeric coordinates of the actual first training batch and update."""
    run=make_run(x,y,split,cfg);m=run['model'];m.train();opt=run['optimizer']
    xb,yb,ids=next(iter(run['loaders']['train']));xb.requires_grad_(True)
    before=clone_state(m);opt.zero_grad(set_to_none=True)
    z=(xb-m.mean)/m.scale;a=m.hidden(z);h=m.activation(a);d=m.dropout(h);pred=m.output(d)
    for value in (z,a,h,d,pred):value.retain_grad()
    loss=batch_mse(pred,yb);loss.backward()
    require((h.detach()!=0).all().item(),'Trace requires nonzero hidden values to infer observed mask')
    mask=torch.where(d.detach()==0,torch.zeros_like(h),torch.full_like(h,1/(1-cfg['dropout'])))
    grads={name:p.grad.detach().clone() for name,p in m.named_parameters()}
    intermediates={'x':xb.detach(),'target':yb,'normalized':z.detach(),'affine':a.detach(),
        'hidden':h.detach(),'mask_scale':mask,'dropped_hidden':d.detach(),'prediction':pred.detach(),
        'residual':pred.detach()-yb,'squared_errors':(pred.detach()-yb).square(),
        'd_prediction':pred.grad,'d_dropped_hidden':d.grad,'d_hidden':h.grad,'d_affine':a.grad,
        'd_normalized':z.grad,'d_input':xb.grad}
    with torch.no_grad():pre_eval=m.output(h.detach())
    opt.step();after=clone_state(m)
    m.eval()
    with torch.no_grad():
        a2=m.hidden(z.detach());h2=m.activation(a2);p2=m.output(h2);eval_loss=batch_mse(p2,yb)
        same_mask_pred=m.output(h2*mask)
    return state_to_json({'ids':ids,'shapes':{k:list(v.shape) for k,v in intermediates.items()},
        'values':intermediates,'loss':loss.item(),'gradients':grads,'before':before,'after':after,
        'momentum_after':{name:opt.state[p]['momentum_buffer'].clone() for name,p in m.named_parameters()},
        'next_forward_same_batch_eval':{'affine':a2,'hidden':h2,'prediction':p2,'loss':eval_loss.item(),
            'pre_update_eval_prediction':pre_eval,'pre_update_eval_loss':batch_mse(pre_eval,yb).item()},
        'next_forward_fixed_original_mask':{'prediction':same_mask_pred,'loss':batch_mse(same_mask_pred,yb).item(),
            'note':'Diagnostic reuses the observed first mask, not the next stochastic training forward.'}})


def run_report(data_dir=ROOT/'data'):
    x,y,split,cfg=load_inputs(data_dir)
    trace=first_step_trace(x,y,split,cfg)
    full=make_run(x,y,split,cfg);train_until(full,cfg['checkpoint_epoch'])
    saved=checkpoint(full);blob=checkpoint_bytes(saved)
    train_until(full,cfg['epochs']);full_end=checkpoint(full)
    resumed=restore(load_checkpoint_bytes(blob),x,y,split,cfg)
    train_until(resumed,cfg['epochs']);resumed_end=checkpoint(resumed)
    require(nested_equal(full_end,resumed_end), 'Complete resume did not match')
    ablations={}
    for omission in ('optimizer','torch_rng','loader_rng'):
        trial=restore(load_checkpoint_bytes(blob),x,y,split,cfg,omit=omission)
        train_until(trial,cfg['epochs'])
        ablations[omission]={'final_parameter_max_gap':maximum_parameter_gap(full_end['model'],clone_state(trial['model'])),
            'same_orders':trial['orders']==full['orders'],'validation_mse':trial['history'][-1]['validation_mse']}
    # Select from validation only; the test loader is first evaluated here.
    full['model'].load_state_dict(full['best'],strict=True)
    test=evaluate(full['model'],full['loaders']['test'])
    summary={'runtime':full_end['runtime'],'parameters':sum(p.numel() for p in full['model'].parameters()),
        'split_sizes':{k:len(v) for k,v in split.items()},'epochs':cfg['epochs'],
        'optimizer_steps':sum(r['optimizer_steps'] for r in full['history']),
        'training_examples_seen':sum(r['examples'] for r in full['history']),
        'checkpoint_epoch':cfg['checkpoint_epoch'],'full_resume_equal':True,
        'best_epoch':full['best_epoch'],'best_validation_mse':full['best_loss'],'test_mse':test['mse'],
        'final_validation_mse':full['history'][-1]['validation_mse'],'ablations':ablations,
        'normalization_mean':full_end['model']['mean'].tolist(),'normalization_scale':full_end['model']['scale'].tolist(),
        'scope':'Synthetic CPU epoch-boundary demonstration; no cross-platform bitwise or real-task performance claim.'}
    def js(o):return (json.dumps(state_to_json(o),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    buffer=io.StringIO();w=csv.DictWriter(buffer,fieldnames=list(full['history'][0]));w.writeheader();w.writerows(full['history'])
    files={'summary.json':js(summary),'history.csv':buffer.getvalue().encode(),
        'orders.json':js(full['orders']),'semantics.json':js(semantic_probes()),
        'test-predictions.json':js(test['predictions']),'final-state.json':js(full_end),
        'checkpoint-epoch-04.pt':blob,'first-step-trace.json':js(trace)}
    return files


def write_report(output,data_dir=ROOT/'data'):
    # All validation/computation/serialization precedes creating any output.
    files=run_report(data_dir);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,content in files.items():
        tmp=output/(name+'.tmp');tmp.write_bytes(content);tmp.replace(output/name)
    return json.loads(files['summary.json'])


def write_resumed(output,checkpoint_path,data_dir=ROOT/'data'):
    x,y,split,cfg=load_inputs(data_dir)
    state=load_checkpoint_bytes(Path(checkpoint_path).read_bytes())
    run=restore(state,x,y,split,cfg);train_until(run,cfg['epochs'])
    content=(json.dumps(state_to_json(checkpoint(run)),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    temp=output/'resume-final-state.json.tmp';temp.write_bytes(content);temp.replace(output/'resume-final-state.json')
    return {'resumed_from':state['epoch'],'completed_epoch':run['epoch'],'best_epoch':run['best_epoch']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--resume-checkpoint',type=Path)
    a=p.parse_args()
    result=write_resumed(a.output,a.resume_checkpoint,a.data_dir) if a.resume_checkpoint else write_report(a.output,a.data_dir)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
