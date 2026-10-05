"""DL044: paired real PyTorch CNN/ResNet training, complete unsuccessful runs kept."""
from pathlib import Path
import argparse,hashlib,json,math,time,platform,copy
import numpy as np
import torch
from torch import nn
from reference import hand_ledger
HERE=Path(__file__).resolve().parent
SEEDS=(11,29,47);LRS=(0.03,1.0);DEPTHS=(2,6);EPOCHS=18;BATCH=32

def check(condition,message):
    if not condition: raise ValueError(message)

class Block(nn.Module):
    def __init__(self,channels,skip):
        super().__init__();self.skip=bool(skip)
        self.conv1=nn.Conv2d(channels,channels,3,padding=1,bias=False);self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,channels,3,padding=1,bias=False);self.bn2=nn.BatchNorm2d(channels)
    def forward(self,x):
        f=self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))
        if self.skip:
            check(f.shape==x.shape,'exact shortcut shape required; broadcasting forbidden')
            f=f+x
        return torch.relu(f)

class SmallCNN(nn.Module):
    def __init__(self,blocks,skip,width=8):
        super().__init__()
        check(type(blocks) is int and blocks in DEPTHS,'blocks must be 2 or 6')
        check(type(width) is int and width==8,'teaching experiment fixes width=8')
        self.stem=nn.Conv2d(1,width,3,padding=1,bias=False);self.stem_bn=nn.BatchNorm2d(width)
        self.blocks=nn.ModuleList([Block(width,skip) for _ in range(blocks)]);self.head=nn.Linear(width,2)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
        # BN gamma stays 1: do not secretly give residual a zero-final-BN advantage.
    def forward(self,x):
        check(x.ndim==4 and x.shape[1:]==(1,12,12),'require N,1,12,12 input')
        check(x.shape[0]>0 and x.dtype==torch.float32 and bool(torch.isfinite(x).all()),'require finite nonempty float32 input')
        z=torch.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:z=block(z)
        return self.head(z.mean((2,3)))

def load_data(split):
    check(split in ('train','validation','test'),'unknown split')
    manifest=json.loads((HERE/'data/manifest.json').read_text())
    vals=[]
    for suffix in ('x','y'):
        p=HERE/'data'/f'{split}_{suffix}.npy'
        check(hashlib.sha256(p.read_bytes()).hexdigest()==manifest[p.name]['sha256'],'dataset bytes changed: '+p.name)
        vals.append(np.load(p,allow_pickle=False))
    x,y=vals;check(x.shape==(len(y),1,12,12) and x.dtype==np.float32,'bad input shape/dtype')
    check(y.dtype==np.int64 and set(np.unique(y))=={0,1},'bad labels')
    check(np.isfinite(x).all(),'nonfinite data')
    return torch.from_numpy(x),torch.from_numpy(y)

def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():
        logits=model(x);loss=nn.functional.cross_entropy(logits,y)
    return dict(loss=float(loss),accuracy=float((logits.argmax(1)==y).float().mean())),logits

def probe(model,x,y):
    # Eval-mode fixed training probe: no BN-running-state change, no optimizer update.
    model.eval();model.zero_grad(set_to_none=True)
    loss=nn.functional.cross_entropy(model(x),y);loss.backward()
    return {n:float(p.grad.square().mean().sqrt()) for n,p in model.named_parameters() if p.grad is not None and ('.conv' in n or n in ('stem.weight','head.weight'))}

def budget(blocks,skip):
    return dict(parameters=106+1184*blocks,conv_layers=1+2*blocks,linear_layers=1,
                forward_conv_linear_macs_per_example=10368+165888*blocks+16,
                shortcut_adds_per_example=blocks*8*12*12 if skip else 0,
                updates=EPOCHS*256//BATCH,training_examples_seen=EPOCHS*256,
                note='MAC excludes BN, ReLU, pooling, additions, backward and evaluation; one MAC is one multiply-accumulate.')

def train_one(blocks,skip,lr,seed,train,val):
    tx,ty=train;vx,vy=val;torch.manual_seed(seed);model=SmallCNN(blocks,skip)
    initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=0)
    rng=np.random.default_rng(10000+seed); history=[];updates=[];probes=[];status='completed';t0=time.perf_counter()
    probes.append(dict(epoch=0,layers=probe(model,tx[:32],ty[:32])))
    peak_state=None;peak_loss=-1.;peak_epoch=None
    for epoch in range(1,EPOCHS+1):
        permutation=rng.permutation(len(tx))
        for start in range(0,len(tx),BATCH):
            idx=permutation[start:start+BATCH];model.train();opt.zero_grad(set_to_none=True)
            logits=model(tx[idx]);loss=nn.functional.cross_entropy(logits,ty[idx]);loss.backward()
            total=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
            first=float(model.stem.weight.grad.square().mean().sqrt());last=float(model.head.weight.grad.square().mean().sqrt())
            finite=math.isfinite(float(loss.detach())) and math.isfinite(total)
            updates.append(dict(update=len(updates)+1,epoch=epoch,batch_loss=float(loss.detach()) if finite else None,
                                global_grad_norm=total if finite else None,stem_grad_rms=first if finite else None,
                                head_grad_rms=last if finite else None,finite=finite))
            if not finite:status='nonfinite_loss_or_gradient';break
            opt.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):status='nonfinite_parameter';break
        if status!='completed':break
        tr,_=evaluate(model,tx,ty);va,_=evaluate(model,vx,vy)
        if not math.isfinite(tr['loss']) or not math.isfinite(va['loss']):status='nonfinite_evaluation';break
        history.append(dict(epoch=epoch,train=tr,validation=va))
        if va['loss']>peak_loss:
            peak_loss=va['loss'];peak_epoch=epoch;peak_state=copy.deepcopy(model.state_dict())
        probes.append(dict(epoch=epoch,layers=probe(model,tx[:32],ty[:32])))
    bn_diagnostic=None
    if peak_state is not None:
        diagnostic=copy.deepcopy(model);diagnostic.load_state_dict(peak_state)
        fixed_eval,_=evaluate(diagnostic,tx[:32],ty[:32]);diagnostic.train()
        with torch.no_grad(): fixed_train_loss=float(nn.functional.cross_entropy(diagnostic(tx[:32]),ty[:32]))
        diagnostic.load_state_dict(peak_state)
        weights_before=[p.detach().clone() for p in diagnostic.parameters()]
        for m in diagnostic.modules():
            if isinstance(m,nn.BatchNorm2d):m.reset_running_stats();m.momentum=None
        diagnostic.train()
        with torch.no_grad():
            for start in range(0,len(tx),BATCH):diagnostic(tx[start:start+BATCH])
        recalibrated,_=evaluate(diagnostic,vx,vy)
        check(all(torch.equal(a,b) for a,b in zip(weights_before,diagnostic.parameters())),'BN diagnostic changed learned parameters')
        bn_diagnostic=dict(epoch=peak_epoch,original_validation_loss=peak_loss,fixed_train_batch_eval_loss=fixed_eval['loss'],
                           same_train_batch_train_mode_loss=fixed_train_loss,training_only_bn_recalibrated_validation_loss=recalibrated['loss'],
                           note='Diagnostic cloned peak checkpoint only; same learned weights; reset BN and cumulative average over 8 training batches. Never used in selection.')
    result=dict(id=f"{'residual' if skip else 'plain'}_b{blocks}_lr{lr:g}_s{seed}",blocks=blocks,skip=skip,lr=lr,seed=seed,
                status=status,bn_peak_diagnostic=bn_diagnostic,initial_parameter_sha256=initial_hash,budget=budget(blocks,skip),epochs_completed=len(history),
                updates=updates,history=history,probes=probes,wall_seconds=time.perf_counter()-t0)
    return result,{k:v.clone() for k,v in model.state_dict().items()}

def run(output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    train=load_data('train');val=load_data('validation');runs=[];states={}
    for blocks in DEPTHS:
        for lr in LRS:
            for seed in SEEDS:
                for skip in (False,True):
                    result,state=train_one(blocks,skip,lr,seed,train,val);runs.append(result);states[result['id']]=state
                    print(result['id'],result['status'],result['history'][-1] if result['history'] else 'no finite epoch',flush=True)
    selections=[]
    for blocks in DEPTHS:
        for skip in (False,True):
            scored=[]
            for lr in LRS:
                rr=[r for r in runs if r['blocks']==blocks and r['skip']==skip and r['lr']==lr]
                score=float(np.mean([r['history'][-1]['validation']['loss'] for r in rr])) if all(r['status']=='completed' for r in rr) else None
                scored.append(dict(lr=lr,mean_validation_loss=score))
            eligible=[q for q in scored if q['mean_validation_loss'] is not None]
            chosen=min(eligible,key=lambda q:(q['mean_validation_loss'],q['lr']))['lr'] if eligible else None
            selections.append(dict(blocks=blocks,skip=skip,chosen_lr=chosen,candidates=scored))
    # Only now parse held-out test labels; selected final-epoch checkpoints only.
    sx,sy=load_data('test');tests=[]
    for sel in selections:
        if sel['chosen_lr'] is None:continue
        for r in runs:
            if r['blocks']==sel['blocks'] and r['skip']==sel['skip'] and r['lr']==sel['chosen_lr']:
                model=SmallCNN(r['blocks'],r['skip']);model.load_state_dict(states[r['id']]);metric,logits=evaluate(model,sx,sy)
                pred=logits.argmax(1).numpy();truth=sy.numpy();conf=np.zeros((2,2),dtype=int)
                for a,b in zip(truth,pred):conf[a,b]+=1
                tests.append(dict(id=r['id'],**metric,predictions=pred.tolist(),logits=logits.tolist(),confusion=conf.tolist(),
                                  mistakes=np.flatnonzero(pred!=truth).tolist()))
    results=dict(protocol=dict(seeds=SEEDS,lrs=LRS,blocks=DEPTHS,epochs=EPOCHS,batch_size=BATCH,selection='lowest mean final validation cross-entropy across 3 paired seeds per architecture; tie lower lr',
                 test='512 fixed independent synthetic samples; only 4 selected configurations x 3 seeds',normalization='BatchNorm train for optimizer steps; eval for all reported full-data metrics and fixed probes',
                 optimizer='SGD momentum=0.9, weight_decay=0, no scheduler, no clipping',initialization='He fan_in conv; default Linear; BN gamma=1 beta=0',probe='first 32 training examples; eval-mode gradients; no updates'),
                 environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),device='cpu'),
                 runs=runs,selections=selections,tests=tests,hand=hand_ledger())
    (output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # No hidden checkpoints needed for reproduction: deterministic full experiment is the source.
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(HERE/'outputs'));run(p.parse_args().output)
