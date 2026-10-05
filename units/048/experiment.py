"""048 protocol-controlled experiment; CPU/offline, no implicit downloads."""
from pathlib import Path
import argparse,json,hashlib,platform,sys,time
import numpy as np
import torch
from torch import nn
from generate_data import sha
from reference import hand_ledger
HERE=Path(__file__).resolve().parent
METHODS=('plain','brightness','random_stamp','mask_stamp')
def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def validate_images(x):
    if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=4 or x.shape[1:]!=(1,20,20) or not 1<=len(x)<=10000:raise ValueError('CPU float32 N,1,20,20, 1<=N<=10000 required')
    if not torch.isfinite(x).all().item() or x.min().item()<0 or x.max().item()>1:raise ValueError('pixel range must be finite [0,1]')
def transform(x,method,generator=None,training=False):
    validate_images(x)
    if method not in METHODS:raise ValueError('unknown method')
    z=x.clone()
    if method=='mask_stamp':z[:,:,:5,:5]=.22
    elif training and method=='random_stamp':
        if generator is None:raise ValueError('explicit training RNG required')
        v=torch.randint(0,2,(len(z),1,1,1),generator=generator,dtype=torch.int64).float();z[:,:,:5,:5]=.06+.76*v
    elif training and method=='brightness':
        if generator is None:raise ValueError('explicit training RNG required')
        z=(z+(torch.rand((len(z),1,1,1),generator=generator,dtype=x.dtype)-.5)*.3).clamp(0,1)
    return z
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(1,8,3,padding=1);self.conv2=nn.Conv2d(8,8,3,padding=1);self.head=nn.Linear(8*5*5,2)
    def forward(self,x):
        x=torch.nn.functional.avg_pool2d(torch.relu(self.conv1(x)),2);x=torch.nn.functional.avg_pool2d(torch.relu(self.conv2(x)),2);return self.head(x.flatten(1))
def load_split(name,data):
    x=torch.from_numpy(np.load(data/f'{name}_x.npy',allow_pickle=False));y=torch.from_numpy(np.load(data/f'{name}_y.npy',allow_pickle=False));validate_images(x)
    if y.dtype!=torch.int64 or y.shape!=(len(x),) or ((y<0)|(y>1)).any().item():raise ValueError('bad class labels')
    meta=json.loads((data/f'{name}_meta.json').read_text())
    if len(meta)!=len(y) or any(r['label']!=int(y[i]) or r['index']!=i or r['stamp'] not in (0,1) for i,r in enumerate(meta)):raise ValueError('metadata/label alignment mismatch')
    return x,y,meta
@torch.no_grad()
def evaluate(model,d,method,retain=True):
    x,y,meta=d;model.eval();logits=model(transform(x,method));loss=torch.nn.functional.cross_entropy(logits,y).item();prob=logits.softmax(1);pred=logits.argmax(1);correct=pred.eq(y)
    out={'loss':loss,'accuracy':correct.float().mean().item(),'n':len(y),'class_recall':{},'groups':{}}
    for cls in (0,1):
        mask=y==cls;out['class_recall'][str(cls)]=correct[mask].float().mean().item() if mask.any() else None
        for stamp in (0,1):
            idx=torch.tensor([r['label']==cls and r['stamp']==stamp for r in meta]);n=idx.sum().item();out['groups'][f'y{cls}_s{stamp}']={'n':n,'errors':int((~correct[idx]).sum()),'error_rate':float((~correct[idx]).float().mean()) if n else None}
    if retain:out.update(logits=logits.tolist(),predictions=pred.tolist(),probabilities=prob.tolist(),labels=y.tolist(),base_ids=[r['base_id'] for r in meta],error_indices=(~correct).nonzero().flatten().tolist())
    return out

def run(output=HERE/'outputs',data=HERE/'data'):
    output=Path(output);data=Path(data);output.mkdir(parents=True,exist_ok=True)
    if (output/'selection-freeze.json').exists():raise ValueError('output already frozen; choose a fresh output directory for reproduction')
    torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    protocol=json.loads((HERE/'protocol.json').read_text());write_json(output/'protocol-before-training.json',protocol)
    manifest=json.loads((data/'manifest.json').read_text())
    for fn,digest in manifest['files'].items():
        if sha(data/fn)!=digest:raise ValueError('dataset file checksum mismatch: '+fn)
    # Hashing a sealed file checks identity, never reveals values to selection.
    train=load_split('train',data);dev={k:load_split(k,data) for k in ('dev_iid','dev_shift')};runs=[];states={};start=time.monotonic();events=[{'event':'protocol_saved','sequence':0},{'event':'development_data_loaded','sequence':1}]
    for method in protocol['methods']:
        for lr in protocol['learning_rates']:
            for seed in protocol['seeds']:
                torch.manual_seed(seed);model=SmallCNN();optim=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
                order=torch.Generator().manual_seed(1000+seed);aug=torch.Generator().manual_seed(2000+seed);name=f'{method}_lr{lr:g}_seed{seed}';history=[]
                for epoch in range(protocol['epochs']+1):
                    if epoch:
                        model.train();perm=torch.randperm(len(train[0]),generator=order)
                        for idx in perm.split(protocol['batch_size']):
                            optim.zero_grad(set_to_none=True);logits=model(transform(train[0][idx],method,aug,True));loss=nn.functional.cross_entropy(logits,train[1][idx]);loss.backward()
                            if not torch.isfinite(loss).item() or any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):raise FloatingPointError(name+' nonfinite training')
                            optim.step()
                    history.append({'epoch':epoch,'train':evaluate(model,train,method,False),**{k:evaluate(model,v,method,False) for k,v in dev.items()}})
                state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()};checkpoint=output/(name+'.npz');np.savez(checkpoint,**state);states[name]=state
                runs.append({'id':name,'method':method,'lr':lr,'seed':seed,'updates':protocol['epochs']*6,'parameters':sum(p.numel() for p in model.parameters()),'macs_per_example':20*20*8*9+10*10*8*8*9+200*2,'history':history,'dev':{k:evaluate(model,v,method) for k,v in dev.items()},'checkpoint_sha256':sha(checkpoint)})
                print('trained',name,flush=True)
    scores=[];selected={}
    for method in protocol['methods']:
        candidates=[]
        for lr in protocol['learning_rates']:
            rr=[r for r in runs if r['method']==method and r['lr']==lr];score=float(np.mean([(r['dev']['dev_iid']['loss']+r['dev']['dev_shift']['loss'])/2 for r in rr]));candidates.append((score,lr));scores.append({'method':method,'lr':lr,'mean_dev_ce':score})
        score,lr=min(candidates);selected[method]={'lr':lr,'score':score,'ids':[r['id'] for r in runs if r['method']==method and r['lr']==lr]}
    winner=min(selected,key=lambda m:(selected[m]['score'],m));freeze={'protocol_sha256':sha(HERE/'protocol.json'),'data_manifest_sha256':sha(data/'manifest.json'),'selected':selected,'winner':winner,'scores':scores,'all_checkpoint_sha256':{r['id']:r['checkpoint_sha256'] for r in runs},'before_test_arrays_loaded':True}
    write_json(output/'selection-freeze.json',freeze);events.append({'event':'selection_frozen','sequence':2,'sha256':sha(output/'selection-freeze.json')})
    tests={k:load_split(k,data) for k in ('test_iid','test_shift')};events.append({'event':'test_arrays_first_loaded','sequence':3});final=[]
    for r in runs:
        if r['id'] not in selected[r['method']]['ids']:continue
        model=SmallCNN();model.load_state_dict({k:torch.from_numpy(v) for k,v in states[r['id']].items()});item={'id':r['id'],'method':r['method'],'seed':r['seed'],'lr':r['lr']}
        for name,d in tests.items():
            item[name]=evaluate(model,d,r['method']);x,y,meta=d;cf=x.clone();cf[:,:,:5,:5]=.88-cf[:,:,:5,:5];cf=cf.clamp(0,1)
            cfr=evaluate(model,(cf,y,meta),r['method']);item[name]['counterfactual']={'flip_rate':float(np.mean(np.array(cfr['predictions'])!=np.array(item[name]['predictions']))),'accuracy':cfr['accuracy'],'logits':cfr['logits'],'predictions':cfr['predictions']}
        final.append(item)
    baseline={}
    for name,(x,y,meta) in tests.items():baseline[name]={'majority_accuracy':float((y==0).float().mean()),'stamp_accuracy':float(np.mean([r['stamp']==r['label'] for r in meta]))}
    results={'protocol':protocol,'hand':hand_ledger(),'runs':runs,'freeze':freeze,'finalists':final,'no_fit_baselines':baseline,'events':events,'wall_seconds':round(time.monotonic()-start,3)}
    write_json(output/'results.json',results);write_json(output/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'device':'cpu','threads':1,'deterministic_algorithms':True,'pretrained':False,'external_dataset':False});print('completed',len(runs),'candidates;',winner,flush=True);return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'outputs');p.add_argument('--data',type=Path,default=HERE/'data');a=p.parse_args();run(a.output,a.data)
