"""DL046 fixed-budget binary semantic segmentation. CPU offline, full batch."""
from pathlib import Path
import argparse,json,time,platform,hashlib
import numpy as np
import torch
from reference import mask_metrics,hand_ledger
ROOT=Path(__file__).resolve().parent
SEEDS=[11,23,37];ARMS=['pixel','cnn_bce','cnn_weighted'];LRS=[.1,1.];STEPS=100

def load_data():
    return {s:(np.load(ROOT/'data'/f'{s}_x.npy'),np.load(ROOT/'data'/f'{s}_y.npy')) for s in ['train','validation','test','shift']}

def init(arm,seed):
    rng=np.random.default_rng(seed)
    if arm=='pixel':raw=[np.array([[[[.1]]]]),np.zeros(1)]
    else:raw=[rng.normal(0,.25,(6,1,3,3)),np.ones(6)*.1,rng.normal(0,.25,(1,6,1,1)),np.zeros(1)]
    return [torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in raw]

def forward(p,x):
    if len(p)==2:return torch.nn.functional.conv2d(x,*p)
    h=torch.relu(torch.nn.functional.conv2d(x,p[0],p[1],padding=1));return torch.nn.functional.conv2d(h,p[2],p[3])

def flat(p):return np.concatenate([a.detach().numpy().ravel() for a in p])

def evaluate(p,x,y):
    with torch.no_grad():z=forward(p,x);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
    pred=(z.numpy()>=0)[:,0];m=mask_metrics(pred,y.numpy()[:,0]);m['unweighted_bce']=float(loss);return m,z.numpy()

def box_mean(x):
    xp=np.pad(x,((0,0),(0,0),(1,1),(1,1)),mode='edge');return np.lib.stride_tricks.sliding_window_view(xp,(3,3),axis=(2,3)).mean((-1,-2))

def run(output):
    start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data=load_data();d={s:(torch.tensor(x),torch.tensor(y,dtype=torch.float64)) for s,(x,y) in data.items()};tr,ty=d['train'];vx,vy=d['validation'];q=float((1-ty.mean())/ty.mean())
    results={'protocol':{'seeds':SEEDS,'arms':ARMS,'lrs':LRS,'stress_lr':10.,'steps':STEPS,'batch_size':64,'dtype':'float64','threads':1,'training':'full-batch SGD; no momentum, decay, augmentation, scheduler or early stopping','selection':'one LR per arm maximizing mean final validation micro foreground IoU over three seeds; tie smaller LR; fixed threshold logit>=0','test_use':'only after arm-level LR frozen; no test tuning','weighted_positive_weight_train_only':q,'positive_weight_dtype':'torch.float64 (same as logits)','baseline_grid':np.linspace(.2,.8,25).tolist()},'candidates':[],'selected':[],'baselines':{},'hand':hand_ledger()};arrays={}
    for arm in ARMS:
        for lr in LRS+([10.] if arm=='cnn_bce' else []):
            for seed in SEEDS:
                key=f'{arm}_lr{lr:g}_s{seed}';p=init(arm,seed);hist=[];traj=[flat(p)];status='completed'
                for step in range(STEPS):
                    z=forward(p,tr);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,ty,pos_weight=torch.tensor(q if arm=='cnn_weighted' else 1.,dtype=z.dtype,device=z.device));loss.backward();grad=float(torch.sqrt(sum((v.grad*v.grad).sum() for v in p)));record={'step':step,'train_objective_before_update':float(loss.detach()),'gradient_l2':grad}
                    if not np.isfinite(grad) or not np.isfinite(float(loss.detach())):status='nonfinite';hist.append(record);break
                    with torch.no_grad():
                        for v in p:v-=lr*v.grad;v.grad=None
                    m,zv=evaluate(p,vx,vy);record.update({'validation_bce_after_update':m['unweighted_bce'],'validation_iou_after_update':m['foreground_iou_micro']});hist.append(record);traj.append(flat(p))
                final,zv=evaluate(p,vx,vy);arrays[key+'_parameters']=np.array(traj);arrays[key+'_validation_logits']=zv
                results['candidates'].append({'id':key,'arm':arm,'lr':lr,'seed':seed,'parameters':sum(v.numel() for v in p),'status':status,'history':hist,'final_validation':final,'stress_only':lr==10.})
    for arm in ARMS:
        scores={lr:float(np.mean([r['final_validation']['foreground_iou_micro'] for r in results['candidates'] if r['arm']==arm and r['lr']==lr])) for lr in LRS};best=max(LRS,key=lambda a:(scores[a],-a))
        for seed in SEEDS:
            key=f'{arm}_lr{best:g}_s{seed}';p=init(arm,seed);vals=arrays[key+'_parameters'][-1];pos=0
            with torch.no_grad():
                for v in p:v.copy_(torch.tensor(vals[pos:pos+v.numel()].reshape(v.shape)));pos+=v.numel()
            item={'id':key,'arm':arm,'seed':seed,'selected_lr':best,'validation_lr_means':scores,'metrics':{}}
            for s,(x,y) in d.items():m,z=evaluate(p,x,y);item['metrics'][s]=m;arrays[key+'_'+s+'_logits']=z
            results['selected'].append(item)
    # Baseline thresholds are chosen on validation only. All 50 scores are retained.
    for kind in ['intensity','mean3x3']:
        feature={s:(x if kind=='intensity' else box_mean(x)) for s,(x,y) in data.items()};grid=[]
        for t in np.linspace(.2,.8,25):m=mask_metrics((feature['validation']>=t)[:,0],data['validation'][1][:,0]);grid.append({'threshold':float(t),'validation_iou':m['foreground_iou_micro']})
        best=max(grid,key=lambda r:(r['validation_iou'],-r['threshold']))['threshold'];r={'threshold':best,'search':grid,'metrics':{}}
        for s,(x,y) in data.items():pred=(feature[s]>=best);r['metrics'][s]=mask_metrics(pred[:,0],y[:,0]);arrays[f'baseline_{kind}_{s}_prediction']=pred
        results['baselines'][kind]=r
    results['baselines']['all_background']={'metrics':{s:mask_metrics(np.zeros_like(y[:,0]),y[:,0]) for s,(x,y) in data.items()}}
    # Post-hoc reporting by known generation groups does not alter any selection.
    meta=json.loads((ROOT/'data/metadata.json').read_text());results['groups']={}
    for r in results['selected']:
        s='test';pred=arrays[r['id']+'_'+s+'_logits'][:,0]>=0;y=data[s][1][:,0];results['groups'][r['id']]={}
        for kind in ['empty','single','overlap','separate']:
            ix=[i for i,x in enumerate(meta['splits'][s]['records']) if x['kind']==kind];results['groups'][r['id']][kind]=mask_metrics(pred[ix],y[ix])
    np.savez_compressed(out/'arrays.npz',**arrays)
    results['wall_seconds']=time.monotonic()-start;(out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    env={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*.npy'))}}
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print(json.dumps({'output':out.name,'seconds':results['wall_seconds'],'candidates':len(results['candidates']),'selected':len(results['selected']),'baselines':{k:v['metrics']['test']['foreground_iou_micro'] for k,v in results['baselines'].items()}},ensure_ascii=False));return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'outputs'));a=p.parse_args();run(a.output)
