"""DL069: a verifiable asset graph and exact local checkpoint continuation.

Uses only original synthetic data. Full-state checkpoints contain plain tensors
and primitive containers, loadable with weights_only=True. No external registry.
"""
from pathlib import Path
import argparse,hashlib,json,platform,copy
import numpy as np
import torch
from torch import nn
CONFIG={'data_seed':6901,'seed':69,'n_train':256,'n_test':128,'epochs':24,'split_epoch':12,'batch_size':32,'lr':.01}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def data():
    rng=np.random.default_rng(CONFIG['data_seed']);x=rng.normal(size=(384,6)).astype('float32')
    y=(x[:,0]*x[:,1]+.7*x[:,2]-.3*x[:,3]>0).astype('int64');return torch.from_numpy(x),torch.from_numpy(y)
def model():return nn.Sequential(nn.Linear(6,24),nn.ReLU(),nn.Dropout(.3),nn.Linear(24,2))
def train(m,opt,g,x,y,start,end):
    hist=[]
    for epoch in range(start,end):
        m.train();order=torch.randperm(256,generator=g);total=0.
        for idx in order.split(32):
            opt.zero_grad();loss=nn.functional.cross_entropy(m(x[idx]),y[idx]);loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        hist.append({'epoch':epoch+1,'train_loss':total/256})
    return hist

def snapshot(m,opt,g,epoch):
    # Deep-copy is necessary: state_dict tensors otherwise share model storage.
    return copy.deepcopy({'model':m.state_dict(),'optimizer':opt.state_dict(),
        'torch_rng':torch.get_rng_state(),'order_rng':g.get_state(),'epoch':epoch,'config':CONFIG})
def restore(path,full=True):
    ck=torch.load(path,weights_only=True);m=model();m.load_state_dict(ck['model']);opt=torch.optim.Adam(m.parameters(),lr=CONFIG['lr'])
    g=torch.Generator().manual_seed(CONFIG['seed']+1)
    if full:
        opt.load_state_dict(ck['optimizer']);g.set_state(ck['order_rng']);torch.set_rng_state(ck['torch_rng'])
    return m,opt,g,ck['epoch']
def assess(m,x,y):
    m.eval()
    with torch.inference_mode():z=m(x[256:]);return {'loss':float(nn.functional.cross_entropy(z,y[256:])), 'accuracy':float((z.argmax(1)==y[256:]).float().mean())}
def verify_assets(root,manifest):
    """Paths must stay within root; a manifest is integrity evidence, not trust."""
    root=Path(root).resolve();failures=[]
    for name,expected in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root):failures.append(name+':outside_root');continue
        if not p.is_file() or digest(p)!=expected:failures.append(name)
    return failures

def run(output='outputs'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(CONFIG['seed'])
    x,y=data();np.savez(out/'data.npz',x=x.numpy(),y=y.numpy(),train_indices=np.arange(256),test_indices=np.arange(256,384))
    (out/'config.json').write_text(json.dumps(CONFIG,sort_keys=True,indent=2))
    # Hash bytes as stored, not a casually reformatted JSON representation.
    m=model();opt=torch.optim.Adam(m.parameters(),lr=CONFIG['lr']);g=torch.Generator().manual_seed(CONFIG['seed']+1)
    first=train(m,opt,g,x,y,0,12);torch.save(snapshot(m,opt,g,12),out/'resume_epoch12.pt')
    second=train(m,opt,g,x,y,12,24);continuous=copy.deepcopy(m.state_dict());torch.save(snapshot(m,opt,g,24),out/'continuous_epoch24.pt')
    full,fo,fg,start=restore(out/'resume_epoch12.pt',True);resumed=train(full,fo,fg,x,y,start,24)
    torch.save(snapshot(full,fo,fg,24),out/'resumed_epoch24.pt')
    # Deliberately omit RNG and optimizer history. This is a restart, not continuation.
    torch.manual_seed(69);bad,bo,bg,start=restore(out/'resume_epoch12.pt',False);bad_history=train(bad,bo,bg,x,y,start,24)
    torch.save(snapshot(bad,bo,bg,24),out/'weights_only_epoch24.pt')
    gap=lambda a,b: max(float((a[k]-b[k]).abs().max()) for k in a)
    asset_names=['data.npz','config.json','resume_epoch12.pt','continuous_epoch24.pt','resumed_epoch24.pt','weights_only_epoch24.pt']
    manifest={n:digest(out/n) for n in asset_names}
    (out/'asset_manifest.json').write_text(json.dumps(manifest,indent=2))
    pristine=verify_assets(out,manifest)
    # In-memory simulated byte change: do not damage the retained artifact.
    blob=(out/'data.npz').read_bytes();modified=blob[:-1]+bytes([blob[-1]^1]);tamper_detected=hashlib.sha256(modified).hexdigest()!=manifest['data.npz']
    # Semantic duplicates survive a different record ID; exact row bytes are our key.
    rows=np.vstack([x.numpy(),x[:4].numpy()]);unique=len({row.tobytes() for row in rows})
    r={'config':CONFIG,'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'device':'CPU','threads':1,
       'histories':{'continuous':first+second,'resumed':resumed,'weights_only':bad_history},
       'resume_max_parameter_gap':gap(continuous,full.state_dict()),'weights_only_max_parameter_gap':gap(continuous,bad.state_dict()),
       'resume_loss_max_gap':max(abs(a['train_loss']-b['train_loss']) for a,b in zip(second,resumed)),
       'metrics':{'continuous':assess(m,x,y),'resumed':assess(full,x,y),'weights_only':assess(bad,x,y)},
       'manifest_failures':pristine,'simulated_tamper_detected':tamper_detected,'duplicate_demo':{'rows':len(rows),'unique_rows':unique,'duplicates':len(rows)-unique},
       'asset_graph':{'data.npz':['config.json'],'resume_epoch12.pt':['data.npz','config.json'],'continuous_epoch24.pt':['resume_epoch12.pt'],'resumed_epoch24.pt':['resume_epoch12.pt']},
       'limits':['Exact match only tested in same Python/PyTorch CPU environment with deterministic operations',
                 'Hash matching establishes byte integrity relative to a trusted manifest, not provenance, permission or model correctness',
                 'Toy exact-row deduplication does not identify semantic near-duplicates']}
    (out/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args();r=run(a.output)
    print(json.dumps({k:r[k] for k in ['resume_max_parameter_gap','weights_only_max_parameter_gap','resume_loss_max_gap','metrics','simulated_tamper_detected']},indent=2))
