"""DL070: small mechanism reproduction of dropout, not original benchmarks.

A fixed 2x2 regularization experiment with five paired initialization seeds.
The plan is saved and hashed before training. Validation selects checkpoints;
test outcomes never select hyperparameters, epochs or reported seeds.
"""
from pathlib import Path
import argparse,copy,json,hashlib,time,platform
import numpy as np
import torch
from torch import nn
from scipy.stats import t
SEEDS=[11,23,37,51,67]
MODES={'baseline':(0.,0.),'dropout':(.4,0.),'weight_decay':(0.,.01),'both':(.4,.01)}
PLAN={'task':'synthetic nonlinear binary classification','data_seed':7001,'train_n':128,'validation_n':192,'test_n':512,
      'seeds':SEEDS,'epochs':160,'optimizer':'Adam','lr':.01,'modes':MODES,'selection':'minimum validation NLL; earliest tie',
      'primary':'paired test NLL difference dropout - baseline; negative is better','secondary':'accuracy, calibration, factorial interaction',
      'hypothesis':'dropout may reduce overfitting at this fixed capacity and budget','scope':'mechanism-scale reproduction, not original paper benchmark scores'}

def data():
    rng=np.random.default_rng(7001);x=rng.normal(size=(832,20)).astype('float32')
    latent=x[:,0]*x[:,1]+.7*x[:,2]-.4*x[:,3]+.25*rng.normal(size=832)
    y=(latent>0).astype('int64');return torch.from_numpy(x),torch.from_numpy(y)
def model(p):return nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Dropout(p),nn.Linear(64,64),nn.ReLU(),nn.Dropout(p),nn.Linear(64,2))
def evaluate(m,x,y):
    m.eval()
    with torch.inference_mode():
        z=m(x);p=z.softmax(1);return {'nll':float(nn.functional.cross_entropy(z,y)),'accuracy':float((z.argmax(1)==y).float().mean()),
            'brier':float(((p[:,1]-y.float())**2).mean())}
def paired(values):
    v=np.asarray(values,float);n=len(v);se=float(v.std(ddof=1)/np.sqrt(n));half=float(t.ppf(.975,n-1)*se)
    return {'values':v.tolist(),'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1)),'standard_error':se,'ci95':[float(v.mean()-half),float(v.mean()+half)],
            'interpretation':'approximate t interval over initialization seeds, conditional on one fixed dataset'}
def run(output='outputs'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.dumps(PLAN,sort_keys=True,indent=2);(out/'plan.json').write_text(plan);plan_hash=hashlib.sha256(plan.encode()).hexdigest()
    x,y=data();np.savez(out/'data.npz',x=x.numpy(),y=y.numpy(),train_indices=np.arange(128),validation_indices=np.arange(128,320),test_indices=np.arange(320,832))
    runs=[]
    for seed in SEEDS:
        for mode,(p,wd) in MODES.items():
            torch.manual_seed(seed);m=model(p);initial=copy.deepcopy(m.state_dict());opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=wd)
            best=float('inf');history=[];selected=0;start=time.perf_counter()
            for epoch in range(160):
                m.train();opt.zero_grad();loss=nn.functional.cross_entropy(m(x[:128]),y[:128]);loss.backward();opt.step()
                val=evaluate(m,x[128:320],y[128:320]);train=evaluate(m,x[:128],y[:128])
                history.append({'epoch':epoch+1,'train_nll':train['nll'],'validation_nll':val['nll']})
                if val['nll']<best:best=val['nll'];state=copy.deepcopy(m.state_dict());selected=epoch+1
            elapsed=time.perf_counter()-start;m.load_state_dict(state);test=evaluate(m,x[320:],y[320:])
            torch.save({'state_dict':state,'seed':seed,'mode':mode,'p':p,'weight_decay':wd,'epoch':selected},out/f'{mode}_seed{seed}.pt')
            runs.append({'seed':seed,'mode':mode,'selected_epoch':selected,'best_validation_nll':best,'test':test,'history':history,'train_seconds':elapsed,
                         'initial_parameter_sum':float(sum(v.sum() for v in initial.values()))})
    lookup={(r['seed'],r['mode']):r for r in runs}
    delta=[lookup[s,'dropout']['test']['nll']-lookup[s,'baseline']['test']['nll'] for s in SEEDS]
    interaction=[lookup[s,'both']['test']['nll']-lookup[s,'dropout']['test']['nll']-lookup[s,'weight_decay']['test']['nll']+lookup[s,'baseline']['test']['nll'] for s in SEEDS]
    summary={mode:{metric:float(np.mean([r['test'][metric] for r in runs if r['mode']==mode])) for metric in ['nll','accuracy','brier']} for mode in MODES}
    r={'plan_sha256':plan_hash,'torch':torch.__version__,'python':platform.python_version(),'device':'CPU','threads':1,'runs':runs,'summary':summary,
       'paired_dropout_minus_baseline':paired(delta),'factorial_interaction':paired(interaction),
       'budget':{'runs':len(runs),'updates_per_run':160,'total_updates':3200,'train_examples_per_update':128,'train_seconds':sum(z['train_seconds'] for z in runs)},
       'limits':['One synthetic dataset, five initialization seeds, no original vision/speech/document benchmarks',
                 'No hyperparameter search; fixed dropout rate and weight decay',
                 'Seed interval excludes dataset and deployment distribution uncertainty',
                 'Validation-selected epochs differ despite equal maximum training budgets']}
    (out/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args();r=run(a.output)
    print(json.dumps({k:r[k] for k in ['summary','paired_dropout_minus_baseline','factorial_interaction','budget']},indent=2))
