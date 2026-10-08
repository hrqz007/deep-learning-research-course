"""DL068: train, export and benchmark a complete CPU inference contract.

Data are original synthetic examples; no remote requests occur. Timings are local
observations, not service SLOs. --output chooses a separate reproducible run folder.
"""
from pathlib import Path
import argparse, copy, json, time, platform, subprocess, sys
import numpy as np
import torch
from torch import nn

def data():
    rng=np.random.default_rng(6801)
    z=rng.normal(size=(1152,4)).astype('float32')
    # Raw sensor units deliberately have unequal scales and nonzero offsets.
    x=z*np.array([3,.3,2,5],dtype='float32')+np.array([8,-4,2,10],dtype='float32')
    y=np.argmax(np.stack([z[:,0]+.4*z[:,1],-z[:,0]+.5*z[:,2],z[:,3]-.4],axis=1),axis=1)
    return x,y.astype('int64')

class Model(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        # Buffers travel with state_dict AND the exported graph.
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(4,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(.25),nn.Linear(32,3))
    def forward(self,raw):
        return self.net((raw-self.mean)/self.scale)

def predict(module,raw):
    """Reject malformed requests rather than emit a plausible-looking class."""
    try:
        a=np.asarray(raw)
    except (TypeError, ValueError):
        return {'status':'rejected','reason':'invalid array representation','prediction':None}
    if a.dtype!=np.float32 or a.ndim!=2 or a.shape!=(1,4) or not np.isfinite(a).all():
        return {'status':'rejected','reason':'expected finite float32 [1,4]','prediction':None}
    with torch.inference_mode():
        try:
            logits=module(torch.from_numpy(a))
        except (RuntimeError, ValueError):
            return {'status':'unavailable','reason':'backend execution failure','prediction':None}
        if not torch.isfinite(logits).all():
            return {'status':'unavailable','reason':'nonfinite output','prediction':None}
        return {'status':'ok','prediction':int(logits.argmax(1).item())}

def timing(module,x,repeats=120):
    # No CUDA is used; CPU calls finish before the next perf_counter_ns reading.
    with torch.inference_mode():
        for _ in range(30): module(x)
        ns=[]
        for _ in range(repeats):
            t=time.perf_counter_ns();module(x);ns.append(time.perf_counter_ns()-t)
    us=np.asarray(ns)/1000
    return {'samples_us':us.tolist(),'p50_us':float(np.quantile(us,.5,method='linear')),
            'p95_us':float(np.quantile(us,.95,method='linear')),'p99_us':float(np.quantile(us,.99,method='linear')),
            'mean_us':float(us.mean()),'throughput_examples_per_second':float(len(x)*1e6/us.mean())}

def fresh_process_timing(path,batch,repeats=3):
    """New interpreter each time; OS page cache is not forcibly dropped."""
    code = """import sys,json,time
start=time.perf_counter()
import torch
import numpy as np
torch.set_num_threads(1)
module=torch.export.load(sys.argv[1]).module()
a=np.load(sys.argv[2]);x=torch.from_numpy(a['x'][768:768+int(sys.argv[3])])
with torch.inference_mode(): z=module(x)
print(json.dumps({'import_load_first_call_ms':(time.perf_counter()-start)*1000,'finite':bool(torch.isfinite(z).all())}))
"""
    rows=[]
    for _ in range(repeats):
        start=time.perf_counter()
        child=subprocess.run([sys.executable,'-c',code,str(path.resolve()),str((path.parent/'data.npz').resolve()),str(batch)],capture_output=True,text=True,check=True)
        wall=(time.perf_counter()-start)*1000;row=json.loads(child.stdout)
        if not row['finite']:raise RuntimeError('nonfinite cold-start output')
        row['spawn_to_exit_ms']=wall;rows.append(row)
    return {'observations':rows,'note':'new Python interpreter and torch import each run; includes process exit in parent wall clock; OS file cache not reset; no container or network'}

def run(output='outputs'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(68);torch.use_deterministic_algorithms(True)
    x,y=data();train=torch.from_numpy(x[:768]);target=torch.from_numpy(y[:768]);test=torch.from_numpy(x[768:])
    m=Model(train.mean(0),train.std(0,unbiased=False));opt=torch.optim.Adam(m.parameters(),lr=.015)
    history=[]
    for epoch in range(180):
        m.train();opt.zero_grad();loss=nn.functional.cross_entropy(m(train),target);loss.backward();opt.step()
        if epoch%10==0 or epoch==179:
            m.eval()
            with torch.inference_mode():
                history.append({'epoch':epoch+1,'train_loss':float(loss.detach()),'test_accuracy':float((m(test).argmax(1).numpy()==y[768:]).mean())})
    m.eval();torch.save(m.state_dict(),out/'model.pt')
    np.savez(out/'data.npz',x=x,y=y,train_indices=np.arange(768),test_indices=np.arange(768,1152))
    with torch.inference_mode(): eager=m(test).numpy()
    exports={};timings={};cold={}; errors={};fresh={}
    for b in (1,32):
        # Static contracts are intentional; no promise of arbitrary batch sizes.
        sample=test[:b];ep=torch.export.export(m,(sample,),strict=True)
        # Debug stack traces contain build-machine paths, not inference semantics.
        for node in ep.graph_module.graph.nodes:
            node.meta.pop('stack_trace',None)
        path=out/f'model_batch{b}.pt2';torch.export.save(ep,path)
        cold_samples=[]
        for _ in range(5):
            t=time.perf_counter_ns();loaded=torch.export.load(path).module()
            with torch.inference_mode(): loaded(sample)
            cold_samples.append((time.perf_counter_ns()-t)/1e6)
        cold[str(b)]={'load_plus_first_call_ms':cold_samples,'note':'same running process, import and OS cold cache excluded'}
        exports[b]=loaded
        with torch.inference_mode(): exported=torch.cat([loaded(test[i:i+b]) for i in range(0,len(test),b)]).numpy()
        np.testing.assert_allclose(exported,eager,rtol=1e-5,atol=1e-5)
        errors[str(b)]={'max_abs_logit_error':float(np.max(np.abs(exported-eager))),
                        'argmax_mismatches':int(np.sum(exported.argmax(1)!=eager.argmax(1)))}
        timings[f'eager_{b}']=timing(m,sample);timings[f'export_{b}']=timing(loaded,sample)
        fresh[str(b)]=fresh_process_timing(path,b)
    # A wrong preprocessing implementation can preserve plausible tensor shapes.
    with torch.inference_mode(): bad=m.net(test).numpy()
    train_copy=copy.deepcopy(m).train()
    with torch.inference_mode():
        a=train_copy(test[:32]);b=train_copy(test[:32]); stochastic=float((a-b).abs().max())
    requests=[predict(exports[1],x[768:769]),predict(exports[1],np.full((1,4),np.nan,dtype='float32')),
              predict(exports[1],np.zeros((1,5),dtype='float32')),predict(exports[1],x[768:769].astype('float64'))]
    # Deterministic queue simulation is separately labelled; times are NOT measured.
    arrivals=np.array([0.,.2,.4,.6,.8,1.,1.2,1.4]);service=.5;finish=[];last=0.
    for arrival in arrivals: last=max(last,arrival)+service;finish.append(last)
    r={'seed':68,'data_seed':6801,'epochs':180,'train_n':768,'test_n':384,'dtype':'float32','device':'CPU','threads':1,
       'torch':torch.__version__,'python':platform.python_version(),'history':history,'export_checks':errors,
       'accuracy':float((eager.argmax(1)==y[768:]).mean()),'wrong_preprocessing_accuracy':float((bad.argmax(1)==y[768:]).mean()),
       'wrong_preprocessing_max_logit_error':float(np.max(np.abs(bad-eager))),'training_mode_repeat_max_difference':stochastic,
       'requests':requests,'timings':timings,'cold':cold,'fresh_process':fresh,
       'queue_simulation':{'arrival_ms':arrivals.tolist(),'finish_ms':finish,'response_ms':(np.array(finish)-arrivals).tolist(),'service_ms':service},
       'limits':['CPU single-process sequential microbenchmark; no network, server, concurrency or GPU',
                 'warm-process reload and new-interpreter startup measured separately; OS cache not reset; no container cold-start test',
                 'torch.export graph execution is not AOT compilation or universal deployment portability']}
    (out/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args();r=run(a.output)
    print(json.dumps({k:r[k] for k in ['accuracy','wrong_preprocessing_accuracy','export_checks','training_mode_repeat_max_difference']},indent=2))
