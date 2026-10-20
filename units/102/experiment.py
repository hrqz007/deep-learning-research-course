"""Real CPU quality/cost study with fixed synchronous replay, no online service claim."""
from pathlib import Path
import os,time,json,csv,platform
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl-course-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/dl-course-cache')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';FIG=ROOT/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
WIDTHS=(8,32,64);SEEDS=(11,23,37);BATCHES=(1,16,64);REPEATS=5
THRESHOLDS={'all_accuracy_min':.90,'tail_accuracy_min':.80,'batch_response_p95_max_seconds':.002}
def dataset(seed,n):
 r=np.random.default_rng(seed);g=r.random(n)<.2;x=r.normal(size=(n,2)).astype('float32');x[g,0]+=3;y=(np.sin(2*x[:,0])+.6*x[:,1]>0).astype('int64');return x,y,g

def eligible(accuracy,tail,p95):return accuracy>=THRESHOLDS['all_accuracy_min'] and tail>=THRESHOLDS['tail_accuracy_min'] and p95<=THRESHOLDS['batch_response_p95_max_seconds']

def pareto(rows):
 # Minimize parameter bytes and measured replay elapsed; maximize overall accuracy.
 out=[]
 for i,a in enumerate(rows):
  dominated=False
  for j,b in enumerate(rows):
   if i==j:continue
   no_worse=b['parameter_bytes']<=a['parameter_bytes'] and b['replay_seconds_median']<=a['replay_seconds_median'] and b['accuracy']>=a['accuracy']
   strict=b['parameter_bytes']<a['parameter_bytes'] or b['replay_seconds_median']<a['replay_seconds_median'] or b['accuracy']>a['accuracy']
   if no_worse and strict:dominated=True;break
  if not dominated:out.append(a['config'])
 return out

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data={};models={};quality=[];predictions={};weights={}
 for split,seed,n in [('train',404,1800),('test',606,1200)]:
  for name,a in zip(('x','y','group'),dataset(seed,n)):data[split+'_'+name]=a
 rng=np.random.default_rng(707);mixed=rng.choice(1200,256,replace=False);tail_pool=np.flatnonzero(data['test_group']);tail=rng.choice(tail_pool,256,replace=True)
 data['replay_mixed_ids']=mixed;data['replay_tail_ids']=tail;np.savez(OUT/'data.npz',**data)
 tx=torch.from_numpy(data['train_x']);ty=torch.from_numpy(data['train_y']);xx=torch.from_numpy(data['test_x'])
 for width in WIDTHS:
  for seed in SEEDS:
   torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(2,width),torch.nn.ReLU(),torch.nn.Linear(width,2));opt=torch.optim.Adam(model.parameters(),lr=.015);t=time.perf_counter()
   for _ in range(700):opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(tx),ty);loss.backward();opt.step()
   training=time.perf_counter()-t;model.eval();name=f'w{width}_s{seed}';models[name]=model
   with torch.no_grad():logits=model(xx).numpy()
   predictions[name]=logits;correct=logits.argmax(1)==data['test_y'];g=data['test_group'];size=sum(p.numel()*p.element_size() for p in model.parameters())
   for k,p in enumerate(model.parameters()):weights[f'{name}_p{k}']=p.detach().numpy()
   quality.append(dict(model=name,width=width,seed=seed,accuracy=float(correct.mean()),common_accuracy=float(correct[~g].mean()),tail_accuracy=float(correct[g].mean()),test_n=len(g),tail_n=int(g.sum()),parameter_bytes=size,training_seconds=training))
 np.savez(OUT/'predictions.npz',**predictions);np.savez(OUT/'parameters.npz',**weights)
 timings=[];replays=[];configs=[(name,batch,work) for name in models for batch in BATCHES for work in ('mixed','tail')]
 for repeat in range(REPEATS):
  order=np.random.default_rng(900+repeat).permutation(len(configs))
  for ci in order:
   name,batch,work=configs[ci];model=models[name];ids=data['replay_'+work+'_ids'];inputs=xx[ids];latencies=[]
   with torch.no_grad():
    for _ in range(5):model(inputs[:batch])
    replay_start=time.perf_counter()
    for start in range(0,len(inputs),batch):
     chunk=inputs[start:start+batch];t=time.perf_counter();out=model(chunk);elapsed=time.perf_counter()-t
     timings.append(dict(model=name,batch=batch,workload=work,repeat=repeat,batch_start=start,n=len(chunk),seconds=elapsed));latencies.extend([elapsed]*len(chunk))
    replay_elapsed=time.perf_counter()-replay_start
   replays.append(dict(model=name,batch=batch,workload=work,repeat=repeat,n=256,replay_seconds=replay_elapsed,requests_per_second=256/replay_elapsed,batch_response_p50_seconds=float(np.quantile(latencies,.5)),batch_response_p95_seconds=float(np.quantile(latencies,.95))))
 summary=[]
 for name,batch,work in configs:
  q=next(q for q in quality if q['model']==name);rs=[r for r in replays if r['model']==name and r['batch']==batch and r['workload']==work];tm=[r for r in timings if r['model']==name and r['batch']==batch and r['workload']==work];lat=np.concatenate([np.repeat(r['seconds'],r['n']) for r in tm]);p95=float(np.quantile(lat,.95));elapsed=float(np.median([r['replay_seconds'] for r in rs]));summary.append(dict(config=f'{name}_b{batch}_{work}',model=name,width=q['width'],seed=q['seed'],batch=batch,workload=work,accuracy=q['accuracy'],tail_accuracy=q['tail_accuracy'],parameter_bytes=q['parameter_bytes'],replay_seconds_median=elapsed,requests_per_second=256/elapsed,batch_response_p95_seconds=p95,eligible=eligible(q['accuracy'],q['tail_accuracy'],p95)))
 fronts={work:pareto([r for r in summary if r['workload']==work and r['eligible']]) for work in ('mixed','tail')}
 for fn,rs in [('quality.csv',quality),('batch_timings.csv',timings),('replays.csv',replays),('summary.csv',summary)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 metrics=dict(protocol='Real CPU float32, one thread, 700 training steps; 5 warmups per replay; 5 repeated synchronous replays; configuration order randomized. No arrivals or queue. Batch response time excludes queue/assembly; replay elapsed includes Python iteration/measurement bookkeeping.',thresholds=THRESHOLDS,quality=quality,summary=summary,feasible_pareto=fronts,training_seconds_total=sum(q['training_seconds'] for q in quality),replay_length=256,tail_replay_unique_examples=len(np.unique(tail)),test_tail_n=int(data['test_group'].sum()),cost_scope='Actual parameter tensor bytes and measured CPU wall times; no process peak memory, energy, money, GPU, network, or production SLO claim.')
 (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
 fig,ax=plt.subplots(1,2,figsize=(9,3.6))
 for width in WIDTHS:
  rows=[r for r in summary if r['width']==width and r['workload']=='mixed'];ax[0].scatter([r['replay_seconds_median']*1000 for r in rows],[r['accuracy'] for r in rows],label=f'width {width}');ax[1].scatter([r['parameter_bytes'] for r in rows],[r['tail_accuracy'] for r in rows],label=f'width {width}')
 ax[0].axhline(.90,color='gray',linestyle='--');ax[1].axhline(.80,color='gray',linestyle='--');ax[0].set(xlabel='Measured 256-request replay median (ms)',ylabel='Overall test accuracy');ax[1].set(xlabel='Actual float parameter bytes',ylabel='Tail-group test accuracy');ax[0].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'01_quality_cost.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.5))
 for work in ('mixed','tail'):
  vals=[np.median([r['batch_response_p95_seconds'] for r in summary if r['workload']==work and r['batch']==b])*1e6 for b in BATCHES];ax.plot(BATCHES,vals,'o-',label=work)
 ax.set(xlabel='Synchronous batch size',ylabel='Median of configuration p95s (microseconds)',title='Measured batch response only: no online queue');ax.legend();fig.tight_layout();fig.savefig(FIG/'02_tail_latency.png',dpi=160);plt.close(fig)
 (ROOT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='CPU',threads=1,dtype='float32',machine=platform.machine(),platform=platform.system(),gpu_used=False),indent=2));print(json.dumps(dict(thresholds=THRESHOLDS,quality=quality,feasible_configurations=sum(r['eligible'] for r in summary),total_configurations=len(summary),feasible_pareto=fronts,tail_replay_unique_examples=metrics['tail_replay_unique_examples']),indent=2))
if __name__=='__main__':main()
