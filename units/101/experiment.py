"""Deterministic capacity simulation, NOT measured GPU/production serving."""
from pathlib import Path
import os,json,csv,time,platform
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl-course-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/dl-course-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';FIG=ROOT/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
PREFILL_BASE=.002;PREFILL_PER_TOKEN=.0001;DECODE_BASE=.0008;DECODE_PER_ACTIVE=.00015

def workload(seed,rate,n=180):
 rng=np.random.default_rng(seed);a=np.cumsum(rng.exponential(1/rate,n));p=rng.choice([32,128,512],n,p=[.5,.3,.2]);o=rng.choice([8,32,64],n,p=[.4,.4,.2]);return a,p,o

def simulate(arrivals,prompts,outputs,capacity=4,continuous=True):
 n=len(arrivals);t=0.;next_id=0;active=[];remaining=outputs.copy();start=np.full(n,np.nan);first=np.full(n,np.nan);end=np.full(n,np.nan);trace=[]
 while next_id<n or active:
  if not active and next_id<n:t=max(t,float(arrivals[next_id]))
  # Static batches admit only when empty. Continuous policy refills freed slots each iteration.
  if continuous or not active:
   eligible_time=t
   while next_id<n and len(active)<capacity and arrivals[next_id]<=eligible_time:
    i=next_id;next_id+=1;start[i]=t;t+=PREFILL_BASE+PREFILL_PER_TOKEN*prompts[i];active.append(i)
  if not active:continue
  t+=DECODE_BASE+DECODE_PER_ACTIVE*len(active)
  for i in active:
   if np.isnan(first[i]):first[i]=t
   remaining[i]-=1
   if remaining[i]==0:end[i]=t
  active=[i for i in active if remaining[i]>0]
  queue=int(np.sum(arrivals[next_id:]<=t));trace.append((t,len(active),queue))
 records=[dict(request=i,arrival=float(arrivals[i]),prompt_tokens=int(prompts[i]),output_tokens=int(outputs[i]),start=float(start[i]),first_token=float(first[i]),finish=float(end[i]),queue_wait=float(start[i]-arrivals[i]),ttft=float(first[i]-arrivals[i]),latency=float(end[i]-arrivals[i])) for i in range(n)]
 return records,trace

def parallel_model(workers,gradient_bytes=64*1024**2,bandwidth=12*1024**3,latency=.00002,serial_compute=.08):
 communication=2*(workers-1)/workers*gradient_bytes/bandwidth+2*(workers-1)*latency
 step=serial_compute/workers+communication
 return dict(workers=workers,ideal_compute_seconds=serial_compute/workers,ring_communication_seconds=communication,step_seconds=step,speedup=serial_compute/step,efficiency=(serial_compute/step)/workers)

def main():
 start_wall=time.perf_counter();arrays={};requests=[];summaries=[];traces=[]
 for rate in (5,15,30,60):
  for seed in (11,23,37):
   a,p,o=workload(seed,rate)
   for name,x in [('arrivals',a),('prompts',p),('outputs',o)]:arrays[f'{name}_r{rate}_s{seed}']=x
   for policy,cap,cont in [('serial',1,True),('static4',4,False),('continuous4',4,True),('continuous8',8,True)]:
    rs,tr=simulate(a,p,o,cap,cont)
    for r in rs:r.update(rate=rate,seed=seed,policy=policy);requests.append(r)
    latency=np.array([r['latency'] for r in rs]);ttft=np.array([r['ttft'] for r in rs]);span=max(r['finish'] for r in rs)-a[0]
    summaries.append(dict(rate=rate,seed=seed,policy=policy,n=len(rs),simulated_elapsed_seconds=float(span),simulated_request_throughput=len(rs)/span,latency_p50=float(np.quantile(latency,.5)),latency_p95=float(np.quantile(latency,.95)),ttft_p95=float(np.quantile(ttft,.95)),fraction_latency_over_1s=float(np.mean(latency>1)),peak_queue=max(t[2] for t in tr),completed_tokens=int(o.sum()),metric_kind='simulated service seconds, not measured hardware'))
    if seed==11 and rate==60:
     for t,active,queue in tr:traces.append(dict(policy=policy,time=t,active=active,queue=queue))
 parallel=[parallel_model(p) for p in (1,2,4,8,16)]
 np.savez(OUT/'workloads.npz',**arrays)
 for name,data in [('requests.csv',requests),('summary.csv',summaries),('queue_trace.csv',traces),('parallel_model.csv',parallel)]:
  with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 metrics=dict(protocol='Synthetic discrete-event simulator; all service time constants assumed, not measured. 180 requests per workload, 3 seeds, 4 arrival rates, common workloads across policies. No KV cache memory model, GPU, network, or distributed execution.',assumptions=dict(prefill_base_seconds=PREFILL_BASE,prefill_seconds_per_token=PREFILL_PER_TOKEN,decode_base_seconds=DECODE_BASE,decode_seconds_per_active_request=DECODE_PER_ACTIVE,prefill_policy='serial unchunked admission; pauses current decode',arrival='Poisson arrivals; finite replay drained to completion',slo_seconds=1.0),summary=summaries,parallel_model=parallel,simulator_cpu_wall_seconds=time.perf_counter()-start_wall)
 (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
 fig,ax=plt.subplots(1,2,figsize=(9,3.5))
 for policy in ('serial','static4','continuous4','continuous8'):
  rates=[5,15,30,60];ax[0].plot(rates,[np.mean([r['latency_p95'] for r in summaries if r['policy']==policy and r['rate']==rr]) for rr in rates],'o-',label=policy);ax[1].plot(rates,[np.mean([r['simulated_request_throughput'] for r in summaries if r['policy']==policy and r['rate']==rr]) for rr in rates],'o-',label=policy)
 ax[0].axhline(1,color='gray',linestyle='--');ax[0].set(xlabel='Assumed arrival requests/s',ylabel='Simulated p95 latency (s)');ax[1].set(xlabel='Assumed arrival requests/s',ylabel='Simulated drained throughput');ax[0].legend(fontsize=7);fig.suptitle('Capacity model only - no production performance claim');fig.tight_layout();fig.savefig(FIG/'01_service_model.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(9,3.3))
 for policy in ('serial','static4','continuous4','continuous8'):
  tr=[r for r in traces if r['policy']==policy];ax[0].plot([r['time'] for r in tr],[r['queue'] for r in tr],label=policy)
 ax[0].set(xlabel='Simulated time (s)',ylabel='Waiting requests');ax[0].legend(fontsize=7);ax[1].plot([r['workers'] for r in parallel],[r['speedup'] for r in parallel],'o-',label='Ring + ideal compute model');ax[1].plot([1,16],[1,16],'--',label='Ideal linear');ax[1].set(xlabel='Assumed workers',ylabel='Modeled training speedup');ax[1].legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/'02_queue_parallel.png',dpi=160);plt.close(fig)
 (ROOT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,execution='CPU simulation only',hardware_performance_measured=False,timing_scope='simulator wall time is implementation runtime; service times are assumed model units'),indent=2));print(json.dumps({'simulation_runs':len(summaries),'request_records':len(requests),'cpu_simulator_seconds':metrics['simulator_cpu_wall_seconds'],'all_service_metrics':'SIMULATED'},indent=2))
if __name__=='__main__':main()
