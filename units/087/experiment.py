"""Frozen synthetic task evaluation: paired ablations and task-cluster bootstrap."""
from pathlib import Path
import json,random,re,hashlib,platform
import numpy as np
ROOT=Path(__file__).resolve().parent

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def fixture():
 docs=[{'id':f'E{i:03d}','value':round(2+i*.125,3),'unit':'a.u.'} for i in range(40)];tasks=[]
 for i in range(80):
  group=['exact','alias','missing','denied'][i//20];j=i%20
  if group=='exact':query=f'read E{j:03d}';expected={'status':'ok','id':f'E{j:03d}','value':docs[j]['value']}
  elif group=='alias':query=f'read sample-{20+j}';expected={'status':'ok','id':f'E{20+j:03d}','value':docs[20+j]['value']}
  elif group=='missing':query=f'read E{100+j:03d}';expected={'status':'abstain'}
  else:query=f'read PRIVATE{j:03d}';expected={'status':'denied'}
  tasks.append({'id':f'T{i:03d}','group':group,'query':query,'expected':expected})
 return docs,tasks

def run_system(task,docs,variant='full',seed=0,budget=2):
 if variant not in ('full','no_normalization','no_retry','no_citation','no_retrieval'):raise ValueError('unknown variant')
 # The controller receives only id/query, never expected or group.
 query=task['query'];rng=random.Random(seed*10000+int(task['id'][1:]));events=[];attempts=0;retrieved=[]
 if variant=='no_retrieval':return {'status':'abstain','events':['skip_retrieval'],'attempts':0,'retrieved':[],'simulated_cost_units':1}
 if variant!='no_normalization':query=re.sub(r'sample-(\d+)',lambda m:f'E{int(m[1]):03d}',query)
 events.append('retrieve')
 private=re.search(r'PRIVATE\d+',query)
 if private:return {'status':'denied','events':events+['permission_denied'],'attempts':0,'retrieved':[],'simulated_cost_units':1}
 match=re.search(r'E\d+',query);rid=match[0] if match else docs[0]['id'];doc=next((d for d in docs if d['id']==rid),None)
 if doc is None:return {'status':'abstain','events':events+['missing'],'attempts':0,'retrieved':[],'simulated_cost_units':1}
 retrieved=[rid];fails_first=rng.random()<.3
 for attempt in range(min(budget,1 if variant=='no_retry' else 2)):
  attempts+=1;events.append('tool_call')
  if attempt==0 and fails_first:events.append('transient_failure');continue
  events.append('answer');return {'status':'ok','value':doc['value'],'citation':None if variant=='no_citation' else rid,'events':events,'attempts':attempts,'retrieved':retrieved,'simulated_cost_units':1+2*attempts}
 return {'status':'failed','events':events+['budget_exhausted'],'attempts':attempts,'retrieved':retrieved,'simulated_cost_units':1+2*attempts}

def score(task,result):
 exp=task['expected'];answerable=exp['status']=='ok';value_ok=result['status']=='ok' and answerable and abs(result['value']-exp['value'])<1e-12
 success=(value_ok and result.get('citation')==exp['id']) if answerable else result['status']==exp['status']
 return {'success':int(success),'answerable':answerable,'value_only':int(value_ok) if answerable else int(result['status']==exp['status']),'retrieval_hit':int(exp.get('id') in result['retrieved']) if answerable else None,'tool_failure':int('transient_failure' in result['events']),'cost':result['simulated_cost_units'],'attempts':result['attempts']}

def paired_bootstrap(a,b,seed=87,nboot=4000):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if a.shape!=b.shape or a.ndim!=1 or len(a)==0:raise ValueError('paired nonempty task vectors required')
 d=a-b;rng=np.random.default_rng(seed);means=d[rng.integers(0,len(d),(nboot,len(d)))].mean(1);lo,hi=np.quantile(means,[.025,.975]);return {'mean_difference':float(d.mean()),'ci95':[float(lo),float(hi)],'resampling_unit':'task after averaging repeated seeds','bootstrap_draws':nboot}

def summarize(tasks,rows):
 success=np.array([[r['score']['success'] for r in run] for run in rows]);flat=[r for run in rows for r in run];ans=[r for r in flat if r['score']['answerable']]
 return {'success_rate':float(success.mean()),'value_only_score':float(np.mean([r['score']['value_only'] for r in flat])),'retrieval_recall_answerable':float(np.mean([r['score']['retrieval_hit'] for r in ans])),'mean_simulated_cost':float(np.mean([r['score']['cost'] for r in flat])),'p95_simulated_cost':float(np.quantile([r['score']['cost'] for r in flat],.95)),'tool_failure_incidence':float(np.mean([r['score']['tool_failure'] for r in flat])),'groups':{g:float(np.mean([r['score']['success'] for r in flat if r['group']==g])) for g in ['exact','alias','missing','denied']},'per_task_mean_success':success.mean(0).tolist()}

def main(out=None):
 out=Path(out or ROOT/'outputs');out.mkdir(parents=True,exist_ok=True);docs,tasks=fixture();variants=['full','no_normalization','no_retry','no_citation','no_retrieval'];results={};traces={}
 for v in variants:
  rows=[]
  for seed in [11,22,33]:
   run=[]
   for task in tasks:
    visible={k:task[k] for k in ['id','query']};r=run_system(visible,docs,v,seed);run.append({'task_id':task['id'],'group':task['group'],'seed':seed,'result':r,'score':score(task,r)})
   rows.append(run)
  traces[v]=rows;results[v]=summarize(tasks,rows)
 comparisons={v:paired_bootstrap(results['full']['per_task_mean_success'],results[v]['per_task_mean_success']) for v in variants[1:]}
 data={'version':'evaluation-1.0','task_hash':digest(tasks),'corpus_hash':digest(docs),'tasks':80,'seeds':[11,22,33],'budget_tool_attempts_per_task':2,'cost_definition':'1 unit retrieval plus 2 units per attempted mock tool; no measured latency or API price','results':results,'paired_full_minus_ablation':comparisons}
 (ROOT/'data'/'tasks_frozen.json').write_text(json.dumps(tasks,indent=2));(ROOT/'data'/'corpus.json').write_text(json.dumps(docs,indent=2));(out/'metrics.json').write_text(json.dumps(data,indent=2));(out/'traces.json').write_text(json.dumps(traces,indent=2));(out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__},indent=2));return data
if __name__=='__main__':print(json.dumps(main(),indent=2))
