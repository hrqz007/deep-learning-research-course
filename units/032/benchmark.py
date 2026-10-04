"""Measure current CPU time separately; never overwrite deterministic outputs."""
from pathlib import Path
import argparse,hashlib,json,math,platform,statistics,time
import experiment as e
import torch

def measure(output,data_dir=e.ROOT/'data',repetitions=5):
 e.require(type(repetitions) is int and 3<=repetitions<=15,'Use 3 to 15 replicates')
 x,y,orders,cfg=e.load_inputs(data_dir);torch.set_num_threads(1)
 cases=[(b,cfg['sample_budget']//b) for b in cfg['batch_sizes']]
 for b,steps in cases:e.train(x,y,orders,cfg['initial_parameters'],b,steps,cfg['learning_rate'],record=False)
 records={b:[] for b,_ in cases}
 for rep in range(repetitions):
  rotated=cases[rep%len(cases):]+cases[:rep%len(cases)]
  for b,steps in rotated:
   start=time.perf_counter_ns();result=e.train(x,y,orders,cfg['initial_parameters'],b,steps,cfg['learning_rate'],record=False);elapsed=time.perf_counter_ns()-start
   e.require(elapsed>0,'Nonpositive measured duration');records[b].append(elapsed)
 # Independent final numeric equality with the record=True report path.
 for b,steps in cases:
  a=e.train(x,y,orders,cfg['initial_parameters'],b,steps,cfg['learning_rate'],record=False)
  z=e.train(x,y,orders,cfg['initial_parameters'],b,steps,cfg['learning_rate'],record=True)
  e.require(a['parameters']==z['parameters'],'Timing path changed math')
 rows=[]
 for b,steps in cases:
  ns=records[b];median=statistics.median(ns);rows.append({'batch_size':b,'steps':steps,'examples':cfg['sample_budget'],'nanoseconds':ns,'median_seconds':median/1e9,'minimum_seconds':min(ns)/1e9,'maximum_seconds':max(ns)/1e9,'examples_per_second':cfg['sample_budget']/(median/1e9)})
 report={'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':1},'protocol':{'warmups_each':1,'repetitions':repetitions,'rotating_order':True,'timer':'time.perf_counter_ns','included':'complete train(record=False) call: validation, index flattening, thread setting, tensor initialization, gather/forward/backward/update and guard checks','excluded':'imports, reading/generating original data, full-risk history logging, final comparison audit and file I/O','limits':'Shared CPU wall times vary. Tiny-network implementation timing, not a GPU throughput or general algorithmic superiority claim.'},'input_sha256':e.INPUT_HASHES,'script_sha256':hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest(),'rows':rows}
 content=(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode();output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix(output.suffix+'.tmp');tmp.write_bytes(content);tmp.replace(output);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data-dir',type=Path,default=e.ROOT/'data');p.add_argument('--repetitions',type=int,default=5);a=p.parse_args();print(json.dumps(measure(a.output,a.data_dir,a.repetitions),indent=2))
if __name__=='__main__':main()
