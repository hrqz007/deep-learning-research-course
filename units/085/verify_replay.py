"""Re-run the experiment in a fresh output directory and compare non-timing metrics."""
from pathlib import Path
import tempfile,json,math
import experiment
ROOT=Path(__file__).resolve().parent
IGNORE={'observed_cpu_seconds'}
def compare(a,b,path='root'):
 if isinstance(a,dict):
  if set(a)!=set(b):raise AssertionError(path+' keys differ')
  for k in a:
   if k not in IGNORE:compare(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  if len(a)!=len(b):raise AssertionError(path+' lengths differ')
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
 elif isinstance(a,float):
  if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):raise AssertionError(path+f': {a} != {b}')
 elif a!=b:raise AssertionError(path+f': {a} != {b}')
def main():
 old=json.loads((ROOT/'outputs/metrics.json').read_text())
 with tempfile.TemporaryDirectory(prefix='course-replay-') as tmp:new=experiment.main(tmp)
 compare(old,new)
 report={'status':'passed','comparison':'all non-timing metrics recursively compared','rtol':1e-10,'atol':1e-12,'ignored_fields':sorted(IGNORE)}
 (ROOT/'outputs/replay-comparison.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
