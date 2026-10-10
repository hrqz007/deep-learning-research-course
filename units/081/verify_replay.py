"""Compare a fresh script replay with the previously executed Notebook outputs.
Wall-clock metrics are deliberately excluded, never claimed bit-identical.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,os
R=Path(__file__).resolve().parent
paths=list((R/'data').glob('*.csv'))+list((R/'outputs').glob('*.csv'))
before={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
old=json.loads((R/'outputs/metrics.json').read_text())
def numerical(x):
 if isinstance(x,dict):return {k:numerical(v) for k,v in x.items() if not k.endswith('_seconds')}
 if isinstance(x,list):return [numerical(v) for v in x]
 return x
env=dict(os.environ);env['OPENBLAS_NUM_THREADS']='1'
r=subprocess.run([sys.executable,'experiment.py'],cwd=R,env=env,capture_output=True,text=True,check=True)
(R/'outputs/standalone-execution.json').write_text(r.stdout)
new=json.loads((R/'outputs/metrics.json').read_text())
after={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if before!=after:raise RuntimeError('CSV replay mismatch')
if numerical(old)!=numerical(new):raise RuntimeError('Numerical metrics replay mismatch')
report={'status':'passed','csv_count':len(paths),'csv_bytes_exact_match':True,'numerical_metrics_exact_match':True,'excluded':'all *_seconds fields; wall clock is not deterministic'}
(R/'outputs/replay-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
