"""Check saved weights, probabilities, metrics and notebook replay without training."""
from pathlib import Path
import json,numpy as np
from experiment import features,sigmoid,domains,METHODS,SEEDS
R=Path(__file__).resolve().parent;out=R/'outputs';d=np.load(out/'data.npz');w=np.load(out/'weights.npz');p=np.load(out/'predictions.npz');r=json.loads((out/'results.json').read_text());ds=domains(d['test'],d['test_y'],d['latent_a']);maxdiff=0.
for method in METHODS:
 for seed in SEEDS:
  key=f'{method}_{seed}'
  for domain,(x,y) in ds.items():
   pred=sigmoid(features(x,method)@w[key]);delta=float(np.max(np.abs(pred-p[key+'_'+domain])));maxdiff=max(maxdiff,delta)
   if delta>1e-12:raise RuntimeError('prediction replay mismatch')
   row=next(v for v in r['results'] if v['method']==method and v['seed']==seed and v['domain']==domain)
   if abs(np.mean((pred>=.5)!=y)-row['error'])>1e-12:raise RuntimeError('metric mismatch')
notebook=json.loads((R/'notebook_outputs/results.json').read_text())
if r['aggregate']!=notebook['aggregate']:raise RuntimeError('Notebook replay metrics differ')
print(json.dumps({'saved_model_predictions_max_abs_diff':maxdiff,'metrics_checked':len(r['results']),'notebook_aggregate_match':True,'status':'PASS'},indent=2))
