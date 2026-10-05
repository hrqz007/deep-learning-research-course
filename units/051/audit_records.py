"""Recompute every retained SGD update using independent NumPy BPTT."""
from pathlib import Path
import json,gzip,argparse
import numpy as np
from experiment import numpy_reference
ROOT=Path(__file__).resolve().parent
def audit(output):
 data=json.loads((ROOT/'data/sequences.json').read_text());r=json.loads((output/'results.json').read_text());tr=json.loads(gzip.decompress((output/'training_traces.json.gz').read_bytes()));maxloss=maxstep=maxnorm=0.;updates=0
 for run in tr:
  if len(run['states'])!=301:raise ValueError('missing trajectory states')
  for s,nxt in zip(run['states'],run['states'][1:]):
   p={k:np.array(v) for k,v in s['parameters'].items()};loss,g,*_=numpy_reference(p,run['kind'],data['train']);norm=float(np.sqrt(sum(np.square(v).sum() for v in g.values())));factor=min(1.,1./norm) if norm else 1.;maxloss=max(maxloss,abs(loss-s['train_nll']));maxnorm=max(maxnorm,abs(norm-s['gradient_norm_before_clip']))
   if abs(factor-s['clip_factor'])>1e-12:raise RuntimeError('recorded clip factor mismatch')
   for k in p:maxstep=max(maxstep,float(np.abs(p[k]-.5*factor*g[k]-np.array(nxt['parameters'][k])).max()))
   updates+=1
 if updates!=1800 or maxstep>1e-10 or maxloss>1e-10 or maxnorm>1e-9:raise RuntimeError((updates,maxstep,maxloss,maxnorm))
 return {'updates':updates,'states':1806,'max_update_abs':maxstep,'max_loss_abs':maxloss,'max_gradient_norm_abs':maxnorm}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');print(json.dumps(audit(p.parse_args().output),indent=2))
