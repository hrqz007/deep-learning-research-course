"""Recreate all original fixtures into a new chosen directory, without downloads."""
from pathlib import Path
import argparse,csv,io,json,hashlib
import numpy as np

def generate(output):
 out=Path(output)
 if out.exists():raise ValueError('Choose a new output directory; refusing to overwrite existing data')
 files={}
 for split,n,seed in [('source',512,41001),('target_train',64,41002),('target_validation',128,41003),('target_test',512,41004)]:
  rng=np.random.default_rng(seed);x=rng.uniform(-2,2,(n,2))
  if split!='source':x[:,0]=x[:,0]*.875+.25
  noise=rng.normal(0,.08,n);ys=np.tanh(1.2*x[:,0])+noise;yo=np.tanh(1.2*x[:,1])+noise
  f=io.StringIO();w=csv.writer(f,lineterminator='\n');w.writerow(['id','x1','x2','related','orthogonal']);w.writerows([f'{split}-{i:04}',*map(lambda v:format(v,'.17g'),[*x[i],ys[i],yo[i]])] for i in range(n));files[split+'.csv']=f.getvalue().encode()
 out.mkdir(parents=True)
 for name,b in files.items():(out/name).write_bytes(b)
 return {name:hashlib.sha256(b).hexdigest() for name,b in files.items()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(json.dumps(generate(a.output),indent=2))
