"""Original synthetic horizontal/vertical bars; no external datasets or people."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent

def dataset(n,seed):
    if isinstance(n,bool) or not isinstance(n,int) or n<2 or n%2: raise ValueError('n must be positive even integer >=2')
    rng=np.random.default_rng(seed); y=np.tile([0,1],n//2);rng.shuffle(y)
    x=rng.normal(0,0.30,(n,1,12,12)); meta=[]
    for i,label in enumerate(y):
        row,col=rng.integers(3,9,size=2);length=int(rng.integers(5,8)); amp=float(rng.uniform(.7,1.3))
        # Width 1, odd/even lengths permitted, every stripe remains inside 12x12.
        start= (col if label==0 else row)-length//2
        start=max(0,min(12-length,start))
        if label==0: x[i,0,row,start:start+length]+=amp
        else: x[i,0,start:start+length,col]+=amp
        meta.append([int(row),int(col),length,amp,start])
    return x.astype('float32'),y.astype('int64'),np.array(meta,dtype='float64')

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest={}
    for split,n,seed in [('train',256,4400),('validation',128,4401),('test',512,4402)]:
        x,y,m=dataset(n,seed)
        for name,a in [('x',x),('y',y),('meta',m)]:
            p=out/f'{split}_{name}.npy';np.save(p,a,allow_pickle=False)
            manifest[p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),shape=list(a.shape),dtype=str(a.dtype))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(HERE/'data'));main(p.parse_args().output)
