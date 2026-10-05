"""Original synthetic independent delayed-bit sequences. No external corpus."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def make_split(n,delay,seed):
    if isinstance(n,bool) or not isinstance(n,int) or not 2<=n<=4096 or n%2: raise ValueError('n must be even, 2..4096')
    if isinstance(delay,bool) or not isinstance(delay,int) or not 1<=delay<=127: raise ValueError('delay must be 1..127')
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32: raise ValueError('seed must be a uint32 integer')
    rng=np.random.default_rng(seed+1000*delay)
    y=np.concatenate([np.zeros(n//2),np.ones(n//2)]).astype(np.float64);rng.shuffle(y)
    x=np.zeros((n,delay+1,3),dtype=np.float64)
    x[:,:,0]=rng.uniform(-.5,.5,size=(n,delay+1))
    x[:,0,0]=2*y-1;x[:,0,1]=1.;x[:,-1,2]=1.
    return x,y

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());out={}
    for d in protocol['delays']:
        for split,n in protocol['counts'].items():
            x,y=make_split(n,d,protocol['data_seeds'][split]);out[f'd{d}_{split}_x']=x;out[f'd{d}_{split}_y']=y
    path=ROOT/'data/delayed_sign.npz';np.savez_compressed(path,**out)
    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in out.items()}
    (ROOT/'data/manifest.json').write_text(json.dumps({'format':'NumPy NPZ float64','array_shapes':{k:list(v.shape) for k,v in out.items()},'array_sha256':hashes,'generation':'make_split with protocol.json; exact balanced labels; uniform distractors'},indent=2))
    print(path.name,len(out),'arrays')
if __name__=='__main__':main()
