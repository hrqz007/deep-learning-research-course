"""Regenerate only the synthetic input fixtures in a user-selected directory."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np

def generate(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,n,seed in [('train',48,4040),('validation',128,4041),('test',512,4042)]:
        rng=np.random.default_rng(seed);x=rng.uniform(-3,3,n)
        truth=.8*np.tanh(1.3*x)+.25*np.sin(3*x)
        sigma=np.where(x<0,.1,.35);y=truth+sigma*rng.standard_normal(n)
        path=destination/(name+'.csv')
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['id','x','y','group'])
            for i in range(n):w.writerow([f'{name}-{i:04d}',format(x[i],'.17g'),format(y[i],'.17g'),'left' if x[i]<0 else 'right'])
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True)
    print(json.dumps(generate(parser.parse_args().destination),indent=2))
