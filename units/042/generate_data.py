"""Original synthetic repeated-measurement data. No downloads or personal data."""
from pathlib import Path
import argparse,csv,hashlib,io,json,os,tempfile
import numpy as np

def truth(x):return .7*np.sin(1.5*x[:,0])+.35*x[:,0]*x[:,1]+.25*x[:,1]**2

def generate(destination):
    destination=Path(destination).resolve()
    if destination.exists() and not destination.is_dir():raise ValueError('destination must be directory')
    payload={}
    for split,groups,seed in [('train',40,4200),('validation',32,4201),('test',160,4202)]:
        rng=np.random.default_rng(seed);centres=rng.uniform(-2,2,(groups,2));offsets=.25*rng.standard_normal(groups)
        buf=io.StringIO(newline='');writer=csv.writer(buf,lineterminator='\n');writer.writerow(['row_id','entity_id','replicate','x1','x2','y','region'])
        for g in range(groups):
            region='left' if centres[g,0]<0 else 'right';m=8 if region=='left' else 2
            x=centres[g]+.05*rng.standard_normal((m,2));y=truth(x)+offsets[g]+.08*rng.standard_normal(m)
            for j in range(m):writer.writerow([f'{split}-{g:03d}-{j}',f'{split}-{g:03d}',j,*[format(v,'.17g') for v in (*x[j],y[j])],region])
        payload[split+'.csv']=buf.getvalue().encode()
    destination.mkdir(parents=True,exist_ok=True)
    for name,b in payload.items():
        with tempfile.NamedTemporaryFile(dir=destination,delete=False) as f:f.write(b);temp=Path(f.name)
        os.replace(temp,destination/name)
    return {k:hashlib.sha256(b).hexdigest() for k,b in payload.items()}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',required=True,type=Path)
    print(json.dumps(generate(p.parse_args().destination),indent=2))
