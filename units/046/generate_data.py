"""Original synthetic rectangles; no external files, model or network."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def generate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);metadata={'license':'CC0-1.0; original procedurally generated data','coordinates':'half-open XYXY; x=column edge, y=row edge; image NCHW, mask N1HW','splits':{}}
    for name,n,seed,shift in [('train',64,4601,0),('validation',32,4602,0),('test',48,4603,0),('shift',32,4604,.16)]:
        rng=np.random.default_rng(seed);x=[];y=[];records=[]
        for i in range(n):
            mode=i%4;mask=np.zeros((16,16),np.uint8);bb=[]
            if mode:
                left=int(rng.integers(1,9));top=int(rng.integers(1,9));w=int(rng.integers(3,7));h=int(rng.integers(3,7));bb=[[left,top,left+w,top+h]]
                if mode==2:bb.append([left+1,top+1,min(16,left+w+2),min(16,top+h+2)])
                if mode==3:bb=[[1,2,5,7],[10,9,14,14]]
            for l,t,r,b in bb:mask[t:b,l:r]=1
            base=float(rng.uniform(.05,.35))+shift;contrast=float(rng.uniform(.32,.52));noise=rng.normal(0,.17,(16,16));im=np.clip(base+contrast*mask+noise,0,1)
            x.append(im[None]);y.append(mask[None]);records.append({'id':f'{name}-{i:03}','kind':['empty','single','overlap','separate'][mode],'boxes':bb,'background':base,'contrast':contrast})
        arrays={'x':np.array(x,np.float64),'y':np.array(y,np.uint8)}
        hashes={}
        for key,a in arrays.items():p=out/f'{name}_{key}.npy';np.save(p,a);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        metadata['splits'][name]={'n':n,'seed':seed,'shift':shift,'records':records,'files':hashes}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n');return metadata
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'data'));a=p.parse_args();generate(a.output);print(a.output)
