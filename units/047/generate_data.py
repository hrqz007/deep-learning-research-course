"""Original offline orientation images; nested train sizes fixed before fitting."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def split(n,seed):
    rng=np.random.default_rng(seed);x=rng.normal(0,.16,(n,1,8,8));y=np.arange(n)%2;center=rng.integers(2,6,(n,2))
    for j in range(n):
        r,c=center[j]
        if y[j]==0:x[j,0,r-1:r+2,c]+=1
        else:x[j,0,r,c-1:c+2]+=1
    return {'x':x.tolist(),'y':y.tolist(),'center_rc':center.tolist(),'seed':seed}
def generate():
    return {'train':split(128,4701),'validation':split(64,4702),'test':split(256,4703),'protocol':{'train_sizes':[32,128],'steps':200,'lr':.15,'seeds':[4711,4712,4713],'models':['cnn','vit','hybrid'],'batch':'full chosen training subset','optimizer':'SGD without momentum or weight decay','selection':'none; fixed schedule and architecture; validation descriptive only','dtype':'float64','device':'cpu','pretraining':'none','augmentation':'none','normalization':'none; synthetic intensities in declared generator units'}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);raw=(json.dumps(generate(),separators=(',',':'),allow_nan=False)+'\n').encode();(a.output/'images.json').write_bytes(raw);(a.output/'data_manifest.json').write_text(json.dumps({'images_sha256':hashlib.sha256(raw).hexdigest(),'numpy':np.__version__,'origin':'original synthetic 8x8 bars, balanced fixed class counts, independent noise/positions'},indent=2)+'\n');print(len(raw),hashlib.sha256(raw).hexdigest())
