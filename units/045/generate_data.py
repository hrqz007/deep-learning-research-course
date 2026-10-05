"""Original bar-orientation images and replay parameters; no external images."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
STEPS=160;NTRAIN=48;SEEDS=[4511,4512,4513]
def dataset(n,seed):
    rng=np.random.default_rng(seed);y=np.arange(n)%2;cy=rng.integers(3,5,n);cx=rng.integers(3,5,n)
    masks=np.zeros((n,1,8,8));boxes=[]
    for i in range(n):
        if y[i]==0: r0,r1,c0,c1=cy[i]-2,cy[i]+2,cx[i],cx[i]+1
        else:r0,r1,c0,c1=cy[i],cy[i]+1,cx[i]-2,cx[i]+2
        masks[i,0,r0:r1,c0:c1]=1;boxes.append([int(c0),int(r0),int(c1),int(r1)])
    background=np.where(y==0,.15,.75);noise=rng.normal(0,.025,masks.shape)
    # Foreground intensity .95; background correlation is deliberately perfect.
    x=.95*masks+background[:,None,None,None]*(1-masks)+noise
    swap=.95*masks+(1-background)[:,None,None,None]*(1-masks)+noise
    neutral=.95*masks+.45*(1-masks)+noise
    return {'x':x.tolist(),'y':y.tolist(),'mask':masks.tolist(),'background':background.tolist(),'noise':noise.tolist(),'boxes_xyxy':boxes,'swapped':swap.tolist(),'neutral':neutral.tolist(),'seed':seed}
def generate():
    d={s:dataset(n,seed) for s,n,seed in [('train',48,4501),('validation',48,4502),('test',128,4503)]}
    d['protocol']={'steps':STEPS,'initialization_seeds':SEEDS,'train_samples':NTRAIN,'arms':['none','background','background_rotate_stale','background_rotate_correct'],'lr':.2,'channels':4,'background_sampling':'Uniform[.1,.8], independent of label','rotate_probability':.5,'validation':'descriptive same-correlation readout only, no selection','test':'frozen paired original/swapped/neutral views, same images and noise','normalization':'training unaugmented global mean/std once, then fixed in all arms'}
    return d

def plans():
    p={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed+1000)
        p[str(seed)]={'background':rng.uniform(.1,.8,(STEPS,NTRAIN)).tolist(),'rotate90':(rng.random((STEPS,NTRAIN))<.5).astype(int).tolist()}
    return p
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);manifest={}
    for name,value in [('images.json',generate()),('augmentation_plans.json',plans())]:
        raw=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode();(a.output/name).write_bytes(raw);manifest[name]=hashlib.sha256(raw).hexdigest()
    (a.output/'data_manifest.json').write_text(json.dumps({'origin':'original synthetic orientation bars and fixed replay parameters','sha256':manifest,'numpy':np.__version__},indent=2)+'\n');print(manifest)
