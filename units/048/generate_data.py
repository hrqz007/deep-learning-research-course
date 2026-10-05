"""Original synthetic corpus: each base object belongs to one split before augmentation."""
from pathlib import Path
import argparse, hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
SPLITS=('train','dev_iid','dev_shift','test_iid','test_shift')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make_split(n,seed,stamp_match):
    if isinstance(n,bool) or not isinstance(n,int) or n<4 or n>10000 or n%2:raise ValueError('n: even integer 4..10000')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('seed: uint32 integer')
    if not np.isfinite(stamp_match) or not 0<=stamp_match<=1:raise ValueError('stamp probability outside [0,1]')
    rng=np.random.default_rng(seed);y=np.tile(np.array([0,1],np.int64),n//2);rng.shuffle(y)
    xs=[];meta=[];rr,cc=np.mgrid[:20,:20]
    for i,label in enumerate(y):
        cx,cy=rng.uniform(8.2,11.2,2);radius=rng.uniform(3.3,4.7);contrast=float(rng.uniform(.18,.40));sigma=float(rng.uniform(.08,.18))
        d=np.sqrt((rr-cy)**2+(cc-cx)**2);shape=(d<=radius)&((d>=radius*.60) if label else True)
        im=.22+rng.normal(0,sigma,(20,20));im+=contrast*shape
        stamp=int(label if rng.random()<stamp_match else 1-label)
        im[:5,:5]=(.82 if stamp else .06)+rng.normal(0,.015,(5,5));im=np.clip(im,0,1).astype(np.float32)
        xs.append(im[None]);meta.append(dict(index=i,label=int(label),stamp=stamp,cx=float(cx),cy=float(cy),radius=float(radius),contrast=contrast,noise_sigma=sigma))
    return np.stack(xs),y,meta

def generate(output=HERE/'data'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);p=json.loads((HERE/'protocol.json').read_text());manifest={'generator':'048-original-v1','split_before_augmentation':True,'files':{},'splits':{}}
    seen=set()
    for name in SPLITS:
        x,y,meta=make_split(**p[name]);np.save(output/f'{name}_x.npy',x,allow_pickle=False);np.save(output/f'{name}_y.npy',y,allow_pickle=False)
        for row,im in zip(meta,x):
            row['base_id']=f"{name}-{row['index']:04d}";row['image_sha256']=hashlib.sha256(im.tobytes()).hexdigest()
            if row['image_sha256'] in seen:raise RuntimeError('duplicate image bytes across splits')
            seen.add(row['image_sha256'])
        (output/f'{name}_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        manifest['splits'][name]={'n':len(y),'class_counts':np.bincount(y,minlength=2).tolist(),'stamp_match_actual':float(np.mean([r['label']==r['stamp'] for r in meta]))}
    for f in sorted(output.glob('*')):
        if f.name!='manifest.json' and f.is_file():manifest['files'][f.name]=sha(f)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'data');a=p.parse_args();print(json.dumps(generate(a.output)['splits'],indent=2))
