"""Generate original synthetic data without the teaching convolution implementation."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
K=np.array([[0.2,-0.1,0.0],[0.1,0.5,-0.2],[0.0,0.1,0.3]])

def generate():
    data={'true_kernel':K.tolist(),'true_bias':0.15}
    for split,n,seed in [('train',12,4301),('test',20,4302)]:
        rng=np.random.default_rng(seed);x=rng.uniform(-1,1,(n,1,6,6))
        # Independent sliding-window/einsum generator, not experiment.conv_forward.
        patches=np.lib.stride_tricks.sliding_window_view(x,(3,3),axis=(2,3))
        y=np.einsum('ncrsij,ij->ncrs',patches,K)+0.15+rng.normal(0,0.05,(n,1,4,4))
        data[split+'_x']=x.tolist();data[split+'_y']=y.tolist()
    # Fixed integer-valued probe makes boundary and sampling failures unambiguous.
    rng=np.random.default_rng(4303);data['symmetry_input']=rng.integers(-2,4,(8,8)).tolist()
    data['symmetry_kernel']=[[0.,1.,0.],[-1.,2.,0.5],[0.,-0.5,1.]]
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(generate(),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();(args.output/'experiment_data.json').write_bytes(raw)
    manifest={'experiment_data_sha256':hashlib.sha256(raw).hexdigest(),'origin':'original synthetic arrays; no people or external dataset','numpy':np.__version__,'seeds':[4301,4302,4303],'noise_std':0.05,'train_images':12,'test_images':20}
    (args.output/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
