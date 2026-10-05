"""DL045: coordinate-safe transforms and replayable augmentation audit.

Core image helpers use CHW finite real float64 images; masks are HW integer IDs;
boxes are half-open XYXY edges. Explicitly no EXIF decoder, complex transforms,
antialiased general resampler, GPU, or automatic task-label inference.
"""
from pathlib import Path
from fractions import Fraction as F
from numbers import Integral
import argparse,gzip,hashlib,json,os,platform,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
ARMS=['none','background','background_rotate_stale','background_rotate_correct']

def real_array(x,name,ndim):
    a=np.asarray(x)
    if a.ndim!=ndim or any(s<=0 for s in a.shape) or a.size>1_000_000:raise ValueError(f'{name}: nonempty {ndim}D <=1e6 elements required')
    if a.dtype.kind not in 'fiu':raise ValueError(f'{name}: finite real values required')
    a=a.astype(np.float64,copy=False)
    if not np.isfinite(a).all():raise ValueError(f'{name}: nonfinite values')
    return a

def positive_integer(x,name):
    if not isinstance(x,Integral) or isinstance(x,(bool,np.bool_)) or x<=0:raise ValueError(name+' must be a positive integer')
    return int(x)

def normalize(x,mean,std):
    x=real_array(x,'image',3);mean=real_array(mean,'mean',1);std=real_array(std,'std',1)
    if mean.shape!=(x.shape[0],) or std.shape!=mean.shape or np.any(std<=0):raise ValueError('normalization channel mismatch or nonpositive std')
    with np.errstate(over='raise',invalid='raise',divide='raise'):out=(x-mean[:,None,None])/std[:,None,None]
    if not np.isfinite(out).all():raise FloatingPointError('normalization overflow')
    return out

def resize_bilinear(x,size):
    """Half-pixel mapping, align_corners=False, edge clamping, antialias=False."""
    x=real_array(x,'image',3)
    if not isinstance(size,(tuple,list)) or len(size)!=2:raise ValueError('size must be (height,width)')
    oh,ow=[positive_integer(v,'output size') for v in size];c,h,w=x.shape
    if c*oh*ow>1_000_000:raise ValueError('output resource limit')
    out=np.empty((c,oh,ow))
    with np.errstate(over='raise',invalid='raise'):
        for r in range(oh):
            u=max(0.,min(h-1.,(r+.5)*h/oh-.5));r0=int(np.floor(u));r1=min(r0+1,h-1);a=u-r0
            for col in range(ow):
                v=max(0.,min(w-1.,(col+.5)*w/ow-.5));c0=int(np.floor(v));c1=min(c0+1,w-1);b=v-c0
                out[:,r,col]=(1-a)*((1-b)*x[:,r0,c0]+b*x[:,r0,c1])+a*((1-b)*x[:,r1,c0]+b*x[:,r1,c1])
    if not np.isfinite(out).all():raise FloatingPointError('resize overflow')
    return out

def resize_mask(mask,size):
    a=np.asarray(mask)
    if a.ndim!=2 or any(s<=0 for s in a.shape) or a.dtype.kind not in 'iu' or a.size>1_000_000:raise ValueError('mask must be nonempty integer HW IDs')
    if not isinstance(size,(tuple,list)) or len(size)!=2:raise ValueError('size must be pair')
    oh,ow=[positive_integer(v,'size') for v in size]
    if oh*ow>1_000_000:raise ValueError('mask output resource limit')
    # Nearest-exact center convention; no interpolation of category IDs.
    rr=np.minimum((np.arange(oh)+.5)*a.shape[0]/oh,a.shape[0]-1).astype(int)
    cc=np.minimum((np.arange(ow)+.5)*a.shape[1]/ow,a.shape[1]-1).astype(int)
    return a[rr[:,None],cc[None,:]]

def crop_flip(x,mask,boxes,crop,flip=False):
    x=real_array(x,'image',3);m=np.asarray(mask);bs=np.asarray(boxes)
    if m.shape!=x.shape[1:] or m.dtype.kind not in 'iu':raise ValueError('integer mask shape mismatch')
    if bs.shape==(0,):bs=np.empty((0,4))
    if bs.ndim!=2 or bs.shape[1]!=4 or bs.dtype.kind not in 'fiu' or not np.isfinite(bs).all():raise ValueError('boxes require finite Kx4 XYXY')
    bs=bs.astype(float);H,W=m.shape
    if np.any(bs[:,0]<0) or np.any(bs[:,1]<0) or np.any(bs[:,2]>W) or np.any(bs[:,3]>H) or np.any(bs[:,2]<=bs[:,0]) or np.any(bs[:,3]<=bs[:,1]):raise ValueError('invalid half-open box edges')
    if not isinstance(crop,(tuple,list)) or len(crop)!=4 or any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) for v in crop):raise ValueError('crop is integer top,left,height,width')
    top,left,h,w=map(int,crop)
    if min(top,left)<0 or min(h,w)<=0 or top+h>H or left+w>W:raise ValueError('crop must lie inside image')
    if not isinstance(flip,(bool,np.bool_)):raise ValueError('flip must be bool')
    out=x[:,top:top+h,left:left+w].copy();om=m[top:top+h,left:left+w].copy();before=(bs[:,2]-bs[:,0])*(bs[:,3]-bs[:,1])
    bb=bs-np.array([left,top,left,top]);bb[:,[0,2]]=np.clip(bb[:,[0,2]],0,w);bb[:,[1,3]]=np.clip(bb[:,[1,3]],0,h)
    areas=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1]);keep=areas>0
    if flip:
        out=out[:,:,::-1].copy();om=om[:,::-1].copy();bb=bb[:,[2,1,0,3]];bb[:,[0,2]]=w-bb[:,[0,2]]
    return {'image':out,'mask':om,'boxes':bb[keep],'keep_indices':np.flatnonzero(keep),'box_area_retention':areas/before}

def hand_ledger():
    """Two transformed 2x2 images, affine head, exact Fraction chain."""
    raw=[[[F(0),F(1)],[F(1,2),F(1,4)]],[[F(1),F(1,2)],[F(3,4),F(1,4)]]]
    targets=[F(7,16),F(5,8)];theta=[F(1,5),F(-1,10),F(1,10),F(3,10),F(0)];states=[]
    # Both images flip horizontally; fixed mean=1/2 and std=1/2. Targets are the original spatial mean; flipping preserves that target exactly.
    features=[[2*v-1 for row in x for v in row[::-1]] for x in raw]
    for step in range(3):
        samples=[];grad=[F(0)]*5
        for u,t in zip(features,targets):
            pred=sum(w*v for w,v in zip(theta[:4],u))+theta[4];r=pred-t;loss=r*r/2;g=r/2
            contrib=[g*v for v in u]+[g];grad=[a+b for a,b in zip(grad,contrib)]
            # dL/d(original pixels): undo horizontal flip after scaling derivative 1/std=2.
            transformed_dx=[2*g*w for w in theta[:4]];dx=[transformed_dx[1],transformed_dx[0],transformed_dx[3],transformed_dx[2]]
            samples.append({'u':u,'prediction':pred,'residual':r,'loss':loss,'upstream':g,'contribution':contrib,'original_input_gradient':dx})
        states.append({'step':step,'theta':theta.copy(),'samples':samples,'gradient':grad,'loss':sum(s['loss'] for s in samples)/2})
        if step<2:theta=[v-F(1,10)*g for v,g in zip(theta,grad)]
    def encode(x):
        if isinstance(x,F):return {'exact':str(x),'float':float(x)}
        if isinstance(x,list):return [encode(v) for v in x]
        if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
        return x
    return encode(states)

def replay(split,plan,step,arm):
    if arm not in ARMS:raise ValueError('unknown augmentation arm')
    x=np.array(split['x'],float);mask=np.array(split['mask'],float);y=np.array(split['y'],np.int64)
    if arm!='none':
        old=np.array(split['background']);new=np.array(plan['background'][step]);x=x+(new-old)[:,None,None,None]*(1-mask)
    if 'rotate' in arm:
        flags=np.array(plan['rotate90'][step],bool)
        for i in np.flatnonzero(flags):x[i]=np.rot90(x[i],1,axes=(-2,-1))
        if arm.endswith('correct'):y[flags]=1-y[flags]
    return x,y

def model_values(params,x):
    k,b,w,c=params
    z=torch.nn.functional.conv2d(x,k,b,padding=1);h=torch.relu(z);logit=h.flatten(1)@w+c
    return logit,z,h

def numpy_gradient(params,x,y):
    """Independent NumPy contractions plus explicit kernel scatter, no autograd."""
    k,b,w,c=[np.asarray(p,float) for p in params];xp=np.pad(x,((0,0),(0,0),(1,1),(1,1)))
    patches=np.lib.stride_tricks.sliding_window_view(xp,(3,3),axis=(2,3));z=np.einsum('ncrsab,ocab->nors',patches,k)+b[None,:,None,None];h=np.maximum(z,0);flat=h.reshape(len(x),-1);logits=flat@w+c
    loss=np.mean(np.logaddexp(0,logits)-y*logits);q=(1/(1+np.exp(-logits))-y)/len(x);dw=flat.T@q;dc=q.sum();dz=(q[:,None]*w).reshape(h.shape)*(z>0)
    dk=np.einsum('nors,ncrsab->ocab',dz,patches);db=dz.sum((0,2,3));return float(loss),(dk,db,dw,np.array(dc))

def metrics(params,x,y):
    with torch.no_grad():
        logits=model_values(params,x)[0];loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none');correct=((logits>=0)==(y>=.5))
    return {'nll':float(loss.mean()),'accuracy':float(correct.double().mean()),'logits':logits.tolist(),'per_image_nll':loss.tolist()}

def training(data,plans):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=data['protocol'];base=np.array(data['train']['x'],float);mean=float(base.mean());std=float(base.std());runs=[];traces=[];max_ref=0.
    tensor=lambda v:torch.tensor(v,dtype=torch.float64)
    eval_sets={}
    for key in ['validation','test']:
        for view in (['x'] if key=='validation' else ['x','swapped','neutral']):eval_sets[key+'_'+view]=(tensor((np.array(data[key][view])-mean)/std),tensor(data[key]['y']))
    for seed in protocol['initialization_seeds']:
        rng=np.random.default_rng(seed);init=[rng.normal(0,.15,(4,1,3,3)),np.full(4,.05),rng.normal(0,.04,256),np.array(0.)]
        for arm in ARMS:
            p=[tensor(v).requires_grad_() for v in init];trace=[]
            for step in range(protocol['steps']):
                xx,yy=replay(data['train'],plans[str(seed)],step,arm);x=(xx-mean)/std;xt=tensor(x);yt=tensor(yy);logit=model_values(p,xt)[0];loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,yt);loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(q.grad).all() for q in p):raise FloatingPointError('nonfinite fixed-protocol training state')
                if step in [0,79,159]:
                    nl,ng=numpy_gradient([q.detach().numpy() for q in p],x,yy);max_ref=max(max_ref,abs(nl-float(loss.detach())),*(float(np.max(np.abs(g-q.grad.numpy()))) for g,q in zip(ng,p)))
                trace.append({'step':step,'augmented_train_nll':float(loss.detach()),'parameters':[q.detach().numpy().ravel().tolist() for q in p]})
                with torch.no_grad():
                    for q in p:q-=protocol['lr']*q.grad;q.grad=None
            trace.append({'step':protocol['steps'],'parameters':[q.detach().numpy().ravel().tolist() for q in p]})
            met={name:metrics(p,x,y) for name,(x,y) in eval_sets.items()};runs.append({'seed':seed,'arm':arm,'metrics':met,'final_parameters':[q.detach().numpy().ravel().tolist() for q in p]});traces.append({'seed':seed,'arm':arm,'states':trace})
    summary={arm:{view:{metric:float(np.mean([r['metrics'][view][metric] for r in runs if r['arm']==arm])) for metric in ['nll','accuracy']} for view in eval_sets} for arm in ARMS}
    return {'protocol':protocol,'normalization':{'mean':mean,'std':std,'fitted_on':'unaugmented training pixels only'},'parameters':297,'total_updates':len(runs)*protocol['steps'],'runs':runs,'summary':summary,'numpy_reference_max_abs_gap':max_ref},traces

def geometry_demo():
    x=np.arange(24.).reshape(1,4,6)/23;mask=np.zeros((4,6),np.int64);mask[1:3,1:5]=2;boxes=np.array([[1,1,5,3],[0,0,1,1]],float)
    g=crop_flip(x,mask,boxes,(1,2,3,4),True)
    return {k:v.tolist() for k,v in g.items()}|{'original_image':x.tolist(),'original_mask':mask.tolist(),'original_boxes':boxes.tolist(),'crop':[1,2,3,4],'flip':True,'bilinear_image':resize_bilinear(x,(3,5)).tolist(),'categorical_mask':[[0,2],[2,0]],'wrong_bilinear_mask':resize_bilinear(np.array([[[0.,2.],[2.,0.]]]),(3,3))[0].tolist(),'nearest_mask':resize_mask(np.array([[0,2],[2,0]]),(3,3)).tolist()}

def run(out):
    manifest=json.loads((ROOT/'data/data_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=digest:raise ValueError('fixed data hash mismatch '+name)
    data=json.loads((ROOT/'data/images.json').read_text());plans=json.loads((ROOT/'data/augmentation_plans.json').read_text());learning,traces=training(data,plans)
    results={'hand':hand_ledger(),'geometry':geometry_demo(),'learning':learning,'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU float64'}}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(traces,separators=(',',':'),allow_nan=False)+'\n').encode();compressed=gzip.compress(raw,mtime=0);results['trace_record']={'file':'training_traces.json.gz','raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'compressed_sha256':hashlib.sha256(compressed).hexdigest()}
    payload=(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    for name,content in [('training_traces.json.gz',compressed),('results.json',payload)]:
        with tempfile.NamedTemporaryFile(dir=out,delete=False) as f:f.write(content);tmp=Path(f.name)
        os.replace(tmp,out/name)
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args();r=run(a.output);print(json.dumps({'summary':r['learning']['summary'],'max_reference_error':r['learning']['numpy_reference_max_abs_gap'],'normalization':r['learning']['normalization']},indent=2))
