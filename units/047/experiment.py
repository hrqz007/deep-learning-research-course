"""DL047: patch embeddings, small ViT/hybrid and bounded comparison.

Offline CPU float64 experiment. Patch API supports finite NCHW and exact divisibility;
there is no silent crop/pad or arbitrary resolution interpolation.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,math,os,platform,tempfile
from numbers import Integral
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent

def patchify(x,p):
    a=np.asarray(x)
    if a.ndim!=4 or any(v<=0 for v in a.shape) or a.size>1_000_000 or a.dtype.kind not in 'fiu':raise ValueError('patchify requires nonempty finite real NCHW <=1e6 elements')
    if not isinstance(p,Integral) or isinstance(p,(bool,np.bool_)) or p<=0:raise ValueError('positive integer patch side required')
    a=a.astype(np.float64,copy=False)
    if not np.isfinite(a).all():raise ValueError('nonfinite input')
    n,c,h,w=a.shape
    if h%p or w%p:raise ValueError('image sides must be exactly divisible by patch side; no silent crop')
    return a.reshape(n,c,h//p,p,w//p,p).transpose(0,2,4,1,3,5).reshape(n,(h//p)*(w//p),c*p*p)

def unpatchify(tokens,p,shape):
    if not isinstance(shape,(tuple,list)) or len(shape)!=4 or any(not isinstance(v,Integral) or isinstance(v,bool) or v<=0 for v in shape):raise ValueError('positive NCHW shape required')
    if not isinstance(p,Integral) or isinstance(p,(bool,np.bool_)) or p<=0:raise ValueError('positive patch side')
    n,c,h,w=map(int,shape)
    if n*c*h*w>1_000_000 or h%p or w%p:raise ValueError('invalid or too-large target')
    a=np.asarray(tokens)
    if a.shape!=(n,(h//p)*(w//p),c*p*p) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():raise ValueError('token shape/value mismatch')
    return a.reshape(n,h//p,w//p,c,p,p).transpose(0,3,1,4,2,5).reshape(shape)

def attention_numpy(z,q,k,v):
    Q=z@q;K=z@k;V=z@v;S=Q@K.swapaxes(-1,-2)/math.sqrt(Q.shape[-1]);A=np.exp(S-S.max(-1,keepdims=True));A/=A.sum(-1,keepdims=True);return A@V,(Q,K,V,A)

def hand_ledger():
    """Independent NumPy scalar-chain gradients, separate from full torch ViT.
    Two row-patches, D=1, no LN/FFN/residual. All 9 parameters are trainable.
    """
    xs=np.array([[[1.,0.],[0.,1.]],[[0.,1.],[1.,1.]]]);ys=np.array([1.,0.]);theta=np.array([.5,-.25,0.,.25,.5,.5,1.,1.,0.]);states=[]
    for step in range(3):
        w=theta[:2];pos=theta[2:4];q,k,v,o,c=theta[4:];samples=[];total=np.zeros(9)
        for x,y in zip(xs,ys):
            e=x@w+pos;Q=e*q;K=e*k;V=e*v;S=np.outer(Q,K);A=np.exp(S-S.max(1,keepdims=True));A/=A.sum(1,keepdims=True);z=A@V;m=z.mean();pred=o*m+c;res=pred-y;loss=res*res/2;up=res/2
            bz=np.full(2,up*o/2);bV=A.T@bz;bA=np.outer(bz,V);bS=A*(bA-(A*bA).sum(1,keepdims=True));bQ=bS@K;bK=bS.T@Q;be=bQ*q+bK*k+bV*v
            contribution=np.r_[x.T@be,be,np.dot(bQ,e),np.dot(bK,e),np.dot(bV,e),up*m,up]
            total+=contribution;samples.append({'patches':x.tolist(),'embedding':e.tolist(),'Q':Q.tolist(),'K':K.tolist(),'V':V.tolist(),'scores':S.tolist(),'attention':A.tolist(),'context':z.tolist(),'mean_context':float(m),'prediction':float(pred),'residual':float(res),'loss':float(loss),'prediction_upstream':float(up),'context_upstream':bz.tolist(),'attention_upstream':bA.tolist(),'score_upstream':bS.tolist(),'Q_upstream':bQ.tolist(),'K_upstream':bK.tolist(),'V_upstream':bV.tolist(),'embedding_upstream':be.tolist(),'contribution':contribution.tolist(),'input_gradient':np.outer(be,w).tolist()})
        states.append({'step':step,'theta':theta.tolist(),'samples':samples,'gradient':total.tolist(),'loss':float(np.mean([s['loss'] for s in samples]))})
        if step<2:theta=theta-.1*total
    return states

class TinyAttention(nn.Module):
    def __init__(self,d=8,heads=2):
        super().__init__();self.d=d;self.heads=heads;self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d)
    def forward(self,x):
        b,n,d=x.shape;qkv=self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4);q,k,v=qkv.unbind(0)
        A=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(d//self.heads),dim=-1);self.last_attention=A.detach()
        return self.proj((A@v).transpose(1,2).reshape(b,n,d))
class VisionModel(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='cnn':self.conv=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU());self.head=nn.Linear(8,2)
        elif kind in ('vit','hybrid'):
            self.stem=nn.Identity() if kind=='vit' else nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.ReLU());self.patch=nn.Linear(4 if kind=='vit' else 16,8);self.pos=nn.Parameter(torch.zeros(1,16,8));self.norm1=nn.LayerNorm(8);self.attn=TinyAttention();self.norm2=nn.LayerNorm(8);self.ff=nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8));self.final_norm=nn.LayerNorm(8);self.head=nn.Linear(8,2)
        else:raise ValueError('kind must be cnn/vit/hybrid')
    def forward(self,x):
        if self.kind=='cnn':return self.head(self.conv(x).mean((2,3)))
        x=self.stem(x);b,c,h,w=x.shape
        if (h,w)!=(8,8):raise ValueError('this model has fixed8x8 resolution and16 position embeddings')
        tokens=x.reshape(b,c,4,2,4,2).permute(0,2,4,1,3,5).reshape(b,16,c*4);z=self.patch(tokens)+self.pos
        z=z+self.attn(self.norm1(z));z=z+self.ff(self.norm2(z));return self.head(self.final_norm(z).mean(1))

def new_model(kind,seed):
    with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);model=VisionModel(kind).double()
    return model

def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():logits=model(x);loss=nn.functional.cross_entropy(logits,y,reduction='none');correct=logits.argmax(-1)==y
    return {'nll':float(loss.mean()),'accuracy':float(correct.double().mean()),'logits':logits.tolist(),'per_image_nll':loss.tolist()}

def saved_tensor_audit(kind,batch):
    """Observed CPU autograd saved-tensor footprint, explicitly not GPU peak memory."""
    model=new_model(kind,4711);x=torch.zeros((batch,1,8,8),dtype=torch.float64);y=torch.zeros(batch,dtype=torch.long);items=[]
    def pack(t):
        st=t.untyped_storage();items.append({'shape':list(t.shape),'logical_bytes':t.numel()*t.element_size(),'storage_key':(str(t.device),st.data_ptr(),st.nbytes()),'storage_bytes':st.nbytes()});return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):loss=nn.functional.cross_entropy(model(x),y);loss.backward()
    unique={tuple(i['storage_key']):i['storage_bytes'] for i in items}
    return {'kind':kind,'batch':batch,'parameter_count':sum(p.numel() for p in model.parameters()),'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'saved_tensor_logical_bytes':sum(i['logical_bytes'] for i in items),'unique_saved_storage_bytes':sum(unique.values()),'save_events':[{'shape':i['shape'],'logical_bytes':i['logical_bytes']} for i in items],'attention_score_bytes':0 if kind=='cnn' else batch*2*16*16*8,'notes':'CPU autograd forward+backward instrumentation; saved payload includes parameter/input storage and aliases; not allocator peak, RSS or VRAM'}

def permutation_audit():
    rng=np.random.default_rng(4790);z=rng.normal(size=(1,4,3));pos=rng.normal(size=z.shape);q=rng.normal(size=(3,2));k=rng.normal(size=(3,2));v=rng.normal(size=(3,3));perm=np.array([2,0,3,1]);base,_=attention_numpy(z,q,k,v);other,_=attention_numpy(z[:,perm],q,k,v);a,_=attention_numpy(z+pos,q,k,v);b,_=attention_numpy(z[:,perm]+pos,q,k,v);joint,_=attention_numpy((z+pos)[:,perm],q,k,v)
    return {'permutation':perm.tolist(),'no_position_equivariance_max_abs':float(np.max(np.abs(other-base[:,perm]))),'no_position_mean_invariance_max_abs':float(np.max(np.abs(other.mean(1)-base.mean(1)))),'fixed_position_mean_change':float(np.max(np.abs(a.mean(1)-b.mean(1)))),'joint_permutation_mean_invariance_max_abs':float(np.max(np.abs(a.mean(1)-joint.mean(1))))}

def orientation_reference(data):
    """Post-hoc nonlearned diagnostic; neural protocol/results remain unchanged."""
    results={}
    for split in ['train','validation','test']:
        x=np.array(data[split]['x'])[:,0];y=np.array(data[split]['y'])
        horizontal=(x[:,:,:-1]*x[:,:,1:]).sum((1,2));vertical=(x[:,:-1,:]*x[:,1:,:]).sum((1,2));score=horizontal-vertical;prediction=(score>=0).astype(int)
        results[split]={'accuracy':float(np.mean(prediction==y)),'score':score.tolist(),'prediction':prediction.tolist()}
    return {'status':'post-hoc diagnostic added after viewing first frozen neural runs; not part of original fixed neural comparison','formula':'sum horizontal adjacent products minus sum vertical adjacent products; score>=0 predicts horizontal','parameters':0,'training_updates':0,'threshold':0,'scope':'uses known orientation-task structure; no NLL because scores are not calibrated probability logits','splits':results}

def run(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);raw=(ROOT/'data/images.json').read_bytes();expected=json.loads((ROOT/'data/data_manifest.json').read_text())['images_sha256']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('fixed input hash mismatch')
    data=json.loads(raw);protocol=data['protocol'];train=torch.tensor(data['train']['x'],dtype=torch.float64);trainy=torch.tensor(data['train']['y'],dtype=torch.long);validation=torch.tensor(data['validation']['x'],dtype=torch.float64);valy=torch.tensor(data['validation']['y'],dtype=torch.long);test=torch.tensor(data['test']['x'],dtype=torch.float64);testy=torch.tensor(data['test']['y'],dtype=torch.long)
    runs=[];traces=[];states=[]
    for size in protocol['train_sizes']:
        x=train[:size];y=trainy[:size]
        for kind in protocol['models']:
            for seed in protocol['seeds']:
                model=new_model(kind,seed);optimizer=torch.optim.SGD(model.parameters(),lr=protocol['lr']);trace=[]
                for step in range(protocol['steps']+1):
                    model.train();logits=model(x);loss=nn.functional.cross_entropy(logits,y)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training objective')
                    trace.append({'step':step,'train_nll':float(loss.detach()),'train_accuracy':float((logits.argmax(-1)==y).double().mean()),'parameters':[p.detach().numpy().ravel().tolist() for p in model.parameters()]})
                    if step==protocol['steps']:break
                    optimizer.zero_grad();loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite training gradient')
                    optimizer.step()
                result={'train_size':size,'kind':kind,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'train':evaluate(model,x,y),'validation':evaluate(model,validation,valy),'test':evaluate(model,test,testy)};runs.append(result);traces.append({'train_size':size,'kind':kind,'seed':seed,'states':trace});states.append({'train_size':size,'kind':kind,'seed':seed,'state_dict':{n:t.detach().tolist() for n,t in model.state_dict().items()}})
    summary={str(n):{kind:{m:float(np.mean([r['test'][m] for r in runs if r['kind']==kind and r['train_size']==n])) for m in ['nll','accuracy']} for kind in protocol['models']} for n in protocol['train_sizes']}
    results={'orientation_reference':orientation_reference(data),'hand':hand_ledger(),'permutation':permutation_audit(),'protocol':protocol,'runs':runs,'summary':summary,'resource_audit':[saved_tensor_audit(k,b) for b in protocol['train_sizes'] for k in protocol['models']],'gpu':{'cuda_available':torch.cuda.is_available(),'peak_vram_bytes':None,'status':'not measured; CPU-only runtime'},'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__}}
    out=Path(out);out.mkdir(parents=True,exist_ok=True);payloads={}
    for name,value in [('training_traces.json.gz',traces),('final_states.json.gz',states)]:
        raw=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode();compressed=gzip.compress(raw,mtime=0);payloads[name]=compressed;results[name]={'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'uncompressed_bytes':len(raw),'compressed_sha256':hashlib.sha256(compressed).hexdigest()}
    payloads['results.json']=(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    for name,raw in payloads.items():
        with tempfile.NamedTemporaryFile(dir=out,delete=False) as f:f.write(raw);temp=Path(f.name)
        os.replace(temp,out/name)
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args();r=run(a.output);print(json.dumps({'summary':r['summary'],'resource':[{k:v for k,v in x.items() if k not in ['save_events','notes']} for x in r['resource_audit']],'gpu':r['gpu']},indent=2))
