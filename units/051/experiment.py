"""Masked encoder-decoder RNN/GRU, full recorded SGD and genuine free decoding.

GRU is the PyTorch reset-after convention: gate order r,z,n, two biases.
The NumPy reference implements its own complete encoder/decoder reverse pass.
"""
from pathlib import Path
from numbers import Integral
import argparse,json,gzip,hashlib,math,os,platform,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PAD,BOS,EOS,UNK=0,1,2,3

def batch(sequences,pad_extra=0):
 if not isinstance(sequences,(list,tuple)) or not 1<=len(sequences)<=256:raise ValueError('1..256 content sequences required')
 if not isinstance(pad_extra,Integral) or isinstance(pad_extra,(bool,np.bool_)) or not 0<=pad_extra<=32:raise ValueError('padding extra integer0..32')
 for s in sequences:
  if not isinstance(s,(list,tuple)) or not 0<=len(s)<=32 or any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) or v not in range(4,8) for v in s):raise ValueError('content list length0..32 with integer IDs4..7')
 B=len(sequences);S=max(map(len,sequences))+2+pad_extra;T=max(map(len,sequences))+1+pad_extra;src=np.zeros((B,S),np.int64);din=np.zeros((B,T),np.int64);target=din.copy();sm=np.zeros_like(src,bool);dm=np.zeros_like(din,bool)
 for i,s in enumerate(sequences):
  source=[BOS]+list(s)+[EOS];output=list(reversed(s))+[EOS];inp=[BOS]+output[:-1];src[i,:len(source)]=source;sm[i,:len(source)]=True;din[i,:len(inp)]=inp;target[i,:len(output)]=output;dm[i,:len(output)]=True
 return src,sm,din,target,dm

def initialize(kind,seed,D=8,H=12):
 if kind not in ['rnn','gru']:raise ValueError('kind rnn/gru')
 if any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) or not 1<=v<=64 for v in [D,H]):raise ValueError('D/H must be integer1..64')
 if not isinstance(seed,Integral) or isinstance(seed,(bool,np.bool_)) or not 0<=seed<=2**32-1:raise ValueError('seed must be uint32 integer')
 rng=np.random.default_rng(seed);p={'E':rng.normal(0,.25,(8,D))};p['E'][0]=0;G=1 if kind=='rnn' else 3
 for prefix in ['enc','dec']:
  p[prefix+'W']=rng.normal(0,1/math.sqrt(D),(G*H,D));p[prefix+'U']=rng.normal(0,1/math.sqrt(H),(G*H,H));p[prefix+'bi']=np.zeros(G*H);p[prefix+'bh']=np.zeros(G*H)
 p['outW']=rng.normal(0,1/math.sqrt(H),(H,8));p['outb']=np.zeros(8);return p

def validate_parameters(p,kind,torch_mode=False):
 if kind not in ['rnn','gru'] or not isinstance(p,dict):raise ValueError('parameter dictionary for rnn/gru required')
 keys={'E','encW','encU','encbi','encbh','decW','decU','decbi','decbh','outW','outb'}
 if set(p)!=keys:raise ValueError('exact model parameter keys required')
 if any(not isinstance(v,torch.Tensor if torch_mode else np.ndarray) for v in p.values()):raise ValueError('all parameters must be arrays of one backend')
 if p['E'].ndim!=2 or p['E'].shape[0]!=8 or p['outW'].ndim!=2 or p['outW'].shape[1]!=8:raise ValueError('embedding/output shapes')
 D=p['E'].shape[1];H=p['outW'].shape[0];G=1 if kind=='rnn' else 3
 if not 1<=D<=64 or not 1<=H<=64:raise ValueError('D/H teaching bounds1..64')
 shapes={'E':(8,D),'outW':(H,8),'outb':(8,)}
 for prefix in ['enc','dec']:shapes.update({prefix+'W':(G*H,D),prefix+'U':(G*H,H),prefix+'bi':(G*H,),prefix+'bh':(G*H,)})
 for k,v in p.items():
  if tuple(v.shape)!=shapes[k]:raise ValueError('parameter shape mismatch:'+k)
  if torch_mode:
   if v.dtype!=torch.float64 or v.device.type!='cpu' or not torch.isfinite(v).all() or torch.abs(v).max()>1e6:raise ValueError('finite CPU float64 parameters with magnitude<=1e6 required')
  elif v.dtype!=np.float64 or not np.isfinite(v).all() or np.abs(v).max()>1e6:raise ValueError('finite NumPy float64 parameters with magnitude<=1e6 required')
 if (bool(torch.count_nonzero(p['E'][0])) if torch_mode else bool(np.count_nonzero(p['E'][0]))):raise ValueError('PAD embedding row must be fixed zero')

def sigmoid(a):
 q=np.exp(-np.abs(a));return np.where(a>=0,1/(1+q),q/(1+q))

def torch_cell(x,h,p,prefix,kind):
 a=x@p[prefix+'W'].T+p[prefix+'bi'];b=h@p[prefix+'U'].T+p[prefix+'bh']
 if kind=='rnn':return torch.tanh(a+b)
 ar,az,an=a.chunk(3,-1);br,bz,bn=b.chunk(3,-1);r=torch.sigmoid(ar+br);z=torch.sigmoid(az+bz);n=torch.tanh(an+r*bn);return (1-z)*n+z*h

def torch_forward(p,kind,sequences,pad_extra=0,zero_context=False):
 validate_parameters(p,kind,True)
 if not isinstance(zero_context,bool):raise ValueError('zero_context bool required')
 src,sm,din,target,dm=batch(sequences,pad_extra);B=len(sequences);H=p['outW'].shape[0];h=torch.zeros((B,H),dtype=torch.float64);E=p['E']
 for t in range(src.shape[1]):
  x=torch.nn.functional.embedding(torch.tensor(src[:,t]),E,padding_idx=0);new=torch_cell(x,h,p,'enc',kind);h=torch.where(torch.tensor(sm[:,t,None]),new,h)
 context=h
 if zero_context:h=h*0
 logits=[]
 for t in range(din.shape[1]):
  x=torch.nn.functional.embedding(torch.tensor(din[:,t]),E,padding_idx=0);new=torch_cell(x,h,p,'dec',kind);h=torch.where(torch.tensor(dm[:,t,None]),new,h);logits.append(h@p['outW']+p['outb'])
 z=torch.stack(logits,1);loss=torch.nn.functional.cross_entropy(z.reshape(-1,8),torch.tensor(target).reshape(-1),ignore_index=0,reduction='sum')/int(dm.sum());return loss,z,context

def numpy_cell(x,h,p,prefix,kind):
 a=x@p[prefix+'W'].T+p[prefix+'bi'];b=h@p[prefix+'U'].T+p[prefix+'bh']
 if kind=='rnn':
  n=np.tanh(a+b);return n,(x,h,n)
 ar,az,an=np.split(a,3,-1);br,bz,bn=np.split(b,3,-1);r=sigmoid(ar+br);z=sigmoid(az+bz);n=np.tanh(an+r*bn);return (1-z)*n+z*h,(x,h,r,z,n,bn)

def numpy_cell_backward(dh,cache,p,prefix,kind):
 if kind=='rnn':
  x,h,n=cache;da=dh*(1-n*n);db=da;direct=np.zeros_like(h)
 else:
  x,h,r,z,n,bn=cache;dn=dh*(1-z);dz=dh*(h-n);da_n=dn*(1-n*n);dr=da_n*bn;da_r=dr*r*(1-r);da_z=dz*z*(1-z);da=np.concatenate([da_r,da_z,da_n],-1);db=np.concatenate([da_r,da_z,da_n*r],-1);direct=dh*z
 g={prefix+'W':da.T@x,prefix+'U':db.T@h,prefix+'bi':da.sum(0),prefix+'bh':db.sum(0)};return da@p[prefix+'W'],db@p[prefix+'U']+direct,g

def numpy_reference(p,kind,sequences,pad_extra=0):
 validate_parameters(p,kind)
 src,sm,din,target,dm=batch(sequences,pad_extra);B=len(sequences);H=p['outW'].shape[0];h=np.zeros((B,H));enc=[];dec=[];zs=[];g={k:np.zeros_like(v,dtype=np.float64) for k,v in p.items()}
 with np.errstate(over='raise',invalid='raise',divide='raise'):
  for t in range(src.shape[1]):
   new,c=numpy_cell(p['E'][src[:,t]],h,p,'enc',kind);h=np.where(sm[:,t,None],new,h);enc.append(c)
  context=h.copy()
  for t in range(din.shape[1]):
   new,c=numpy_cell(p['E'][din[:,t]],h,p,'dec',kind);h=np.where(dm[:,t,None],new,h);dec.append((c,h));zs.append(h@p['outW']+p['outb'])
  z=np.stack(zs,1);shift=z-z.max(-1,keepdims=True);exp=np.exp(shift);prob=exp/exp.sum(-1,keepdims=True);losses=np.log(exp.sum(-1))-np.take_along_axis(shift,target[...,None],-1)[...,0];loss=float(losses[dm].sum()/dm.sum());dz=prob;ii,jj=np.indices(target.shape);dz[ii,jj,target]-=1;dz*=dm[...,None]/dm.sum();dh=np.zeros_like(h)
  for t in reversed(range(din.shape[1])):
   c,ht=dec[t];g['outW']+=ht.T@dz[:,t];g['outb']+=dz[:,t].sum(0);dh+=dz[:,t]@p['outW'].T;active=dm[:,t,None];dx,prev,gg=numpy_cell_backward(dh*active,c,p,'dec',kind);dh=prev+dh*(~active)
   for k,v in gg.items():g[k]+=v
   np.add.at(g['E'],din[:,t],dx)
  for t in reversed(range(src.shape[1])):
   active=sm[:,t,None];dx,prev,gg=numpy_cell_backward(dh*active,enc[t],p,'enc',kind);dh=prev+dh*(~active)
   for k,v in gg.items():g[k]+=v
   np.add.at(g['E'],src[:,t],dx)
  g['E'][PAD]=0;return loss,g,z,context

def greedy(p,kind,sequences,max_steps=10,zero_context=False):
 validate_parameters(p,kind)
 if not isinstance(zero_context,bool):raise ValueError('zero_context bool required')
 if not isinstance(max_steps,Integral) or isinstance(max_steps,(bool,np.bool_)) or not 1<=max_steps<=64:raise ValueError('max_steps integer1..64')
 src,sm,*_=batch(sequences);h=np.zeros((len(sequences),p['outW'].shape[0]));outputs=[[] for _ in sequences];finished=np.zeros(len(sequences),bool)
 for t in range(src.shape[1]):new,_=numpy_cell(p['E'][src[:,t]],h,p,'enc',kind);h=np.where(sm[:,t,None],new,h)
 if zero_context:h[:]=0
 current=np.full(len(sequences),BOS,np.int64)
 for t in range(max_steps):
  new,_=numpy_cell(p['E'][current],h,p,'dec',kind);h=np.where(finished[:,None],h,new);ids=(h@p['outW']+p['outb']).argmax(-1)
  for i,v in enumerate(ids):
   if not finished[i]:outputs[i].append(int(v))
  finished|=(ids==EOS);current=np.where(finished,EOS,ids)
  if finished.all():break
 return outputs

def evaluate(p,kind,sequences):
 loss,g,z,context=numpy_reference(p,kind,sequences);src,sm,din,target,dm=batch(sequences);prediction=z.argmax(-1);shift=z-z.max(-1,keepdims=True);nll=(np.log(np.exp(shift).sum(-1))-np.take_along_axis(shift,target[...,None],-1)[...,0])*dm;decoded=greedy(p,kind,sequences);gold=[list(reversed(s))+[EOS] for s in sequences];correct=[a==b for a,b in zip(decoded,gold)];tfexact=[np.array_equal(prediction[i,dm[i]],target[i,dm[i]]) for i in range(len(sequences))];terminated=[len(s)>0 and s[-1]==EOS for s in decoded];invalid=[any(v in [PAD,BOS,UNK] for v in s) for s in decoded]
 return {'teacher_forced_nll':loss,'teacher_forced_token_accuracy':float((prediction[dm]==target[dm]).mean()),'teacher_forced_sequence_exact':float(np.mean(tfexact)),'free_sequence_exact':float(np.mean(correct)),'free_aligned_token_accuracy':sum(sum(t<len(a) and a[t]==v for t,v in enumerate(b)) for a,b in zip(decoded,gold))/sum(map(len,gold)),'eos_rate':float(np.mean(terminated)),'invalid_control_rate':float(np.mean(invalid)),'per_sequence_nll':nll.sum(1).tolist(),'gold':gold,'decoded':decoded,'correct':correct,'teacher_logits':z.tolist(),'encoder_context':context.tolist()}

def run(output):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);raw=(ROOT/'data/sequences.json').read_bytes();expected=json.loads((ROOT/'data/manifest.json').read_text())['sha256']
 if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('fixed data SHA mismatch')
 data=json.loads(raw);protocol=data['protocol'];runs=[];traces=[];final=[];maxref=0.
 for kind in protocol['models']:
  for seed in protocol['seeds']:
   init=initialize(kind,seed);p={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in init.items()};states=[];clipcount=0
   for step in range(protocol['steps']+1):
    loss,z,c=torch_forward(p,kind,data['train'])
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective; previous output retained')
    state={'step':step,'train_nll':float(loss.detach()),'parameters':{k:v.detach().tolist() for k,v in p.items()}}
    if step==protocol['steps']:states.append(state);break
    loss.backward();gn=math.sqrt(sum(float((v.grad*v.grad).sum()) for v in p.values()))
    if not math.isfinite(gn):raise FloatingPointError('nonfinite gradient')
    factor=min(1.,protocol['clip_global_norm']/gn) if gn else 1.;clipcount+=factor<1;state.update(gradient_norm_before_clip=gn,clip_factor=factor);states.append(state)
    if step in [0,149,299]:
     nl,ng,*_=numpy_reference({k:v.detach().numpy() for k,v in p.items()},kind,data['train']);maxref=max(maxref,abs(nl-float(loss.detach())),*(float(np.abs(ng[k]-v.grad.numpy()).max()) for k,v in p.items()))
    with torch.no_grad():
     for v in p.values():v-=protocol['learning_rate']*factor*v.grad;v.grad=None
   params={k:v.detach().numpy() for k,v in p.items()};metrics={split:evaluate(params,kind,data[split]) for split in ['train','validation','test','long_test']};zero=greedy(params,kind,data['test'],zero_context=True);zeroexact=float(np.mean([a==b for a,b in zip(zero,metrics['test']['gold'])]));runs.append({'kind':kind,'seed':seed,'parameter_coordinates':sum(v.size for v in params.values()),'fixed_pad_coordinates':8,'updates_clipped':clipcount,'metrics':metrics,'zero_context_diagnostic':{'predeclared':'same trained decoder with encoder context zeroed; no retraining, not an equal-capacity model','test_free_exact':zeroexact,'decoded':zero}});traces.append({'kind':kind,'seed':seed,'states':states});final.append({'kind':kind,'seed':seed,'parameters':{k:v.tolist() for k,v in params.items()}})
 summary={kind:{split:{metric:float(np.mean([r['metrics'][split][metric] for r in runs if r['kind']==kind])) for metric in ['teacher_forced_nll','teacher_forced_token_accuracy','teacher_forced_sequence_exact','free_sequence_exact','free_aligned_token_accuracy','eos_rate']} for split in ['test','long_test']} for kind in protocol['models']}
 result={'protocol':protocol,'runs':runs,'summary':summary,'numpy_reference_max_abs_gap':maxref,'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU float64'}};payloads={}
 for name,obj in [('training_traces.json.gz',traces),('final_states.json.gz',final)]:
  b=(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n').encode();content=gzip.compress(b,mtime=0);payloads[name]=content;result[name]={'raw_sha256':hashlib.sha256(b).hexdigest(),'compressed_sha256':hashlib.sha256(content).hexdigest(),'raw_bytes':len(b)}
 payloads['results.json']=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();output=Path(output);output.mkdir(parents=True,exist_ok=True)
 for name,content in payloads.items():
  with tempfile.NamedTemporaryFile(dir=output,delete=False) as f:f.write(content);temp=Path(f.name)
  os.replace(temp,output/name)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');r=run(p.parse_args().output);print(json.dumps({'summary':r['summary'],'reference':r['numpy_reference_max_abs_gap']},indent=2))
