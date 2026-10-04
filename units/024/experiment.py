"""Original NumPy binary/softmax classification teaching experiment, offline CPU."""
from pathlib import Path
import csv,io,json,math,sys
from decimal import Decimal, InvalidOperation
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT={'iterations':400,'learning_rate':0.25,'l2':0.03,'binary_iterations':400}
LOGIT_BOUND=1e6

def real_array(value,ndim,name,bound):
 raw=np.asarray(value)
 if any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):raise ValueError(f'{name}: boolean entries forbidden')
 if raw.dtype.kind not in 'iuf' or raw.ndim!=ndim or 0 in raw.shape:raise ValueError(f'{name}: nonempty numeric array with ndim={ndim} required')
 with np.errstate(over="ignore",under="ignore"):arr=np.asarray(value,dtype=np.float64)
 if np.any((raw!=0)&(arr==0)):raise ValueError(f"{name}: nonzero input lost in float64 conversion")
 if not np.all(np.isfinite(arr)) or np.max(np.abs(arr))>bound:raise ValueError(f'{name}: finite magnitude <= {bound} required')
 return arr

def labels(value,n,k):
 a=np.asarray(value)
 if any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):raise ValueError('boolean labels forbidden')
 if a.shape!=(n,) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=k):raise ValueError('integer label vector with exact shape and range required')
 return a.astype(np.int64)

def scalar(x,lo,hi,name,integer=False):
 if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi or (integer and type(x) is not int):raise ValueError(f'{name}: invalid scalar')
 return x

def sigmoid(z):
 z=real_array(z,1,'binary logits',LOGIT_BOUND);out=np.empty_like(z);positive=z>=0
 out[positive]=1/(1+np.exp(-z[positive]));q=np.exp(z[~positive]);out[~positive]=q/(1+q)
 return out

def binary_loss_gradient(z,y,reduction='mean'):
 z=real_array(z,1,'binary logits',LOGIT_BOUND);y=labels(y,len(z),2)
 if reduction not in ('mean','sum'):raise ValueError('reduction must be mean or sum')
 # Branching avoids cancellation in max(z,0)-y*z and in sigmoid(z)-1.
 signed=np.where(y==1,-z,z)
 losses=np.maximum(signed,0)+np.log1p(np.exp(-np.abs(signed)))
 grad=np.where(y==1,-sigmoid(-z),sigmoid(z));factor=len(z) if reduction=='mean' else 1
 return float(np.sum(losses)/factor),grad/factor

def log_softmax(z):
 z=real_array(z,2,'multiclass logits',LOGIT_BOUND)
 if z.shape[1]<2:raise ValueError('at least two classes required')
 maximum=np.argmax(z,axis=1);shift=z-np.max(z,axis=1,keepdims=True)
 terms=np.exp(shift);terms[np.arange(len(z)),maximum]=0
 return shift-np.log1p(np.sum(terms,axis=1,keepdims=True))

def multiclass_loss_gradient(z,y,reduction='mean'):
 logp=log_softmax(z);n,k=logp.shape;y=labels(y,n,k)
 if reduction not in ('mean','sum'):raise ValueError('reduction must be mean or sum')
 factor=n if reduction=='mean' else 1;p=np.exp(logp);g=p.copy();g[np.arange(n),y]=0
 g[np.arange(n),y]=-np.sum(g,axis=1)
 return float(-np.sum(logp[np.arange(n),y])/factor),g/factor,p

def objective(x,y,w,l2=0,reduction='mean'):
 x=real_array(x,2,'features',100);w=real_array(w,2,'weights',10000)
 if x.shape[1]!=w.shape[0] or w.shape[1]<2:raise ValueError('feature/weight shapes do not align')
 y=labels(y,len(x),w.shape[1]);scalar(l2,0,1,'l2')
 loss,g,p=multiclass_loss_gradient(x@w,y,reduction)
 # Last feature is explicitly the constant-one bias; it is excluded from L2.
 if not np.all(x[:,-1]==1):raise ValueError('last feature must be constant-one bias')
 penalty=w.copy();penalty[-1]=0
 return loss+float(l2*np.sum(penalty*penalty)/2),x.T@g+l2*penalty,p

def fit(x,y,iterations,rate,l2=0):
 x=real_array(x,2,'features',100);y=labels(y,len(x),3);scalar(iterations,0,2000,'iterations',True);scalar(rate,1e-6,5,'learning_rate');scalar(l2,0,1,'l2')
 if x.shape[1]!=3 or not np.all(x[:,-1]==1) or set(y.tolist())!={0,1,2}:raise ValueError('three features including bias and all three classes required')
 w=np.zeros((3,3));history=[]
 for step in range(iterations+1):
  loss,grad,p=objective(x,y,w,l2);history.append({'step':step,'objective':loss,'mean_cross_entropy':float(-np.mean(log_softmax(x@w)[np.arange(len(x)),y])),'gradient_norm':float(np.linalg.norm(grad)),'weight_norm':float(np.linalg.norm(w))})
  if step<iterations:w=w-rate*grad
 return w,history

def binary_path(iterations,rate):
 scalar(iterations,0,2000,'binary_iterations',True);scalar(rate,1e-6,5,'learning_rate');x=np.array([-2.,-1.,1.,2.]);y=np.array([0,0,1,1]);w=0.;rows=[]
 for step in range(iterations+1):
  loss,g=binary_loss_gradient(w*x,y);rows.append({'step':step,'weight':w,'mean_bce':loss,'gradient':float(x@g)})
  if step<iterations:w-=rate*float(x@g)
 return rows

def parsed_float(text):
 try:d=Decimal(text);v=float(text)
 except (InvalidOperation,ValueError,OverflowError) as error:raise ValueError('invalid numeric text') from error
 if d.is_finite() and d!=0 and v==0:raise ValueError('nonzero text lost in float64 conversion')
 return v

def strict_json(path):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON key')
   d[k]=v
  return d
 return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_float=parsed_float)

def read_data(path):
 with Path(path).open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['id','split','x1','x2','y']:raise ValueError('exact CSV columns/order required')
  raw=list(reader)
 if not 3<=len(raw)<=1000:raise ValueError('3..1000 rows required')
 rows=[]
 for r in raw:
  if set(r)!=set(reader.fieldnames) or any(v is None or v=='' for v in r.values()):raise ValueError('complete records required')
  if r['split'] not in ('train','test') or not r['y'].isascii() or not r['y'].isdigit():raise ValueError('invalid split or integer class')
  xx=[parsed_float(r['x1']),parsed_float(r['x2']),1.];real_array([xx],2,'CSV features',100);yy=int(r['y']);labels([yy],1,3);rows.append((r['id'],r['split'],xx,yy))
 if len({r[0] for r in rows})!=len(rows):raise ValueError('unique row IDs required')
 if not all(any(r[1]==s for r in rows) for s in ('train','test')):raise ValueError('both fixed splits required')
 if {r[3] for r in rows if r[1]=='train'}!={0,1,2}:raise ValueError('training requires all three classes')
 return rows

def csv_text(rows):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return f.getvalue()

def run(output,config_path=None,data_path=None):
 config=strict_json(config_path or HERE/'data/config.json')
 if type(config) is not dict or set(config)!=set(DEFAULT):raise ValueError('exact config keys required')
 scalar(config['iterations'],0,2000,'iterations',True);scalar(config['binary_iterations'],0,2000,'binary_iterations',True);scalar(config['learning_rate'],1e-6,5,'learning_rate');scalar(config['l2'],0,1,'l2')
 rows=read_data(data_path or HERE/'data/classification.csv');train=[r for r in rows if r[1]=='train'];test=[r for r in rows if r[1]=='test']
 x=np.array([r[2] for r in train]);y=np.array([r[3] for r in train]);xt=np.array([r[2] for r in test]);yt=np.array([r[3] for r in test])
 w,history=fit(x,y,config['iterations'],config['learning_rate'],config['l2']);binary=binary_path(config['binary_iterations'],config['learning_rate'])
 predictions=[];metrics={}
 for name,group,xx,yy in [('train',train,x,y),('test',test,xt,yt)]:
  loss,g,p=multiclass_loss_gradient(xx@w,yy);pred=np.argmax(p,axis=1);metrics[name]={'rows':len(group),'mean_ce':loss,'correct':int(np.sum(pred==yy)),'accuracy':float(np.mean(pred==yy))}
  for row,prob,guess in zip(group,p,pred):predictions.append({'id':row[0],'split':name,'y':row[3],'prediction':int(guess),'p0':float(prob[0]),'p1':float(prob[1]),'p2':float(prob[2])})
 extremes=[]
 for z,y0 in [(np.array([[1000.,1001.,-1000.]]),2),(np.array([[8.,0.,0.]]),0),(np.array([[0.,0.,0.]]),0)]:
  loss,g,p=multiclass_loss_gradient(z,[y0]);wrong=multiclass_loss_gradient(p,[y0])[0];extremes.append({'logits':z.tolist()[0],'label':y0,'stable_loss':loss,'probability':p.tolist()[0],'gradient':g.tolist()[0],'wrong_probabilities_as_logits_loss':wrong})
 result={'config':config,'input_rows':rows,'weights':w.tolist(),'metrics':metrics,'initial_gradient':objective(x,y,np.zeros((3,3)),0)[1].tolist(),'initial_loss':objective(x,y,np.zeros((3,3)),0)[0],'binary_final':binary[-1],'extreme_cases':extremes,'limits':'Synthetic fixed data only; test never selects parameters; probabilities are not calibrated guarantees.'}
 env={'python':sys.version.split()[0],'numpy':np.__version__,'device':'CPU','data':'synthetic','numerics':'float64; finite bounded logits; tiny probabilities may underflow to zero'}
 texts={'results.json':json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n','training.csv':csv_text(history),'binary_path.csv':csv_text(binary),'predictions.csv':csv_text(predictions),'environment.json':json.dumps(env,indent=2)+'\n'}
 out=Path(output);protected=Path(data_path or HERE/'data/classification.csv').resolve()
 if out.is_symlink() or any((out/n).is_symlink() for n in texts):raise ValueError('output symlinks unsupported')
 if out.resolve()==HERE or out.resolve() in protected.parents:raise ValueError('output overlaps source/input location')
 if out.exists() and not out.is_dir():raise ValueError('output directory required')
 out.mkdir(parents=True,exist_ok=True)
 for name,text in texts.items():(out/name).write_text(text,encoding='utf-8')
 return result

# Populated with deterministic public-input digests after author generation.
TEACHING_INPUT_HASHES={'data/classification.csv': 'ccb89c9f6ae4918e4575a53bdba5e144720483155b7fd361e42a1b6041c4bc76', 'data/config.json': '220fc3fe88b4a0f4b8727ac3e271145220c6f2672e4348453bf7a55dcd5244f3', 'outputs/binary_path.csv': 'e2f3a5ed8d8e9d38f4ac324fdb92d369c06d1d15652bb826d6999652b19a2014', 'outputs/environment.json': '086ff52e62d72a373c1c2bccf318cb2c3508c235004c87aa4aafa5bc372dfa0a', 'outputs/predictions.csv': 'f7dac6c46b601d7a95fc52a4f2ca49f18121da1a504f796e0d43aacbcc35e4b8', 'outputs/results.json': 'e045d557b4caf2e4da8a1215c734353792d0d4983dde425233c6f3c74063911b', 'outputs/training.csv': 'f1b8188137e91a881617e8ea146d53f855fd7933314e0d71533b1415696b11bb'}
def require_teaching_inputs():
 import hashlib
 for relative,sha in TEACHING_INPUT_HASHES.items():
  p=HERE/relative
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('fixed illustrations and Notebook require original teaching inputs and outputs; save custom runs separately')
 if not TEACHING_INPUT_HASHES:raise ValueError('teaching manifest is missing')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'outputs');p.add_argument('--config',type=Path);p.add_argument('--data',type=Path);a=p.parse_args();r=run(a.output,a.config,a.data);print(json.dumps(r['metrics'],indent=2))
