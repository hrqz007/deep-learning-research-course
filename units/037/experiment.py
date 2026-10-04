"""037: explicit dropout trace and isolated regularization ablations, CPU only.
Synthetic data. No download, no test-guided selection, no combined treatment.
"""
from pathlib import Path
import argparse, csv, hashlib, io, itertools, json, math, os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
DTYPE = torch.float64
SEEDS = (370, 371, 372)
CONFIGS = ('baseline', 'early_stop', 'dropout', 'decay', 'augment_valid', 'augment_invalid', 'label_smooth')
PARAM_NAMES = ('W11','W12','W21','W22','b1','b2','u1','u2','c')
HAND_INITIAL = np.array([.5,-.2,.3,.4,.1,.2,.6,-.5,.1])
HAND_X = np.array([[1.,2.],[-1.,1.]])
HAND_Y = np.array([.7,-.2])
HAND_MASK = np.array([[1.,0.],[1.,1.]])
PENALIZED = np.array([1,1,1,1,0,0,1,1,0],dtype=float)
DEFAULT = {'epochs':160,'batch_size':16,'lr':.01,'betas':[.9,.99], 'epsilon':1e-8,
           'width':24,'drop_probability':.3,'decay':.1,'label_smoothing':.1,
           'patience':20,'min_delta':0., 'train_n':48,'validation_n':96,'test_n':384}
DATA_HASHES = {'train.csv': '6f5c7acc5d166b12381c70d17856f61c36b7652bc447ae88fc51b8cbe8a4b345', 'validation.csv': 'e39035f479d309e67352365edb4505369846a74533fd7aa6ab0e1a5d94c928d5', 'test.csv': 'e78dc1e856bccce001fe041e7660a0488a3547c0a345e58069719ab49269b51e'}

def finite_number(value, name, low=None, high=None):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(float(value)):
        raise ValueError(f'{name}: finite number required')
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'{name}: out of range')
    return float(value)

def integer(value, name, low=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<low:
        raise ValueError(f'{name}: integer >= {low} required')
    return int(value)

def numeric_array(value, name, shape=None):
    def has_bool(v):
        if isinstance(v,(bool,np.bool_)):return True
        if isinstance(v,(list,tuple)):return any(has_bool(x) for x in v)
        return isinstance(v,np.ndarray) and v.dtype.kind=='b'
    if has_bool(value):raise ValueError(f'{name}: boolean values are not numeric data')
    a=np.asarray(value)
    # Object/string/bool arrays are not silently coerced into experiment inputs.
    if a.dtype.kind not in 'iuf' or not np.isfinite(a).all(): raise ValueError(f'{name}: finite numeric array required')
    if shape is not None and a.shape != shape: raise ValueError(f'{name}: expected shape {shape}, got {a.shape}')
    return a.astype(float,copy=False)

def validate_batch(x,y):
    x=numeric_array(x,'x'); y=numeric_array(y,'y')
    if x.ndim!=2 or x.shape[1]!=2 or len(x)==0 or y.shape!=(len(x),): raise ValueError('expected nonempty x[n,2], y[n]')
    if not np.isin(y,[0,1]).all(): raise ValueError('hard targets must be 0 or 1')
    return x,y

def read_data(data_dir=None):
    root=Path(data_dir) if data_dir is not None else BASE/'data'
    result={}; ids=set()
    for split,expected_n in [('train',48),('validation',96),('test',384)]:
        path=root/(split+'.csv'); raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=DATA_HASHES[split+'.csv']: raise ValueError(f'input digest mismatch: {split}')
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        if len(rows)!=expected_n: raise ValueError('wrong split size')
        x=[];y=[];clean=[];rowids=[]
        for r in rows:
            if set(r)!={'id','x1','x2','y','clean_y','flipped'}: raise ValueError('wrong data schema')
            i=int(r['id'])
            if i in ids: raise ValueError('overlapping row ID')
            ids.add(i);rowids.append(i);x.append([float(r['x1']),float(r['x2'])]);y.append(float(r['y']));clean.append(int(r['clean_y']))
            if int(r['flipped']) != int(float(r['y'])!=int(r['clean_y'])): raise ValueError('noise audit mismatch')
        x,y=validate_batch(x,y)
        if not np.array_equal(np.array(clean),(x[:,0]>0).astype(int)): raise ValueError('clean-label rule mismatch')
        result[split]={'x':x,'y':y,'clean_y':np.array(clean),'id':np.array(rowids)}
    return result

def hand_forward(theta=HAND_INITIAL, mask=HAND_MASK, q=.5, lam=.1):
    theta=numeric_array(theta,'theta',(9,)); mask=numeric_array(mask,'mask',(2,2))
    q=finite_number(q,'q',1e-12,1);lam=finite_number(lam,'lambda',0)
    if not np.isin(mask,[0,1]).all(): raise ValueError('mask must contain only zero and one')
    w=theta[:4].reshape(2,2);b=theta[4:6];u=theta[6:8];c=theta[8]
    z=HAND_X@w+b;h=np.maximum(z,0);scale=mask/q;hd=h*scale;p=hd@u+c;r=p-HAND_Y
    per=.5*r*r;data=float(per.mean());penalty=float(lam*.5*np.sum(PENALIZED*theta**2))
    dp=r/2;dhd=dp[:,None]*u;dh=dhd*scale;gate=(z>0).astype(float);dz=dh*gate
    # Each row is that sample's contribution to all nine shared parameters.
    paths=[]
    for i in range(2):
        paths.append(np.concatenate([np.outer(HAND_X[i],dz[i]).ravel(),dz[i],dp[i]*hd[i],[dp[i]]]))
    paths=np.asarray(paths);reg=lam*PENALIZED*theta;g=paths.sum(0)+reg
    return {'theta':theta.tolist(),'mask':mask.tolist(),'q':q,'lambda':lam,'z':z.tolist(),'h':h.tolist(),'scale':scale.tolist(),
            'h_dropout':hd.tolist(),'prediction':p.tolist(),'residual':r.tolist(),'per_sample_half_square':per.tolist(),
            'data_loss':data,'penalty':penalty,'objective':data+penalty,'d_prediction':dp.tolist(),'d_h_dropout':dhd.tolist(),
            'd_h':dh.tolist(),'relu_gate':gate.tolist(),'d_z':dz.tolist(),'input_gradient':(dz@w.T).tolist(),
            'sample_parameter_paths':paths.tolist(),'data_gradient':paths.sum(0).tolist(),'penalty_gradient':reg.tolist(),'gradient':g.tolist()}

def hand_torch(theta=HAND_INITIAL,mask=HAND_MASK,q=.5,lam=.1):
    # Explicit validation stays effective under python -O.
    hand_forward(theta,mask,q,lam)
    t=torch.tensor(theta,dtype=DTYPE,requires_grad=True);x=torch.tensor(HAND_X,dtype=DTYPE,requires_grad=True)
    h=F.relu(x@t[:4].reshape(2,2)+t[4:6]);p=(h*torch.tensor(mask,dtype=DTYPE)/q)@t[6:8]+t[8]
    loss=((p-torch.tensor(HAND_Y))**2).mean()/2+lam*(t*t*torch.tensor(PENALIZED)).sum()/2
    loss.backward();return float(loss.detach()),t.grad.numpy().copy(),x.grad.numpy().copy()

def hand_trace():
    first=hand_forward();new=HAND_INITIAL-.05*np.array(first['gradient']);second=hand_forward(new)
    newmask=np.array([[1.,1.],[0.,1.]])
    return {'parameter_order':PARAM_NAMES,'learning_rate':.05,'initial':first,'updated_theta':new.tolist(),'same_mask_next':second,
            'fresh_mask_next':hand_forward(new,newmask),'eval_next':hand_forward(new,np.ones((2,2)),q=1)}

def enumerate_masks():
    rows=[]
    for bits in itertools.product((0.,1.),repeat=4):
        h=hand_forward(mask=np.array(bits).reshape(2,2))
        p=np.array(h['prediction']); rows.append({'mask':list(bits),'probability':1/16,'prediction':p.tolist(),
            'sigmoid_prediction':(1/(1+np.exp(-p))).tolist(),'data_loss':h['data_loss'],'gradient':h['gradient']})
    clean=hand_forward(mask=np.ones((2,2)),q=1)
    return {'rows':rows,'mean_prediction':np.mean([r['prediction'] for r in rows],0).tolist(),
            'mean_sigmoid_prediction':np.mean([r['sigmoid_prediction'] for r in rows],0).tolist(),
            'mean_data_loss':float(np.mean([r['data_loss'] for r in rows])),
            'mean_gradient':np.mean([r['gradient'] for r in rows],0).tolist(),
            'no_dropout':clean,'variance_term':float(.5*np.mean(np.sum((np.array(clean['h'])*HAND_INITIAL[6:8])**2,axis=1)))}

def mode_probe():
    rows=[]
    for training in (True,False):
        for grad in (True,False):
            torch.manual_seed(37001);module=nn.Dropout(.5);module.train(training);x=torch.ones(2,6,dtype=DTYPE,requires_grad=True)
            with torch.set_grad_enabled(grad):a=module(x);b=module(x)
            derivative=None
            if grad:a.sum().backward();derivative=x.grad.tolist()
            rows.append({'training':training,'grad_enabled':grad,'first':a.detach().tolist(),'second':b.detach().tolist(),
                         'requires_grad':a.requires_grad,'input_gradient':derivative,'has_running_buffers':bool(list(module.buffers()))})
    logits=torch.tensor([[0.,2.],[1.,-.5]],dtype=DTYPE);ys=torch.tensor([1,0]);alpha=.1
    ce=float(F.cross_entropy(logits,ys,label_smoothing=alpha));target=(1-alpha)*F.one_hot(ys,2).to(DTYPE)+alpha/2
    manual=float(-(target*F.log_softmax(logits,dim=1)).sum(1).mean())
    return {'dropout_modes':rows,'label_smoothing_framework':{'torch_ce':ce,'manual_ce':manual,'targets':target.tolist()}}

class Network(nn.Module):
    """2 -> 24 tanh -> observed dropout -> 24 tanh -> one logit. No BN."""
    def __init__(self,seed=370,p=0.,width=24):
        super().__init__();seed=integer(seed,'seed',0);width=integer(width,'width');p=finite_number(p,'p',0,1)
        if p==1:raise ValueError('this teaching model requires p<1')
        self.p=p;self.width=width;rng=np.random.default_rng(seed+20000)
        shapes=((2,width),(width,),(width,width),(width,),(width,1),(1,))
        vals=[]
        for i,s in enumerate(shapes):
            vals.append(rng.normal(0,math.sqrt(2/sum(s)),size=s) if len(s)==2 else np.zeros(s))
        self.params=nn.ParameterList([nn.Parameter(torch.tensor(v,dtype=DTYPE)) for v in vals])
    def forward(self,x,mask=None):
        w,b,v,d,u,c=self.params;h=torch.tanh(x@w+b)
        if self.training and self.p>0:
            if mask is None or mask.shape!=h.shape:raise ValueError('training dropout requires a same-shape explicit mask')
            h=h*mask/(1-self.p)
        elif mask is not None:raise ValueError('mask supplied when dropout is inactive')
        return (torch.tanh(h@v+d)@u+c).squeeze(1)

def eval_metrics(model,x,y):
    was=model.training;model.eval()
    with torch.no_grad():
        logit=model(torch.tensor(x,dtype=DTYPE));target=torch.tensor(y,dtype=DTYPE);prob=logit.sigmoid()
        result={'ce':float(F.binary_cross_entropy_with_logits(logit,target)), 'error':float(((prob>=.5)!=target.bool()).double().mean()),
                'brier':float(((prob-target)**2).mean())}
        p=prob.numpy().copy()
    model.train(was);return result,p

def augment(x,kind,rng):
    x=numeric_array(x,'augmentation x')
    if x.ndim!=2 or x.shape[1]!=2:raise ValueError('augmentation requires x[n,2]')
    if kind not in ('none','valid','invalid'):raise ValueError('unknown augmentation')
    z=x.copy();selected=np.zeros(len(x),dtype=bool)
    if kind!='none':
        selected=rng.random(len(x))<.5
        if kind=='valid':z[selected,1]=-z[selected,1]
        else:z[selected,0]=-z[selected,0]
    return z,selected

def train_one(train,validation,config,seed,epochs=160,patience=20):
    if config not in CONFIGS:raise ValueError('unknown configuration')
    seed=integer(seed,'seed',0);epochs=integer(epochs,'epochs');patience=integer(patience,'patience')
    x,y=validate_batch(train['x'],train['y']);vx,vy=validate_batch(validation['x'],validation['y'])
    if len(x)%16:raise ValueError('training n must be divisible by 16')
    net=Network(seed,p=.3 if config=='dropout' else 0)
    decay=.1 if config=='decay' else 0;alpha=.1 if config=='label_smooth' else 0
    opt=torch.optim.AdamW([{'params':[net.params[i] for i in (0,2,4)],'weight_decay':decay},
                           {'params':[net.params[i] for i in (1,3,5)],'weight_decay':0}],lr=.01,betas=(.9,.99),eps=1e-8,foreach=False,fused=False)
    order=np.random.default_rng(seed+10000);drop=np.random.default_rng(seed+30000);aug=np.random.default_rng(seed+40000)
    history=[];best_value=math.inf;best_epoch=None;best_state=None;wait=0;updates=0;selected_total=0;semantic_total=0
    def record(epoch,training_objective=None):
        tr,_=eval_metrics(net,x,y);va,_=eval_metrics(net,vx,vy)
        row={'config':config,'seed':seed,'epoch':epoch,'updates':updates,'sample_presentations':updates*16,
             'train_ce':tr['ce'],'validation_ce':va['ce'],'train_error':tr['error'],'validation_error':va['error'],
             'stochastic_training_objective': '' if training_objective is None else training_objective}
        history.append(row);return va['ce']
    record(0)
    for epoch in range(1,epochs+1):
        perm=order.permutation(len(x));sum_loss=0
        for start in range(0,len(x),16):
            ix=perm[start:start+16];xb=x[ix];yb=y[ix]
            kind='valid' if config=='augment_valid' else 'invalid' if config=='augment_invalid' else 'none'
            xa,chosen=augment(xb,kind,aug);selected_total+=int(chosen.sum())
            semantic_total+=int(np.sum((xa[:,0]>0)!=(xb[:,0]>0)))
            mask=None
            if net.p>0:mask=torch.tensor((drop.random((len(ix),24))>=net.p).astype(float),dtype=DTYPE)
            target=torch.tensor((1-alpha)*yb+alpha/2,dtype=DTYPE)
            net.train();opt.zero_grad(set_to_none=True);logit=net(torch.tensor(xa,dtype=DTYPE),mask)
            loss=F.binary_cross_entropy_with_logits(logit,target)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training objective')
            loss.backward();opt.step();updates+=1;sum_loss+=float(loss.detach())*len(ix)
        val=record(epoch,sum_loss/len(x))
        if config=='early_stop':
            if val<best_value:
                best_value=val;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in net.state_dict().items()};wait=0
            else:wait+=1
            if wait>=patience:break
    actual_epochs=epoch
    if config=='early_stop':net.load_state_dict(best_state);used_epoch=best_epoch
    else:used_epoch=actual_epochs
    tr,_=eval_metrics(net,x,y);va,_=eval_metrics(net,vx,vy)
    detail={'config':config,'seed':seed,'epochs_run':actual_epochs,'selected_epoch':used_epoch,'updates':updates,'sample_presentations':updates*16,
            'early_stop_triggered':config=='early_stop' and wait>=patience,'selected_augmentation_count':selected_total,
            'semantic_change_count':semantic_total,'parameters':[v.detach().tolist() for v in net.params],
            'train':tr,'validation':va,'validation_evaluations':actual_epochs+2,
            'weight_norm':float(torch.sqrt(sum(net.params[i].detach().square().sum() for i in (0,2,4))))}
    return net,history,detail

def json_bytes(obj):return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def csv_bytes(rows):
    if not rows:raise ValueError('cannot serialize empty table')
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return out.getvalue().encode()

def run(output=None,data_dir=None,epochs=160):
    epochs=integer(epochs,'epochs');data=read_data(data_dir)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    calculation={'hand':hand_trace(),'enumeration':enumerate_masks(),'mode_probe':mode_probe()}
    histories=[];details=[];nets={}
    # Test arrays are deliberately absent from this entire fitting/selection call.
    for config in CONFIGS:
        for seed in SEEDS:
            net,h,d=train_one(data['train'],data['validation'],config,seed,epochs)
            histories.extend(h);details.append(d);nets[(config,seed)]=net
    # All parameters / early-stop choices now frozen. Evaluate every prespecified arm.
    predictions=[];summary=[]
    for d in details:
        key=(d['config'],d['seed']);score,p=eval_metrics(nets[key],data['test']['x'],data['test']['y'])
        d['test']=score
        summary.append({k:d[k] for k in ('config','seed','epochs_run','selected_epoch','updates','sample_presentations','weight_norm','selected_augmentation_count','semantic_change_count')}|
                       {'train_ce':d['train']['ce'],'validation_ce':d['validation']['ce'],'test_ce':score['ce'],'test_error':score['error'],'test_brier':score['brier']})
        for rid,y,prob in zip(data['test']['id'],data['test']['y'],p):
            predictions.append({'config':key[0],'seed':key[1],'id':int(rid),'y':int(y),'probability':float(prob)})
    axis=np.linspace(-1,1,51);xx,yy=np.meshgrid(axis,axis);gx=np.column_stack((xx.ravel(),yy.ravel()))
    grids={'axis':axis.tolist(),'seed':370,'probabilities':{}}
    for config in CONFIGS:
        _,p=eval_metrics(nets[(config,370)],gx,np.zeros(len(gx)));grids['probabilities'][config]=p.reshape(51,51).tolist()
    protocol={**DEFAULT,'epochs':epochs,'seeds':list(SEEDS),'configurations':list(CONFIGS),'dtype':'float64','device':'cpu',
              'test_access':'after all prespecified fits and validation-only checkpoint choices','search':'none; one fixed setting per isolated arm',
              'seed_scope':'joint initialization/order/dropout/augmentation streams; data fixed', 'data_hashes':DATA_HASHES,
              'python':__import__('platform').python_version(),'torch':torch.__version__,'numpy':np.__version__}
    payload={'calculations.json':json_bytes(calculation),'training-history.csv':csv_bytes(histories),'ablation.csv':csv_bytes(summary),
             'test-predictions.csv':csv_bytes(predictions),'run-details.json':json_bytes({'protocol':protocol,'runs':details}),'decision-grid.json':json_bytes(grids)}
    # Validate/compute/serialize everything before creating or replacing any output.
    out=Path(output) if output is not None else BASE/'outputs';out.mkdir(parents=True,exist_ok=True)
    for name,raw in payload.items():
        temp=out/(name+'.tmp');temp.write_bytes(raw);os.replace(temp,out/name)
    return {'runs':len(details),'updates':sum(d['updates'] for d in details),'history_rows':len(histories),
            'test_prediction_rows':len(predictions),'hashes':{k:hashlib.sha256(v).hexdigest() for k,v in payload.items()}}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--data-dir',type=Path);p.add_argument('--epochs',type=int,default=160)
    args=p.parse_args();print(json.dumps(run(args.output,args.data_dir,args.epochs),ensure_ascii=False,indent=2))
