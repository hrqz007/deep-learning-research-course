"""042: one auditable project with entity-aware risk, matched MLP budgets and strong baselines.
CPU float64. Selection accepts no test data. Serialize all payloads before any write.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import argparse,csv,gzip,hashlib,io,json,math,os,platform,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
FIXTURES={'protocol.json':'32062805bbbec66cc2d1dde82a242fb9abab70eaa7b2d6e3062bf51efc8a5420','test.csv':'14794b91105620fbd6519df4dbb4477e5c39e68e1a39d58fdc8316741bf93292','train.csv':'6528f8a4eaff59dd33b77292cfe3ce2a0039e90b86d49871990e6556684943e9','validation.csv':'36ef9b9efdb692ffe6e25df46029cd116c5eb0ce7f33f0e1d830706b6ec6c6eb'}
OUTPUT_NAMES=('results.json','hand_chain.json','final_evaluation.json','identity_leakage.json','training_curves.csv','training_traces.json.gz')

def require(ok,message):
    if not ok:raise ValueError(message)
def integer(x,name,low,high):
    require(not isinstance(x,(bool,np.bool_)) and isinstance(x,(int,np.integer)) and low<=x<=high,'invalid '+name)
    return int(x)
def real(x,name,low,high):
    require(not isinstance(x,(bool,np.bool_)) and isinstance(x,(int,float,np.integer,np.floating)),'real scalar required: '+name)
    with np.errstate(over='ignore',invalid='ignore'):v=float(x)
    require(math.isfinite(v) and low<=v<=high,'invalid '+name)
    return v
def array(x,name,ndim,limit=1e6):
    raw=np.asarray(x,dtype=object)
    require(raw.ndim==ndim and raw.size>0,'nonempty shape required: '+name)
    require(all(not isinstance(v,(bool,np.bool_)) and isinstance(v,(int,float,np.integer,np.floating)) for v in raw.flat),'real numeric values required: '+name)
    with np.errstate(over='ignore',invalid='ignore'):a=np.asarray(x,dtype=np.float64)
    require(np.isfinite(a).all() and np.max(np.abs(a))<=limit,'nonfinite or outside numeric domain: '+name)
    return a

def weights(groups,mode):
    require(mode in ('row','entity'),'unknown weighting mode')
    require(isinstance(groups,(list,tuple)) and len(groups)>0 and all(isinstance(g,str) and g for g in groups),'nonempty string entity IDs required')
    if mode=='row':return np.full(len(groups),1/len(groups))
    c=Counter(groups);return np.array([1/(len(c)*c[g]) for g in groups])

def validate_split(d):
    require(isinstance(d,dict) and all(k in d for k in ('x','y','groups')),'split keys')
    x=array(d['x'],'x',2);y=array(d['y'],'y',1);q=weights(d['groups'],'entity')
    require(x.shape==(len(y),2) and len(q)==len(y),'split paired shapes')
    return x,y,q

def verify_inputs():
    for name,digest in FIXTURES.items():require(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest,'modified fixture '+name)
    return json.loads((ROOT/'data/protocol.json').read_text())

def load_split(name):
    require(name in ('train','validation','test'),'unknown split')
    with (ROOT/'data'/f'{name}.csv').open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f);require(r.fieldnames==['row_id','entity_id','replicate','x1','x2','y','region'],'CSV schema');rows=list(r)
    d={'x':[[float(r['x1']),float(r['x2'])] for r in rows],'y':[float(r['y']) for r in rows],'groups':[r['entity_id'] for r in rows],'ids':[r['row_id'] for r in rows],'regions':[r['region'] for r in rows]}
    d['x'],d['y'],_=validate_split(d)
    require(len(set(d['groups']))=={'train':40,'validation':32,'test':160}[name],'entity count')
    require(len(set(d['ids']))==len(rows) and all(g.startswith(name+'-') for g in d['groups']),'unique split IDs')
    for g in set(d['groups']):
        rs=[r for r in rows if r['entity_id']==g];region=rs[0]['region'];require(region in ('left','right') and all(r['region']==region for r in rs),'region consistency')
        require(len(rs)==(8 if region=='left' else 2) and sorted(int(r['replicate']) for r in rs)==list(range(len(rs))),'replicate IDs')
    return d

def normalization(train):
    x,_,q=validate_split(train);mu=q@x;sd=np.sqrt(q@((x-mu)**2))
    require(np.isfinite(sd).all() and (sd>1e-12).all(),'constant or near-constant training feature')
    return {'mean':mu.tolist(),'std':sd.tolist(),'fitted_on':'training entities only'}

def transform(x,norm):return (x-np.asarray(norm['mean']))/np.asarray(norm['std'])
def initialize(seed,width=6):
    seed=integer(seed,'seed',0,2**31-1);width=integer(width,'width',1,32);r=np.random.default_rng(seed)
    return np.r_[r.normal(0,1/math.sqrt(2),(2,width)).ravel(),np.zeros(width),r.normal(0,1/math.sqrt(width),width),0.]

def forward(theta,x,width=6):
    """Internal differentiable kernel. Call fit for shape/domain-checked public training."""
    h=torch.tanh(x@theta[:2*width].reshape(2,width)+theta[2*width:3*width])
    return h@theta[3*width:4*width]+theta[-1],h

def numpy_forward(theta,x,width=6):return np.tanh(x@theta[:2*width].reshape(2,width)+theta[2*width:3*width])@theta[3*width:4*width]+theta[-1]
def penalty(theta,width=6):return (theta[:2*width].square().sum()+theta[3*width:4*width].square().sum())/2

def numpy_gradient(theta,x,y,q,l2,width=6):
    """Independent algebraic gradient, used for readable checks; NumPy arrays are internal."""
    w=theta[:2*width].reshape(2,width);b=theta[2*width:3*width];v=theta[3*width:4*width]
    h=np.tanh(x@w+b);pred=h@v+theta[-1];u=q*(pred-y);dz=u[:,None]*v*(1-h*h)
    return np.r_[(x.T@dz+l2*w).ravel(),dz.sum(0),h.T@u+l2*v,u.sum()]

def fit(train,validation,mode,lr,seed,steps=350,l2=.002,width=6):
    x,y,_=validate_split(train);vx,vy,vq=validate_split(validation);q=weights(train['groups'],mode)
    lr=real(lr,'learning rate',1e-8,1.);l2=real(l2,'L2',0,1.);steps=integer(steps,'steps',1,1000);width=integer(width,'width',1,32)
    norm=normalization(train);xs=transform(x,norm);vxs=transform(vx,norm)
    tx,ty,tq,tvx,tvy,tvq=[torch.tensor(v,dtype=torch.float64) for v in (xs,y,q,vxs,vy,vq)]
    theta=torch.tensor(initialize(seed,width),dtype=torch.float64,requires_grad=True);states=[];curve=[]
    for step in range(steps+1):
        pred,_=forward(theta,tx,width);data=(tq*(pred-ty).square()).sum()/2;objective=data+l2*penalty(theta,width)
        g=torch.autograd.grad(objective,theta)[0]
        with torch.no_grad():
            vp,_=forward(theta,tvx,width);entity_val=(tvq*(vp-tvy).square()).sum()/2
            entity_train=float(np.sum(weights(train['groups'],'entity')*(pred.detach().numpy()-y)**2)/2)
        row={'step':step,'train_objective':float(objective.detach()),'train_weighted_half_mse':float(data.detach()),'train_entity_half_mse':entity_train,'validation_entity_half_mse':float(entity_val),'gradient_norm':float(torch.linalg.vector_norm(g))}
        require(all(math.isfinite(v) for v in row.values()) and torch.isfinite(theta).all(),'nonfinite training state')
        states.append(theta.detach().tolist());curve.append(row)
        if step<steps:
            candidate=theta.detach()-lr*g
            require(torch.isfinite(candidate).all() and float(candidate.abs().max())<=1e6,'candidate outside numeric domain; run aborted before writing')
            theta=candidate.requires_grad_()
    return {'mode':mode,'lr':lr,'seed':seed,'steps':steps,'l2':l2,'width':width,'normalization':norm,'parameters':states,'curve':curve,'final_parameters':states[-1],'row_presentations':steps*len(y),'validation_prediction':numpy_forward(np.asarray(states[-1]),vxs,width).tolist()}

def select_networks(train,validation,plan):
    runs=[];scores=[];selected={}
    for mode in plan['objective_modes']:
        for lr in plan['learning_rates']:
            group=[fit(train,validation,mode,lr,seed,plan['steps'],plan['network']['l2_weights_only'],plan['network']['width']) for seed in plan['seeds']]
            runs.extend(group);scores.append({'mode':mode,'lr':lr,'mean_validation_entity_half_mse':float(np.mean([r['curve'][-1]['validation_entity_half_mse'] for r in group])),'seed_losses':[r['curve'][-1]['validation_entity_half_mse'] for r in group]})
        candidates=[s for s in scores if s['mode']==mode];selected[mode]=min(enumerate(candidates),key=lambda pair:(pair[1]['mean_validation_entity_half_mse'],pair[0]))[1]['lr']
    return {'runs':runs,'scores':scores,'selected':selected,'test_used':False}

def design(x,degree):
    integer(degree,'degree',1,3);require(degree in (1,3),'degree must be 1 or 3')
    powers=[(a,d-a) for d in range(degree+1) for a in range(d,-1,-1)]
    return np.column_stack([x[:,0]**a*x[:,1]**b for a,b in powers]),powers

def ridge(train,validation,degree,l2,norm):
    x,y,q=validate_split(train);vx,vy,vq=validate_split(validation);l2=real(l2,'ridge penalty',1e-8,1.)
    X,powers=design(transform(x,norm),degree);V,_=design(transform(vx,norm),degree)
    penalty_matrix=np.eye(X.shape[1]);penalty_matrix[0,0]=0
    A=np.vstack([np.sqrt(q)[:,None]*X,math.sqrt(l2)*penalty_matrix]);b=np.r_[np.sqrt(q)*y,np.zeros(X.shape[1])]
    beta=np.linalg.lstsq(A,b,rcond=None)[0];pred=V@beta
    return {'degree':degree,'l2':l2,'powers':powers,'coefficients':beta.tolist(),'normalization':norm,'validation_entity_half_mse':float(vq@((pred-vy)**2)/2),'validation_prediction':pred.tolist(),'augmented_shape':list(A.shape)}

def select_baselines(train,validation,plan):
    norm=normalization(train);candidates=[ridge(train,validation,3,l,norm) for l in plan['strong_baseline']['ridge_lambdas']]
    chosen=min(range(len(candidates)),key=lambda i:(candidates[i]['validation_entity_half_mse'],i))
    return {'constant':float(weights(train['groups'],'entity')@train['y']),'linear':ridge(train,validation,1,.001,norm),'polynomial_candidates':candidates,'selected_polynomial_index':chosen,'test_used':False}

def entity_losses(pred,y,groups):
    pred=array(pred,'prediction',1);y=array(y,'labels',1);weights(groups,'entity');require(len(pred)==len(y)==len(groups),'loss paired shapes')
    names=list(dict.fromkeys(groups));loss=(pred-y)**2/2
    return names,np.array([float(np.mean(loss[np.array(groups)==g])) for g in names])

def paired_bootstrap(differences,replicates=2000,seed=4291):
    d=array(differences,'entity differences',1,1e12);replicates=integer(replicates,'replicates',10,10000);seed=integer(seed,'seed',0,2**31-1)
    rng=np.random.default_rng(seed);means=d[rng.integers(0,len(d),(replicates,len(d)))].mean(1)
    return {'mean':float(d.mean()),'percentile_95':np.quantile(means,[.025,.975]).tolist(),'bootstrap_means':means.tolist(),'differences':d.tolist(),'seed':seed,'replicates':replicates,'unit':'independent test entity; all repeats stay together','conditional_on':'fixed training/selection/fitted models; no retraining, no retuning'}

def final_evaluate(test,selection,baselines,plan):
    x,y,q=validate_split(test);predictions={};seed_metrics=[]
    for mode,lr in selection['selected'].items():
        runs=[r for r in selection['runs'] if r['mode']==mode and r['lr']==lr]
        ps=[numpy_forward(np.asarray(r['final_parameters']),transform(x,r['normalization']),r['width']) for r in runs]
        predictions[mode]=np.mean(ps,0)
        for r,p in zip(runs,ps):seed_metrics.append({'mode':mode,'seed':r['seed'],'entity_half_mse':float(q@((p-y)**2)/2),'prediction':p.tolist()})
    predictions['constant']=np.full(len(y),baselines['constant'])
    for name,model in [('linear',baselines['linear']),('polynomial',baselines['polynomial_candidates'][baselines['selected_polynomial_index']])]:
        X,_=design(transform(x,model['normalization']),model['degree']);predictions[name]=X@np.array(model['coefficients'])
    metrics=[];per_entity={};names=None
    for name,p in predictions.items():
        ids,ls=entity_losses(p,y,test['groups']);names=ids;per_entity[name]=ls.tolist()
        for region in ('all','left','right'):
            mask=np.ones(len(y),bool) if region=='all' else np.array(test['regions'])==region
            gs=[g for g,m in zip(test['groups'],mask) if m];rr=p[mask]-y[mask];w=weights(gs,'entity')
            loss=float(w@(rr*rr)/2);metrics.append({'model':name,'region':region,'entities':len(set(gs)),'rows':int(mask.sum()),'entity_half_mse':loss,'entity_rmse':math.sqrt(2*loss),'row_half_mse':float(np.mean(rr*rr)/2)})
    differences=np.array(per_entity['entity'])-np.array(per_entity['row'])
    return {'test_row_ids':test['ids'],'test_entity_ids':names,'predictions':{k:v.tolist() for k,v in predictions.items()},'seed_metrics':seed_metrics,'metrics':metrics,'per_entity_half_mse':per_entity,'primary_paired_bootstrap':paired_bootstrap(differences,plan['bootstrap']['replicates'],plan['bootstrap']['seed']),'comparison_locked_before_test':True,'post_test_selection':False}

def hand_step(theta):
    q=[F(1,4),F(1,4),F(1,2)];rows=[];contributions=[]
    for x,y,weight in zip([F(-1),F(0),F(1)],[F(0),F(1),F(0)],q):
        z=[theta[j]*x+theta[2+j] for j in range(2)];h=[max(F(0),v) for v in z];gate=[F(int(v>0)) for v in z]
        pred=sum(theta[4+j]*h[j] for j in range(2))+theta[-1];r=pred-y;u=weight*r;dz=[u*theta[4+j]*gate[j] for j in range(2)]
        cg=[dz[0]*x,dz[1]*x,*dz,u*h[0],u*h[1],u];contributions.append(cg)
        rows.append({'x':x,'y':y,'weight':weight,'z':z,'h':h,'gate':gate,'prediction':pred,'residual':r,'local_loss_derivative':r,'weighted_upstream':u,'hidden_upstream':[u*theta[4+j] for j in range(2)],'dz':dz,'half_squared_error':r*r/2,'weighted_loss':weight*r*r/2,'contribution':cg})
    data_grad=[sum(c[j] for c in contributions) for j in range(7)];reg_grad=[theta[j]/10 if j in (0,1,4,5) else F(0) for j in range(7)]
    grad=[a+b for a,b in zip(data_grad,reg_grad)];after=[a-g/10 for a,g in zip(theta,grad)]
    data=sum(r['weighted_loss'] for r in rows);reg=sum(theta[j]**2 for j in (0,1,4,5))/20
    return {'theta':theta,'rows':rows,'data_loss':data,'regularizer':reg,'objective':data+reg,'data_gradient':data_grad,'regularizer_gradient':reg_grad,'gradient':grad,'after':after}

def hand_trace():
    t=[F(1,2),F(-1,2),F(1),F(1),F(1),F(-1,2),F(0)];steps=[]
    for _ in range(2):s=hand_step(t);steps.append(s);t=s['after']
    third=hand_step(t)
    return {'parameter_order':['w1','w2','b1','b2','a1','a2','c'],'sample_entities':['A','A','B'],'learning_rate':F(1,10),'lambda':F(1,10),'steps':steps,'third_forward':third['rows'],'third_data_loss':third['data_loss'],'third_objective':third['objective'],'third_parameters':t}

def identity_leakage():
    rng=np.random.default_rng(4280);offset=rng.standard_normal(80);y=offset[:,None]+.1*rng.standard_normal((80,4))
    seen_pred=y[:,:3].mean(1);seen=(seen_pred-y[:,3])**2/2
    unseen_pred=np.full(20,y[:60].mean());unseen=(unseen_pred[:,None]-y[60:])**2/2
    # Closed-form variance contrast for the same zero-mean model with independent b and epsilon.
    return {'seed':4280,'entity_effect_variance':1.,'measurement_noise_variance':.01,'labels':y.tolist(),'seen_entity_predictions':seen_pred.tolist(),'seen_entity_losses':seen.tolist(),'unseen_entity_predictions':unseen_pred.tolist(),'unseen_entity_losses':unseen.mean(1).tolist(),'seen_empirical_half_mse':float(seen.mean()),'unseen_empirical_half_mse':float(unseen.mean()),'seen_theoretical_half_mse':.01*(1+1/3)/2,'unseen_theoretical_half_mse':(1+.01+(1+.01/4)/60)/2,'limits':'Different prediction tasks and different test entity sets; descriptive leakage mechanism, not a paired MLP comparison or proof about all splits.'}

def serial(v):
    if isinstance(v,F):return {'fraction':str(v),'float':float(v)}
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return v

def json_bytes(v):return (json.dumps(serial(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def gzip_bytes(raw):
    b=io.BytesIO()
    with gzip.GzipFile(fileobj=b,mode='wb',filename='',mtime=0,compresslevel=9) as f:f.write(raw)
    return b.getvalue()
def csv_bytes(rows):
    b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return b.getvalue().encode()
def atomic_write(p,b):
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.'+p.name,delete=False) as f:f.write(b);temp=Path(f.name)
        os.replace(temp,p)
    finally:
        if temp is not None and temp.exists():temp.unlink()

def run(output=None):
    plan=verify_inputs();out=Path(output).resolve() if output is not None else ROOT/'outputs'
    require(not out.exists() or out.is_dir(),'output must be directory')
    require(out not in (ROOT,ROOT/'data',ROOT/'figures'),'protected output directory')
    for name in OUTPUT_NAMES:require(not (out/name).exists() or (out/name).is_file(),'output target is not a file')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train=load_split('train');validation=load_split('validation');require(not(set(train['groups'])&set(validation['groups'])),'entity overlap')
    access=['checked fixture hashes without using test labels for decisions','parsed train and validation','fit one training-only entity-weighted normalization for both MLP modes']
    selection=select_networks(train,validation,plan);baselines=select_baselines(train,validation,plan);access.append('froze both MLP rates and polynomial regularization using validation only')
    test=load_split('test');require(not(set(test['groups'])&(set(train['groups'])|set(validation['groups']))),'test entity overlap');access.append('parsed test only after all choices; final fixed predictions and entity-paired interval')
    final=final_evaluate(test,selection,baselines,plan);runs=selection['runs'];curves=[dict(mode=r['mode'],lr=r['lr'],seed=r['seed'],**v) for r in runs for v in r['curve']]
    summary={'unit':'042','environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'dtype':'float64','device':'cpu'},'protocol':plan,'access_ledger':access,'split_summary':{name:{'rows':len(d['y']),'entities':len(set(d['groups'])),'left_entities':len(set(g for g,r in zip(d['groups'],d['regions']) if r=='left')),'row_left_mass':float(np.mean(np.array(d['regions'])=='left')),'entity_left_mass':float(weights(d['groups'],'entity')@(np.array(d['regions'])=='left'))} for name,d in [('train',train),('validation',validation),('test',test)]},'network_scores':selection['scores'],'selected_learning_rates':selection['selected'],'baselines':baselines,'counts':{'neural_fits':len(runs),'updates':sum(r['steps'] for r in runs),'parameter_states':sum(len(r['parameters']) for r in runs),'parameter_coordinates':sum(len(r['parameters'])*len(r['parameters'][0]) for r in runs),'neural_parameters_per_model':25,'row_presentations_per_mode':sum(r['row_presentations'] for r in runs if r['mode']=='row'),'polynomial_fits':4,'linear_fits':1,'constant_fits':1},'final_summary':[m for m in final['metrics'] if m['region']=='all'],'primary_interval':{k:v for k,v in final['primary_paired_bootstrap'].items() if k not in ('bootstrap_means','differences')}}
    raw=json_bytes({'runs':runs});summary['trace_storage']={'format':'lossless gzip','mtime':0,'filename':'','raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest()}
    payload={'results.json':json_bytes(summary),'hand_chain.json':json_bytes(hand_trace()),'final_evaluation.json':json_bytes(final),'identity_leakage.json':json_bytes(identity_leakage()),'training_curves.csv':csv_bytes(curves),'training_traces.json.gz':gzip_bytes(raw)}
    out.mkdir(parents=True,exist_ok=True)
    for name,b in payload.items():atomic_write(out/name,b)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);s=run(p.parse_args().output)
    print(json.dumps({'selected_learning_rates':s['selected_learning_rates'],'counts':s['counts'],'final_summary':s['final_summary'],'primary_interval':s['primary_interval']},ensure_ascii=False))
