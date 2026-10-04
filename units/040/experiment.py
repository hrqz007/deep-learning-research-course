"""040: reproducible training diagnosis, bounded search and one final test report.
CPU float64. Search never accepts test data. All results serialize before writes.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,csv,gzip,hashlib,io,json,math,os,platform,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
DTYPE=torch.float64
FIXTURES={'train.csv':'5f2474a8dca33e5e2a2d48d242782d8f54fedea248e9002b961d36f84de9fa63','validation.csv':'8d9abf336e3d617ddf2ca9e1894ef35f2d5783d95420582ed8b7821850586bb1','test.csv':'6dcec43f6145b543597a08f33bfd3e13b7903aa569bdb941477c174d224894f5','protocol.json':'a3d25d17510f1ff7e93421d4f21cae6226a6fc778a8697661a6ab0cabb8f5b69'}

def require(ok,message):
    if not ok:raise ValueError(message)
def real(x,name,low,high):
    require(not isinstance(x,(bool,np.bool_)) and isinstance(x,(int,float,np.integer,np.floating)) and math.isfinite(x) and low<=x<=high,'invalid '+name)
    return float(x)
def integer(x,name,low,high):
    require(not isinstance(x,(bool,np.bool_)) and isinstance(x,(int,np.integer)) and low<=x<=high,'invalid '+name)
    return int(x)
def vector(x,name):
    raw=np.asarray(x,dtype=object)
    require(raw.ndim==1 and raw.size>0,'expected nonempty vector '+name)
    require(all(not isinstance(v,(bool,np.bool_)) and isinstance(v,(int,float,np.integer,np.floating)) for v in raw),'real numeric '+name+' required')
    a=np.asarray(x,dtype=np.float64);require(np.isfinite(a).all(),'nonfinite '+name);return a

def verify_inputs():
    for name,digest in FIXTURES.items():
        require(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest,'modified fixture '+name)
    plan=json.loads((ROOT/'data/protocol.json').read_text())
    for config in plan['search']+plan['diagnostics']+[plan['tiny_probe'],plan['overfit_probe']]:validate_config(config)
    return plan

def load_split(name):
    require(name in ('train','validation','test'),'unknown split')
    with (ROOT/'data'/f'{name}.csv').open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f);require(r.fieldnames==['id','x','y','group'],'CSV schema');rows=list(r)
    require(len(rows)=={'train':48,'validation':128,'test':512}[name],'split row count')
    ids=[r['id'] for r in rows];require(len(set(ids))==len(ids) and all(i.startswith(name+'-') for i in ids),'split ID')
    x=vector([float(r['x']) for r in rows],'x');y=vector([float(r['y']) for r in rows],'y')
    groups=[r['group'] for r in rows];require(all(g==('left' if v<0 else 'right') for g,v in zip(groups,x)),'incorrect group')
    return {'ids':ids,'x':x,'y':y,'groups':groups}

def validate_config(c):
    require(set(c)=={'id','width','lr','l2','input_scale','label_mode','budget'},'config schema')
    require(isinstance(c['id'],str) and c['id'],'configuration ID')
    integer(c['width'],'width',0,64);real(c['lr'],'lr',1e-8,10.);real(c['l2'],'l2',0,10.)
    real(c['input_scale'],'input_scale',.01,100.);integer(c['budget'],'budget',1,3000)
    require(c['label_mode'] in ('intact','permuted'),'label mode')

def initialize(width,seed):
    width=integer(width,'width',0,64);seed=integer(seed,'seed',0,2**31-1)
    rng=np.random.default_rng(seed)
    if width==0:return np.array([float(rng.normal(0,.2)),0.])
    return np.r_[rng.normal(0,.8,width),np.zeros(width),rng.normal(0,1/math.sqrt(width),width),0.]

def forward(theta,x,width):
    if width==0:return theta[0]*x+theta[1],None
    h=torch.tanh(x[:,None]*theta[:width]+theta[width:2*width])
    return h@theta[2*width:3*width]+theta[-1],h

def numpy_forward(theta,x,width):
    if width==0:return theta[0]*x+theta[1]
    return np.tanh(x[:,None]*theta[:width]+theta[width:2*width])@theta[2*width:3*width]+theta[-1]

def penalty(theta,width):
    if width==0:return theta[0].square()/2
    return (theta[:width].square().sum()+theta[2*width:3*width].square().sum())/2

def fit(train,validation,config,seed,loss_limit=1e6):
    """Receives no test object/path. Records actual finite states, including failures."""
    validate_config(config);real(loss_limit,'loss_limit',1.,1e12)
    x=vector(train['x'],'train x');y=vector(train['y'],'train y')
    vx=vector(validation['x'],'validation x');vy=vector(validation['y'],'validation y')
    require(x.size==y.size and vx.size==vy.size and x.size>=2,'paired shapes')
    mean=float(x.mean());std=float(x.std());require(std>1e-12,'constant training feature')
    scale=config['input_scale'];xs=(x-mean)/std*scale;vxs=(vx-mean)/std*scale
    used_y=y.copy()
    if config['label_mode']=='permuted':used_y=used_y[np.random.default_rng(4070).permutation(len(y))]
    tx=torch.from_numpy(xs);ty=torch.from_numpy(used_y);tvx=torch.from_numpy(vxs);tvy=torch.from_numpy(vy)
    width=config['width'];theta=torch.tensor(initialize(width,seed),dtype=DTYPE,requires_grad=True)
    curve=[];states=[];failure=None;status='completed'
    for step in range(config['budget']+1):
        pred,h=forward(theta,tx,width);data=((pred-ty)**2).mean()/2
        objective=data+config['l2']*penalty(theta,width)
        gradient=torch.autograd.grad(objective,theta)[0]
        with torch.no_grad():
            valid,_=forward(theta,tvx,width);vl=((valid-tvy)**2).mean()/2
            pn=float(torch.linalg.vector_norm(theta));gn=float(torch.linalg.vector_norm(gradient));delta=-config['lr']*gradient
            row={'step':step,'train_data_loss':float(data),'train_objective':float(objective),'validation_data_loss':float(vl),'gradient_norm':gn,'parameter_norm':pn,'proposed_update_ratio':float(torch.linalg.vector_norm(delta))/(pn+1e-12),'saturated_fraction':float((torch.abs(h)>.99).double().mean()) if h is not None else 0.}
        require(all(math.isfinite(v) for v in row.values()),'nonfinite initial or accepted state')
        curve.append(row);states.append(theta.detach().tolist())
        if step==config['budget']:break
        candidate=theta.detach()+delta
        with torch.no_grad():
            cp,_=forward(candidate,tx,width);cl=float(((cp-ty)**2).mean()/2+config['l2']*penalty(candidate,width))
        if not torch.isfinite(candidate).all() or not math.isfinite(cl) or cl>loss_limit:
            status='stopped_candidate_loss_limit' if math.isfinite(cl) else 'stopped_nonfinite_candidate'
            failure={'attempted_update':step+1,'candidate_objective':cl if math.isfinite(cl) else str(cl),'candidate_parameters':candidate.tolist() if torch.isfinite(candidate).all() else None,'accepted_updates':step}
            break
        theta=candidate.requires_grad_()
    p=theta.detach().numpy();tp=numpy_forward(p,xs,width);vp=numpy_forward(p,vxs,width)
    return {'config':dict(config),'seed':seed,'status':status,'updates_applied':len(curve)-1,'sample_presentations':len(x)*(len(curve)-1),'normalization':{'mean':mean,'std':std,'input_scale':scale},'curve':curve,'parameters':states,'final_parameters':p.tolist(),'failure':failure,'train_prediction':tp.tolist(),'training_labels_used':used_y.tolist(),'validation_prediction':vp.tolist(),'train_clean_aligned_half_mse':float(np.mean((tp-y)**2)/2)}

def search(train,validation,plan):
    records=[];scores=[]
    for config in plan['search']:
        runs=[fit(train,validation,config,seed,plan['loss_limit']) for seed in plan['seeds']]
        records.extend(runs);complete=all(r['status']=='completed' for r in runs)
        scores.append({'id':config['id'],'eligible':complete,'mean_validation_half_mse':float(np.mean([r['curve'][-1]['validation_data_loss'] for r in runs])) if complete else None,'seed_validation_half_mse':[r['curve'][-1]['validation_data_loss'] for r in runs],'updates':sum(r['updates_applied'] for r in runs)})
    eligible=[(s['mean_validation_half_mse'],i,s['id']) for i,s in enumerate(scores) if s['eligible']]
    require(bool(eligible),'no eligible completed configuration')
    winner=min(eligible)[2]
    contrasts=[]
    for axis,fixed in [('lr','l2'),('l2','lr')]:
        fixed_values=sorted(set(c[fixed] for c in plan['search']))
        for fv in fixed_values:
            configs=sorted([c for c in plan['search'] if c[fixed]==fv],key=lambda c:c[axis])
            for a,b in zip(configs,configs[1:]):
                av=next(s['seed_validation_half_mse'] for s in scores if s['id']==a['id'])
                bv=next(s['seed_validation_half_mse'] for s in scores if s['id']==b['id'])
                ds=[y-x for x,y in zip(av,bv)]
                contrasts.append({'changed_parameter':axis,'from':a[axis],'to':b[axis],'fixed_parameter':fixed,'fixed_value':fv,'from_id':a['id'],'to_id':b['id'],'paired_seed_differences':ds,'mean_difference':float(np.mean(ds))})
    return {'single_factor_contrasts':contrasts,'scores':scores,'selected_id':winner,'baseline_id':plan['baseline_id'],'rule':plan['selection'],'runs':records,'test_data_used':False}

def paired_bootstrap(differences,replicates=2000,seed=4091):
    d=vector(differences,'paired differences');integer(replicates,'replicates',10,10000);integer(seed,'seed',0,2**31-1)
    rng=np.random.default_rng(seed);means=np.mean(d[rng.integers(0,len(d),(replicates,len(d)))],axis=1)
    return {'mean':float(d.mean()),'interval_percentile_95':np.quantile(means,[.025,.975]).tolist(),'replicates':replicates,'seed':seed,'row_differences':d.tolist(),'bootstrap_means':means.tolist(),'conditional_scope':'fixed training/selection and fitted ensembles; independent test rows only'}

def final_evaluate(train,test,selection,plan):
    predictions={};seed_results=[]
    for label,cid in [('baseline',selection['baseline_id']),('selected',selection['selected_id'])]:
        runs=[r for r in selection['runs'] if r['config']['id']==cid]
        require(len(runs)==len(plan['seeds']) and all(r['status']=='completed' for r in runs),'incomplete final models')
        ps=[]
        for r in runs:
            n=r['normalization'];x=(test['x']-n['mean'])/n['std']*n['input_scale']
            pred=numpy_forward(np.asarray(r['final_parameters']),x,r['config']['width']);ps.append(pred)
            seed_results.append({'label':label,'configuration':cid,'seed':r['seed'],'half_mse':float(np.mean((pred-test['y'])**2)/2),'prediction':pred.tolist()})
        predictions[label]=np.mean(ps,axis=0)
    # Prespecified simple references are fit only on training data.
    predictions['constant']=np.full(len(test['y']),float(train['y'].mean()))
    design=np.c_[train['x'],np.ones(len(train['x']))];beta=np.linalg.lstsq(design,train['y'],rcond=None)[0]
    predictions['least_squares_linear']=np.c_[test['x'],np.ones(len(test['x']))]@beta
    metrics=[]
    for label,pred in predictions.items():
        for group in ('all','left','right'):
            mask=np.ones(len(pred),bool) if group=='all' else np.array(test['groups'])==group
            residual=pred[mask]-test['y'][mask];hm=float(np.mean(residual**2)/2)
            metrics.append({'model':label,'group':group,'n':int(mask.sum()),'half_mse':hm,'rmse':math.sqrt(2*hm),'mean_residual':float(residual.mean())})
    d=((predictions['selected']-test['y'])**2-(predictions['baseline']-test['y'])**2)/2
    return {'selected_id':selection['selected_id'],'seed_results':seed_results,'ensemble_metrics':metrics,'predictions':{k:v.tolist() for k,v in predictions.items()},'test_ids':test['ids'],'paired_bootstrap':paired_bootstrap(d,plan['bootstrap']['replicates'],plan['bootstrap']['seed']),'linear_coefficients':beta.tolist()}

def hand_step(theta,lr=F(1,5)):
    """Exact scalar reference, order w1,w2,b1,b2,a1,a2,c."""
    rows=[];contributions=[]
    for x,y in [(F(-1),F(0)),(F(1),F(1))]:
        z=[theta[j]*x+theta[2+j] for j in range(2)];h=[max(F(0),t) for t in z]
        gate=[F(int(t>0)) for t in z];pred=sum(theta[4+j]*h[j] for j in range(2))+theta[6];residual=pred-y;up=residual/2
        dh=[up*theta[4+j] for j in range(2)];dz=[dh[j]*gate[j] for j in range(2)]
        g=[dz[0]*x,dz[1]*x,dz[0],dz[1],up*h[0],up*h[1],up]
        rows.append({'x':x,'y':y,'z':z,'h':h,'gate':gate,'prediction':pred,'residual':residual,'sample_half_square':residual**2/2,'dloss_dprediction':up,'dprediction_dh':theta[4:6],'dh_dz':gate,'dz_dw':[x,x],'dz_db':[F(1),F(1)],'dloss_dh':dh,'dloss_dz':dz,'sample_parameter_contributions':g})
        contributions.append(g)
    gradient=[sum(g[j] for g in contributions) for j in range(7)];after=[p-lr*g for p,g in zip(theta,gradient)]
    return {'theta':theta,'rows':rows,'loss':sum(r['sample_half_square'] for r in rows)/2,'gradient':gradient,'lr':lr,'delta':[-lr*g for g in gradient],'after':after}

def hand_trace():
    theta=[F(1,2),F(-1,2),F(1),F(1),F(1),F(-1,2),F(1,10)];steps=[]
    for _ in range(2):
        result=hand_step(theta);steps.append(result);theta=result['after']
    third=hand_step(theta)
    return {'parameter_order':['w1','w2','b1','b2','a1','a2','c'],'steps':steps,'third_forward':third['rows'],'third_loss':third['loss']}

def selection_noise_simulation():
    rng=np.random.default_rng(4080);n=2000;val=1.+.1*rng.standard_normal((n,64));audit=1.+.1*rng.standard_normal((n,64));records=[]
    for k in (1,4,16,64):
        chosen=np.argmin(val[:,:k],axis=1);v=val[np.arange(n),chosen];a=audit[np.arange(n),chosen]
        records.append({'candidates':k,'chosen_indices':chosen.tolist(),'validation_selected':v.tolist(),'independent_audit':a.tolist(),'mean_validation':float(v.mean()),'mean_audit':float(a.mean()),'optimism':float((a-v).mean()),'mcse_optimism':float(np.std(a-v,ddof=1)/math.sqrt(n))})
    return {'seed':4080,'repetitions':n,'model':'all candidates have true score 1; validation/audit errors are independent N(0,0.1^2), not trained networks','records':records,'bernoulli_exact':[{'k':k,'expected_selected_validation_error':2.**(-k),'independent_error':.5,'optimism':.5-2.**(-k)} for k in (1,2,4,8)]}

def serial(value):
    if isinstance(value,F):return {'fraction':str(value),'float':float(value)}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value

def json_bytes(value):return (json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def gzip_bytes(raw):
    buffer=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',compresslevel=9,mtime=0,fileobj=buffer) as f:f.write(raw)
    return buffer.getvalue()
def csv_bytes(rows):
    buffer=io.StringIO(newline='');w=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return buffer.getvalue().encode('utf-8')
def atomic_write(path,data):
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name,delete=False) as f:temp=Path(f.name);f.write(data)
        os.replace(temp,path)
    finally:
        if temp is not None and temp.exists():temp.unlink()

def run(output=None):
    plan=verify_inputs();out=Path(output).resolve() if output else ROOT/'outputs'
    require(not out.exists() or out.is_dir(),'output must be directory')
    require(out not in (ROOT,ROOT/'data',ROOT/'figures'),'output conflicts with protected artifacts')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train=load_split('train');validation=load_split('validation')
    require(not set(train['ids'])&set(validation['ids']),'split overlap')
    ledger=['validated fixture integrity (hashes only)','loaded train and validation','fit normalization inside each training run']
    selected=search(train,validation,plan);ledger.append('completed 27 search runs and selected '+selected['selected_id']+' without test')
    diagnostics=[fit(train,validation,c,s,plan['loss_limit']) for c in plan['diagnostics'] for s in plan['seeds']]
    tiny={k:(v[:8] if isinstance(v,(list,np.ndarray)) else v) for k,v in train.items()}
    probes=[fit(tiny,tiny,dict(plan['tiny_probe'],label_mode=mode),4001) for mode in ('intact','permuted')]
    small={k:(v[:16] if isinstance(v,(list,np.ndarray)) else v) for k,v in train.items()}
    overfit=fit(small,validation,plan['overfit_probe'],4001);best=min(range(len(overfit['curve'])),key=lambda i:overfit['curve'][i]['validation_data_loss'])
    ledger.append('completed prespecified diagnostic probes; no changes to selected configuration')
    test=load_split('test');require(not(set(test['ids'])&(set(train['ids'])|set(validation['ids']))),'test overlap')
    ledger.append('loaded test for final fixed comparison only')
    final=final_evaluate(train,test,selected,plan);ledger.append('reported selected/baseline ensembles and prespecified simple references; no post-test changes')
    allruns=[]
    for stage,runs in [('search',selected['runs']),('diagnostic',diagnostics),('tiny_probe',probes),('overfit_probe',[overfit])]:
        for i,r in enumerate(runs):allruns.append(dict(r,stage=stage,run_id=f'{stage}-{i:02d}'))
    curves=[dict(stage=r['stage'],run_id=r['run_id'],configuration=r['config']['id'],seed=r['seed'],**row) for r in allruns for row in r['curve']]
    summaries=[{'run_id':r['run_id'],'stage':r['stage'],'config':r['config'],'seed':r['seed'],'status':r['status'],'updates_applied':r['updates_applied'],'sample_presentations':r['sample_presentations'],'final_train_data_loss':r['curve'][-1]['train_data_loss'],'final_validation_data_loss':r['curve'][-1]['validation_data_loss'],'failure':r['failure']} for r in allruns]
    summary={'unit':'040','environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'dtype':'float64','device':'cpu'},'protocol':plan,'access_ledger':ledger,'search_scores':selected['scores'],'single_factor_contrasts':selected['single_factor_contrasts'],'selected_id':selected['selected_id'],'baseline_id':selected['baseline_id'],'runs':summaries,'overfit_probe':{'best_validation_step':best,'best_validation_loss':overfit['curve'][best]['validation_data_loss'],'final_validation_loss':overfit['curve'][-1]['validation_data_loss'],'final_train_loss':overfit['curve'][-1]['train_data_loss'],'used_for_main_selection':False},'counts':{'runs':len(allruns),'accepted_updates':sum(r['updates_applied'] for r in allruns),'curve_rows':len(curves),'completed_runs':sum(r['status']=='completed' for r in allruns),'stopped_runs':sum(r['status']!='completed' for r in allruns)}}
    tracebytes=json_bytes({'runs':allruns});summary['trace_storage']={'format':'gzip','mtime':0,'filename':'','uncompressed_sha256':hashlib.sha256(tracebytes).hexdigest(),'uncompressed_bytes':len(tracebytes)}
    payload={'results.json':json_bytes(summary),'hand_chain.json':json_bytes(hand_trace()),'final_evaluation.json':json_bytes(final),'selection_simulation.json':json_bytes(selection_noise_simulation()),'training_curves.csv':csv_bytes(curves),'training_traces.json.gz':gzip_bytes(tracebytes)}
    # Serialize all outputs before creating/replacing anything. Per-file, not multi-file, atomicity.
    out.mkdir(parents=True,exist_ok=True)
    for name,data in payload.items():atomic_write(out/name,data)
    return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    r=run(parser.parse_args().output);print(json.dumps({'selected_id':r['selected_id'],'counts':r['counts'],'overfit_probe':r['overfit_probe']},ensure_ascii=False))
