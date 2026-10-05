"""DL052 audited teaching implementation. CPU, float64, no hidden broadcasting."""
from pathlib import Path
import argparse,hashlib,json,math,platform,time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
DTYPE=torch.float64;DEVICE=torch.device('cpu')
torch.set_num_threads(1)
def tensor(x,grad=False):return torch.tensor(x,dtype=DTYPE,device=DEVICE,requires_grad=grad)
def require(ok,message):
    if not ok:raise ValueError(message)
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check_array(x,name):
    require(isinstance(x,np.ndarray) and x.dtype==np.float64,name+' must be float64 ndarray')
    require(x.ndim==3 and all(0<n<=512 for n in x.shape),name+' requires nonempty rank 3, axes <=512')
    require(np.isfinite(x).all() and np.max(np.abs(x))<=1e4,name+' must be finite, abs<=1e4')
def attention(q,k,v,allow,query_valid=None):
    for x,name in [(q,'Q'),(k,'K'),(v,'V')]:check_array(x,name)
    b,l,d=q.shape;bk,s,dk=k.shape
    require(b==bk==v.shape[0] and d==dk and s==v.shape[1],'incompatible shapes')
    require(isinstance(allow,np.ndarray) and allow.dtype==np.bool_ and allow.shape==(b,l,s),'allow must be bool [B,L,S]')
    if query_valid is None:query_valid=np.ones((b,l),dtype=np.bool_)
    require(isinstance(query_valid,np.ndarray) and query_valid.dtype==np.bool_ and query_valid.shape==(b,l),'query_valid must be bool [B,L]')
    require(np.all(allow.any(-1)|~query_valid),'active query has no allowed key')
    safe=allow.copy();safe[~query_valid]=False
    for bi,li in zip(*np.where(~query_valid)):safe[bi,li,0]=True
    score=q@k.swapaxes(1,2)/math.sqrt(d);masked=np.where(safe,score,-np.inf)
    exp=np.exp(masked-masked.max(-1,keepdims=True));a=exp/exp.sum(-1,keepdims=True)
    a=np.where(query_valid[...,None],a,0.0)
    return a@v,{'q':q,'k':k,'v':v,'a':a,'score':score,'allow':allow,'query_valid':query_valid}
def reverse(c,go):
    q,k,v,a=[c[z] for z in ['q','k','v','a']]
    require(isinstance(go,np.ndarray) and go.dtype==np.float64 and go.shape==(q.shape[0],q.shape[1],v.shape[2]),'bad output gradient')
    require(np.isfinite(go).all(),'nonfinite gradient');go=np.where(c['query_valid'][...,None],go,0.)
    ga=go@v.swapaxes(1,2);gs=a*(ga-(a*ga).sum(-1,keepdims=True))
    return {'go':go,'ga':ga,'gs':gs,'gq':gs@k/math.sqrt(q.shape[-1]),'gk':gs.swapaxes(1,2)@q/math.sqrt(q.shape[-1]),'gv':a.swapaxes(1,2)@go}
def scalar_attention(q,k,v,allow):
    b,l,d=q.shape;s=k.shape[1];dv=v.shape[2];out=np.zeros((b,l,dv),dtype=np.float64)
    for n in range(b):
        for i in range(l):
            js=[j for j in range(s) if allow[n,i,j]]
            if not js:raise ValueError('empty scalar support')
            scores=[sum(float(q[n,i,r])*float(k[n,j,r]) for r in range(d))/math.sqrt(d) for j in js]
            top=max(scores);ws=[math.exp(z-top) for z in scores];den=sum(ws)
            for c in range(dv):out[n,i,c]=sum(w/den*float(v[n,j,c]) for j,w in zip(js,ws))
    return out
def hand_inputs():
    x=np.array([[[1,0],[0,1]],[[1,1],[1,0]]],dtype=np.float64)
    y=np.array([[[.25],[-.25]],[[.5],[.75]]],dtype=np.float64)
    w=[np.array(z,dtype=np.float64) for z in [[[.5],[-.5]],[[1],[.5]],[[1],[-1]]]]
    return x,y,w
def objective(x,y,w):
    q,k,v=[x@z for z in w];o,c=attention(q,k,v,np.ones((x.shape[0],x.shape[1],x.shape[1]),dtype=np.bool_))
    loss=float(((o-y)**2).sum()/(2*y.size));g=reverse(c,(o-y)/y.size)
    per=[x.swapaxes(1,2)@g[z] for z in ['gq','gk','gv']];gw=[z.sum(0) for z in per]
    paths=[g[z]@ww.T for z,ww in zip(['gq','gk','gv'],w)]
    return loss,c,g,per,gw,paths
def hand_ledger():
    x,y,w=hand_inputs();states=[]
    for step in range(3):
        loss,c,g,per,gw,paths=objective(x,y,w);a=c['a'];jac=np.zeros((*a.shape,a.shape[-1]),dtype=np.float64)
        for b in range(2):
            for i in range(2):jac[b,i]=np.diag(a[b,i])-np.outer(a[b,i],a[b,i])
        rec={'step':step,'x':x.tolist(),'target':y.tolist(),'weights':[z.tolist() for z in w],'loss':loss,'sample_loss':(((a@c['v']-y)**2).sum((1,2))/4).tolist(),'prediction':(a@c['v']).tolist(),'jacobian':jac.tolist(),**{z:c[z].tolist() for z in ['q','k','v','a','score']},**{z:v.tolist() for z,v in g.items()},'weight_gradient_per_sample':[z.tolist() for z in per],'weight_gradient':[z.tolist() for z in gw],'input_gradient_paths':[z.tolist() for z in paths],'input_gradient':sum(paths).tolist()}
        states.append(rec)
        if step<2:w=[z-.1*gg for z,gg in zip(w,gw)]
    return states
def torch_objective(x,y,w):
    q,k,v=[x@z for z in w];a=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]),dim=-1);o=a@v
    return ((o-y)**2).mean()/2,o
def verify_derivatives():
    x,y,w=hand_inputs();_,c,g,per,gw,paths=objective(x,y,w)
    xt=tensor(x,True);wt=[tensor(z,True) for z in w];loss,_=torch_objective(xt,tensor(y),wt);loss.backward()
    result={'weight_autograd_max_abs':max(float(np.max(np.abs(t.grad.numpy()-z))) for t,z in zip(wt,gw)),'input_autograd_max_abs':float(np.max(np.abs(xt.grad.numpy()-sum(paths)))),'epsilon':1e-6,'detail':[]}
    h=result['epsilon']
    for wi in range(3):
        for ij in np.ndindex(w[wi].shape):
            plus=[z.copy() for z in w];minus=[z.copy() for z in w];plus[wi][ij]+=h;minus[wi][ij]-=h
            num=(objective(x,y,plus)[0]-objective(x,y,minus)[0])/(2*h);ana=float(gw[wi][ij]);result['detail'].append({'parameter':['WQ','WK','WV'][wi]+str(ij),'analytic':ana,'central_difference':num,'abs_error':abs(num-ana)})
    for ij in np.ndindex(x.shape):
        plus=x.copy();minus=x.copy();plus[ij]+=h;minus[ij]-=h
        num=(objective(plus,y,w)[0]-objective(minus,y,w)[0])/(2*h);ana=float(sum(paths)[ij]);result['detail'].append({'parameter':'X'+str(ij),'analytic':ana,'central_difference':num,'abs_error':abs(num-ana)})
    result['finite_difference_max_abs']=max(z['abs_error'] for z in result['detail']);return result
def api_probes():
    q=tensor([[[1.,0.],[0.,1.]]]);k=tensor([[[1.,0.],[0.,2.],[2.,1.]]]);v=tensor([[[2.,-1.],[4.,3.],[8.,5.]]]);allow=torch.tensor([[[True,False,True],[False,True,True]]],dtype=torch.bool,device=DEVICE)
    expected,c=attention(q.numpy(),k.numpy(),v.numpy(),allow.numpy());p=1/(1+math.exp(-1/math.sqrt(2)));closed=np.array([[[2*(1-p)+8*p,-1*(1-p)+5*p],[4*p+8*(1-p),3*p+5*(1-p)]]],dtype=np.float64);np.testing.assert_allclose(expected,closed,atol=2e-14,rtol=1e-14);sdpa=F.scaled_dot_product_attention(q,k,v,attn_mask=allow,dropout_p=0.)
    mha=torch.nn.MultiheadAttention(2,1,bias=False,dropout=0.,batch_first=True,dtype=DTYPE,device=DEVICE)
    with torch.no_grad():mha.in_proj_weight.copy_(torch.cat([torch.eye(2,dtype=DTYPE,device=DEVICE)]*3));mha.out_proj.weight.copy_(torch.eye(2,dtype=DTYPE,device=DEVICE))
    out,weights=mha(q,k,v,attn_mask=~allow[0],need_weights=True);wrong=F.scaled_dot_product_attention(q,k,v,attn_mask=~allow,dropout_p=0.)
    none=torch.zeros((2,3),dtype=torch.bool,device=DEVICE);raw=torch.softmax(torch.full((1,3),-torch.inf,dtype=DTYPE,device=DEVICE),dim=-1)
    empty_sdpa=F.scaled_dot_product_attention(q,k,v,attn_mask=none,dropout_p=0.);empty_true,_=mha(q,k,v,attn_mask=~none,need_weights=True);empty_false,_=mha(q,k,v,attn_mask=~none,need_weights=False)
    kp=torch.tensor([[False,False,True]],dtype=torch.bool,device=DEVICE);pk,_=mha(q,k,v,key_padding_mask=kp,need_weights=True)
    return {'torch':torch.__version__,'device':'CPU','dtype':'float64','expected':expected.tolist(),'closed_form':closed.tolist(),'sdpa':sdpa.tolist(),'mha':out.detach().tolist(),'weights':weights.detach().tolist(),'sdpa_error':float(np.max(np.abs(sdpa.numpy()-expected))),'mha_error':float(np.max(np.abs(out.detach().numpy()-expected))),'inverted_sdpa_max_change':float(torch.max(torch.abs(wrong-sdpa)).detach()),'all_masked':{'raw_softmax_all_nan':bool(torch.isnan(raw).all()),'sdpa_all_zero':bool((empty_sdpa==0).all()),'mha_need_weights_true_all_nan':bool(torch.isnan(empty_true).all()),'mha_need_weights_false_all_zero':bool((empty_false==0).all())},'key_padding_output':pk.detach().tolist(),'key_padding_does_not_zero_query':bool((pk[0,1]!=0).any())}
def leakage_probe():
    x=np.array([[[1.,0.],[0.,1.],[1.,1.],[2.,-1.]]],dtype=np.float64);alt=x.copy();alt[0,2:]=np.array([[9.,-7.],[-6.,8.]],dtype=np.float64);full=np.ones((1,4,4),dtype=np.bool_);results={}
    for name,mask in [('causal',np.tril(full)),('unmasked',full)]:
        old,_=attention(x,x,x,mask);new,_=attention(alt,alt,alt,mask);results[name]={'before':old.tolist(),'after':new.tolist(),'prefix_max_change':float(np.max(np.abs(old[:,:2]-new[:,:2])))}
    return results
def generate_dataset():
    data={'protocol':'fixed-v1','train_n':128,'test_n':512,'tokens':4,'seeds':[5201,5202,5203],'values':{}}
    for seed in data['seeds']:
        rng=np.random.default_rng(seed);data['values'][str(seed)]={'train':rng.normal(size=(128,4)).tolist(),'test':rng.normal(size=(512,4)).tolist()}
    return data
def leak_experiment(data):
    runs=[]
    for seed in data['seeds']:
        train=np.array(data['values'][str(seed)]['train'],dtype=np.float64);test=np.array(data['values'][str(seed)]['test'],dtype=np.float64)
        for alpha in [0.,8.]:
            q=np.zeros((1,3,4),dtype=np.float64)
            for t in range(3):q[0,t,t+1]=alpha*2
            k=np.eye(4,dtype=np.float64)[None]
            for condition in ['causal','unmasked']:
                allow=np.ones((1,3,4),dtype=np.bool_)
                if condition=='causal':
                    for t in range(3):allow[0,t,t+1:]=False
                _,c=attention(q,k,np.zeros((1,4,1),dtype=np.float64),allow);a=c['a'][0];htr=train@a.T;hte=test@a.T
                design=np.column_stack([htr.ravel(),np.ones(htr.size,dtype=np.float64)]);coef=np.linalg.lstsq(design,train[:,1:].ravel(),rcond=None)[0]
                ptr=coef[0]*htr+coef[1];pte=coef[0]*hte+coef[1]
                runs.append({'seed':seed,'alpha':alpha,'condition':condition,'attention':a.tolist(),'coefficient':coef.tolist(),'train_mse':float(np.mean((ptr-train[:,1:])**2)),'test_mse':float(np.mean((pte-test[:,1:])**2)),'test_per_sequence_mse':np.mean((pte-test[:,1:])**2,axis=1).tolist(),'test_predictions':pte.tolist(),'zero_baseline_test_mse':float(np.mean(test[:,1:]**2))})
    return runs
def scaling_experiment():
    rng=np.random.default_rng(5210);rows=[]
    for d in [1,4,16,64]:
        q=rng.normal(size=(256,1,d)).astype(np.float64);k=rng.normal(size=(256,16,d)).astype(np.float64);score=(q@k.swapaxes(1,2))[:,0]
        for scaled in [False,True]:
            z=score/(math.sqrt(d) if scaled else 1);e=np.exp(z-z.max(-1,keepdims=True));a=e/e.sum(-1,keepdims=True);entropy=-(a*np.log(a)).sum(-1)
            rows.append({'d':d,'scaled':scaled,'score_variance':float(z.var()),'mean_entropy_nats':float(entropy.mean()),'row_entropy':entropy.tolist()})
    return rows
def run(output):
    start=time.perf_counter();data=json.loads((ROOT/'data/sequences.json').read_text());manifest=json.loads((ROOT/'data/manifest.json').read_text());require(hashlib.sha256((ROOT/'data/sequences.json').read_bytes()).hexdigest()==manifest['sha256'],'dataset hash mismatch')
    r={'protocol':json.loads((ROOT/'data/protocol.json').read_text()),'hand':hand_ledger(),'derivatives':verify_derivatives(),'api':api_probes(),'leakage':leakage_probe(),'runs':leak_experiment(data),'scaling':scaling_experiment()};dump(output/'results.json',r)
    dump(output/'runtime.json',{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU','dtype':'float64','torch_num_threads':torch.get_num_threads(),'elapsed_seconds':time.perf_counter()-start,'meaning':'compute and JSON writing; not a speed benchmark; no peak-memory instrumentation','parameters':{'hand_trainable':6,'readout_trainable_per_fit':2,'fixed_qk_scalars':28},'scope':'single process; no GPU, half precision, dropout, sparse attention, long context or distributed measurements'})
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');args=p.parse_args();r=run(args.output);print(json.dumps({'loss':[s['loss'] for s in r['hand']],'max_fd_error':r['derivatives']['finite_difference_max_abs'],'all_masked':r['api']['all_masked'],'fits':len(r['runs'])},ensure_ascii=False,indent=2))
