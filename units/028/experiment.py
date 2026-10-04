"""028: broadcast VJP, shared nodes, numerical checks and explicit AD boundaries."""
from pathlib import Path
from decimal import Decimal
import argparse,csv,io,json,hashlib,math,os
import numpy as np
from engine import Tensor,vjp,topological,sum_to_shape,numeric
HERE=Path(__file__).resolve().parent
OUTPUT_NAMES=('summary.json','coordinates.csv','vjp.csv','finite_difference.csv','boundaries.csv')
TEACHING_HASHES={'data/config.json': 'cae51c4e66ef4fe920a9d4e457a7eae97aa0b990f4cb064994aa2cec72347cef', 'outputs/boundaries.csv': '8a2b2df0aed4bfe2dc2c8ac1d597ee6ce53f654bafe205fe4a4ff054e38707bf', 'outputs/coordinates.csv': '084244a88755db5de003b1c5418563140fc37ab1145a8e6ea1bfc18887c8f1a1', 'outputs/finite_difference.csv': '28d47b344b476c6f092a3162423d357d38bb064a3f80af6fa3202692ca84bddc', 'outputs/summary.json': '746a67c1510968d7e5856fed99a2b6dde748357e745b487b57124510f8de3c7a', 'outputs/vjp.csv': '405de2f8efbabd17daaafb71991eb4dac2115c9ce026238332ba9394b1f69ca4'}

def parse_float(text):
    d=Decimal(text);f=float(d)
    if not math.isfinite(f) or (d!=0 and f==0):raise ValueError('JSON numeric conversion lost a finite nonzero value')
    return f

def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out

def read_config(path):
    path=Path(path)
    if path.stat().st_size>10000:raise ValueError('config exceeds10000 bytes')
    def invalid(x):raise ValueError('nonfinite JSON token')
    c=json.loads(path.read_text(),parse_float=parse_float,object_pairs_hook=unique,parse_constant=invalid)
    if type(c) is not dict or set(c)!={'schema','X','w','b','seed'} or type(c['schema']) is not int or c['schema']!=1:raise ValueError('exact schema1 config required')
    for key,shape in [('X',(2,3)),('w',(3,)),('b',()),('seed',(2,3))]:
        a=numeric(c[key],key)
        if a.shape!=shape or np.any(np.abs(a)>3):raise ValueError(key+': fixed shape and magnitude<=3 required')
        c[key]=a
    return c

def build(X,w,b):
    nodes={'X':Tensor(X,name='X'),'w':Tensor(w,name='w'),'b':Tensor(b,name='b')}
    nodes['a']=nodes['X']*nodes['w']+nodes['b'];nodes['h']=nodes['a'].tanh()
    nodes['z']=nodes['h']*nodes['h']+nodes['h'];nodes['L']=nodes['z'].sum()*(1/nodes['z'].value.size)
    return nodes

def plain(X,w,b):
    """Independent NumPy reference for already validated inputs, not a public validator."""
    h=np.tanh(X*w+b);return h*h+h

def scalar_reference(X,w,b,seed):
    """Indexed reference for already validated inputs; no graph/pullback calls."""
    gx=np.zeros_like(X);gw=np.zeros_like(w);gb=0.
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            a=float(X[i,j])*float(w[j])+float(b);h=math.tanh(a);q=math.exp(-2*abs(a))
            local=(2*h+1)*4*q/(1+q)**2*float(seed[i,j])
            gx[i,j]=local*float(w[j]);gw[j]+=local*float(X[i,j]);gb+=local
    return gx,gw,gb

def csv_string(rows,fields):
    # Reject nonfinite numeric fields before CSV can stringify them silently.
    json.dumps(rows,allow_nan=False)
    b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return b.getvalue()

def serialize(c):
    X,w,b,seed=(c[k] for k in ['X','w','b','seed']);nodes=build(X,w,b);g=vjp(nodes['L']);weighted=vjp(nodes['z'],seed)
    meanseed=np.full((2,3),1/6);ref=scalar_reference(X,w,b,meanseed);wref=scalar_reference(X,w,b,seed)
    rows=[]
    for key in ['X','w','b','a','h','z','L']:
        value=nodes[key].value
        for idx in np.ndindex(value.shape):rows.append({'node':key,'index':','.join(map(str,idx)) or 'scalar','value':float(value[idx]),'mean_gradient':float(g[nodes[key]][idx])})
    vr=[]
    for key,r in zip(['X','w','b'],wref):
        r=np.asarray(r)
        for idx in np.ndindex(nodes[key].shape):vr.append({'node':key,'index':','.join(map(str,idx)) or 'scalar','vjp':float(weighted[nodes[key]][idx]),'scalar_reference':float(r[idx])})
    fd=[]
    for exponent in range(1,10):
        eps=10.**(-exponent)
        for key in ['X','w','b']:
            target=g[nodes[key]]
            for idx in np.ndindex(c[key].shape):
                plus={k:c[k].copy() for k in ['X','w','b']};minus={k:c[k].copy() for k in ['X','w','b']}
                plus[key][idx]+=eps;minus[key][idx]-=eps
                estimate=(float(plain(**plus).mean())-float(plain(**minus).mean()))/(2*eps)
                exact=float(target[idx]);fd.append({'node':key,'index':','.join(map(str,idx)) or 'scalar','epsilon':eps,'vjp':exact,'finite_difference':estimate,'absolute_error':abs(estimate-exact)})
    # Closed-form/higher-order boundary: deliberately rebuild the derivative expression.
    t=Tensor(.5);f=.5*(t*t+t)*(t*t+t);first=float(vjp(f)[t]);d=(t*t+t)*(2*t+1);second=float(vjp(d)[t])
    zero=Tensor(0.);relu=zero.relu();identity=zero.relu()-(-zero).relu()
    q=Tensor(2.);connected=q*q*q;barrier=(q*q).detach()*q
    br=[{'case':'relu_at_zero','forward':float(relu.value),'engine_gradient':float(vjp(relu)[zero]),'comparison':.5,'meaning':'comparison is symmetric difference; ordinary derivative does not exist'},
        {'case':'relu_identity_at_zero','forward':float(identity.value),'engine_gradient':float(vjp(identity)[zero]),'comparison':1.,'meaning':'forward equals x; local ReLU conventions compose to0at0'},
        {'case':'cube_connected','forward':float(connected.value),'engine_gradient':float(vjp(connected)[q]),'comparison':12.,'meaning':'ordinary derivative of x cubed at2'},
        {'case':'cube_with_barrier','forward':float(barrier.value),'engine_gradient':float(vjp(barrier)[q]),'comparison':12.,'meaning':'deliberate stopped-gradient rule, not ordinary derivative of forward x cubed'},
        {'case':'polynomial_first','forward':float(f.value),'engine_gradient':first,'comparison':1.5,'meaning':'first derivative at0.5'},
        {'case':'rebuilt_derivative','forward':float(d.value),'engine_gradient':second,'comparison':5.5,'meaning':'manual derivative expression rebuilt as graph; not automatic higher-order support'}]
    # Exact broadcast sums: each output seed returns to the copied source position.
    u=Tensor([[1.],[2.]]);v=Tensor([[.5,-1.,1.5]]);C=np.array([[1.,2.,3.],[4.,5.,6.]])
    bg=vjp(u+v,C)
    sx=Tensor(2.);sy=Tensor(-1.)
    F=sx*sy*np.array([1.,0.])+(sx+sy*sy)*np.array([0.,1.])
    smallg=vjp(F,[3.,-1.]);small={'input':[2.,-1.],'output':F.value.tolist(),'jacobian':[[-1.,2.],[1.,-2.]],'seed':[3.,-1.],'vjp':[float(smallg[sx]),float(smallg[sy])],'direction':[.5,2.],'jvp':[3.5,-3.5],'dual_dot':14.}
    summary={'config':{'schema':1,**{k:c[k].tolist() for k in ['X','w','b','seed']}},'mean_objective':float(nodes['L'].value),'mean_gradients':{k:g[nodes[k]].tolist() for k in ['X','w','b']},'weighted_gradients':{k:weighted[nodes[k]].tolist() for k in ['X','w','b']},'mean_scalar_reference_max_error':max(float(np.max(np.abs(g[nodes[k]]-r))) for k,r in zip(['X','w','b'],ref)),'weighted_scalar_reference_max_error':max(float(np.max(np.abs(weighted[nodes[k]]-r))) for k,r in zip(['X','w','b'],wref)),'small_jacobian':small,'broadcast_hand':{'u_gradient':bg[u].tolist(),'v_gradient':bg[v].tolist()},'node_count':len(topological(nodes['L'])),'edge_count':sum(len(v._parents) for v in topological(nodes['L'])),'scope':'First-order toy differentiation; no training or generalization claim; explicit derivative barriers and nonsmooth conventions must be interpreted.'}
    return {'summary.json':json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n','coordinates.csv':csv_string(rows,['node','index','value','mean_gradient']),'vjp.csv':csv_string(vr,['node','index','vjp','scalar_reference']),'finite_difference.csv':csv_string(fd,['node','index','epsilon','vjp','finite_difference','absolute_error']),'boundaries.csv':csv_string(br,['case','forward','engine_gradient','comparison','meaning'])}

def run(config=None,output=None):
    config=Path(config) if config is not None else HERE/'data/config.json';output=Path(os.path.abspath(output)) if output is not None else HERE/'outputs'
    c=read_config(config)
    if any(p.is_symlink() for p in [output,*output.parents]) or (output.exists() and not output.is_dir()):raise ValueError('unsafe output directory')
    for name in OUTPUT_NAMES:
        p=output/name
        if p.is_symlink() or (p.exists() and not p.is_file()) or p.resolve()==config.resolve():raise ValueError('unsafe output target')
    payload=serialize(c)
    output.mkdir(parents=True,exist_ok=True)
    for name in OUTPUT_NAMES:(output/name).write_text(payload[name],encoding='utf-8')
    return json.loads(payload['summary.json'])

def require_teaching_inputs():
    if not TEACHING_HASHES:raise ValueError('teaching digest ledger not initialized')
    for name,digest in TEACHING_HASHES.items():
        p=HERE/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('fixed teaching file differs: '+name)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=HERE/'data/config.json');p.add_argument('--output',type=Path,default=HERE/'outputs');a=p.parse_args();s=run(a.config,a.output);print(json.dumps({'mean_objective':s['mean_objective'],'weighted_reference_error':s['weighted_scalar_reference_max_error']}))
