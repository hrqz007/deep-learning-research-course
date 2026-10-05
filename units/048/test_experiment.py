"""Executable checks use explicit exceptions: python -O cannot disable them."""
from pathlib import Path
import json,math,tempfile
import numpy as np
import torch
from torch import nn
from experiment import SmallCNN,transform,validate_images,load_split,evaluate,METHODS
from reference import hand_ledger,hand_round,metrics,conv_relu_pool_reference
from generate_data import make_split
HERE=Path(__file__).resolve().parent

def close(a,b,tol=1e-7):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    if not error<=tol:raise RuntimeError(f'numerical error {error} > {tol}')
    return error

def rejects(f):
    try:f()
    except (ValueError,TypeError):return
    raise RuntimeError('invalid input was accepted')

def main():
    torch.set_num_threads(1);errors={}
    for j,r in enumerate(hand_ledger()['rounds']):
        t=torch.tensor(r['theta'],dtype=torch.float64,requires_grad=True);w,b,a,c=t;x=torch.tensor([[0.,1.],[1.,2.]],dtype=torch.float64);y=torch.tensor([0.,1.],dtype=torch.float64);h=torch.relu(w*x+b);logits=a*h.mean(1)+c;loss=nn.functional.binary_cross_entropy_with_logits(logits,y);loss.backward();errors[f'hand_autograd_{j}']=close(t.grad,r['gradient'],1e-12);close(loss.item(),r['loss'],1e-12)
        finite=[]
        for k in range(4):
            plus=np.array(r['theta']);minus=plus.copy();plus[k]+=1e-6;minus[k]-=1e-6;finite.append((hand_round(plus)['loss']-hand_round(minus)['loss'])/2e-6)
        errors[f'hand_finite_difference_{j}']=close(finite,r['gradient'],2e-9)
    torch.manual_seed(482);m=SmallCNN().double();x=torch.rand((2,1,20,20),dtype=torch.float64)
    with torch.no_grad():target=m(x).numpy()
    h=conv_relu_pool_reference(x.numpy(),m.conv1.weight.detach().numpy(),m.conv1.bias.detach().numpy());h=conv_relu_pool_reference(h,m.conv2.weight.detach().numpy(),m.conv2.bias.detach().numpy());actual=h.reshape(2,-1)@m.head.weight.detach().numpy().T+m.head.bias.detach().numpy();errors['full_cnn_numpy_forward']=close(actual,target,2e-12)
    for logits,y in [(np.array([[1000.,-1000.],[-1000.,1000.]]),np.array([1,1])),(np.array([[0.,0.],[.1,-.4],[8.,-9.]]),np.array([0,1,0]))]:
        r=metrics(logits,y);loss=nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y)).item();close(r['loss'],loss,1e-12);close(r['probabilities'],torch.softmax(torch.from_numpy(logits),1).numpy(),1e-12)
    x,y,meta=make_split(8,9,.5);xx=torch.from_numpy(x);base=xx.clone()
    for method in METHODS:
        for training in (False,True):
            z=transform(xx,method,torch.Generator().manual_seed(8),training);validate_images(z)
            if method in ('random_stamp','mask_stamp'):close(z[:,:,5:,:],xx[:,:,5:,:],0);close(z[:,:,:5,5:],xx[:,:,:5,5:],0)
            if not training and method!='mask_stamp':close(z,xx,0)
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        transformed=transform(xx,'brightness',torch.Generator().manual_seed(8),True)
        if transformed.dtype!=xx.dtype:raise RuntimeError('augmentation leaked global default dtype')
    finally:torch.set_default_dtype(previous)
    close(base,xx,0);close(transform(xx,'random_stamp',torch.Generator().manual_seed(8),True),transform(xx,'random_stamp',torch.Generator().manual_seed(8),True),0)
    invalid=[torch.empty((0,1,20,20)),torch.ones((1,1,19,20)),torch.ones((1,3,20,20)),torch.ones((1,1,20,20),dtype=torch.int64),torch.ones((1,1,20,20),dtype=torch.float64),torch.full((1,1,20,20),float('nan')),torch.full((1,1,20,20),float('inf')),torch.full((1,1,20,20),-1.),torch.full((1,1,20,20),2.)]
    for val in invalid:rejects(lambda val=val:validate_images(val))
    rejects(lambda:transform(xx,'unknown'));rejects(lambda:transform(xx,'random_stamp',training=True));rejects(lambda:transform(xx,'brightness',training=True))
    for n in [True,3,0,-2,10002,2.5]:rejects(lambda n=n:make_split(n,1,.5))
    rejects(lambda:make_split(4,1,1.1));rejects(lambda:make_split(4,-1,.5));rejects(lambda:hand_round([1,2,3,float('inf')]));rejects(lambda:hand_round([1,2,3,101]));rejects(lambda:metrics(np.empty((0,2)),np.array([],int)));rejects(lambda:metrics(np.array([[1e300,0.]]),np.array([0])));rejects(lambda:metrics(np.array([[0.,1.]]),np.array([2])));rejects(lambda:metrics(np.array([[0.,1.]]),np.array([0.])))
    result_file=HERE/'outputs/results.json'
    if result_file.exists():
        r=json.loads(result_file.read_text());ids=[];hashes=[]
        for name in ('train','dev_iid','dev_shift','test_iid','test_shift'):
            meta=json.loads((HERE/f'data/{name}_meta.json').read_text());ids.extend(q['base_id'] for q in meta);hashes.extend(q['image_sha256'] for q in meta)
        if len(set(ids))!=len(ids) or len(set(hashes))!=len(hashes):raise RuntimeError('split identity collision')
        for method in METHODS:
            score=[]
            for lr in (.03,.1):
                rr=[q for q in r['runs'] if q['method']==method and q['lr']==lr];v=sum((metrics(np.array(q['dev']['dev_iid']['logits']),np.array(q['dev']['dev_iid']['labels']))['loss']+metrics(np.array(q['dev']['dev_shift']['logits']),np.array(q['dev']['dev_shift']['labels']))['loss'])/2 for q in rr)/3;score.append((v,lr))
            if min(score)[1]!=r['freeze']['selected'][method]['lr']:raise RuntimeError('selection differs from independent CE recomputation')
        for q in r['finalists']:
            for name in ('test_iid','test_shift'):
                v=q[name];z=metrics(np.array(v['logits']),np.array(v['labels']));close(z['loss'],v['loss'],2e-7);close(z['accuracy'],v['accuracy'],1e-8)
                if sum(g['errors'] for g in v['groups'].values())!=len(v['error_indices']):raise RuntimeError('group errors do not reconcile')
                if sum(g['n'] for g in v['groups'].values())!=v['n']:raise RuntimeError('group denominators do not reconcile')
        if [e['event'] for e in r['events']][-2:]!=['selection_frozen','test_arrays_first_loaded']:raise RuntimeError('test gate event order')
    result={'status':'passed','numerical_errors':errors,'boundary_cases':26,'checks':['three full hand rounds vs autograd and finite differences','full CNN NumPy forward','stable CE extreme logits','reproducible nonmutating augmentation','unsupported input rejection','all final metrics and groups','development-only selection recomputation','disjoint identities and bytes','freeze event before first test load']};print(json.dumps(result,ensure_ascii=False,indent=2));return result
if __name__=='__main__':main()
