"""Small explicit numerical references. No torch, no autograd, no COCO claim."""
import numpy as np

def finite(x, name, ndim=None):
    a=np.asarray(x)
    if a.dtype.kind not in 'fiu' or (ndim is not None and a.ndim!=ndim) or a.size>2_000_000: raise ValueError(name+': bounded real array required')
    a=a.astype(np.float64,copy=False)
    if not np.isfinite(a).all() or np.any(np.abs(a)>1e6): raise ValueError(name+': finite magnitude <=1e6 required')
    return a

def boxes(x):
    a=finite(x,'boxes',2)
    if a.shape[1]!=4 or np.any(a[:,2:]-a[:,:2]<1e-6): raise ValueError('boxes must be Kx4 half-open XYXY with side lengths >=1e-6')
    return a

def iou(a,b):
    a=boxes(a);b=boxes(b)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(-1);area=(a[:,2:]-a[:,:2]).prod(-1);other=(b[:,2:]-b[:,:2]).prod(-1)
    return inter/(area[:,None]+other[None,:]-inter)

def nms(b,s,t=.5):
    b=boxes(b);s=finite(s,'scores',1)
    if s.shape!=(len(b),) or not np.isfinite(t) or not 0<=t<=1: raise ValueError('scores or threshold invalid')
    order=list(np.argsort(-s,kind='stable'));keep=[]
    while order:
        i=order.pop(0);keep.append(int(i))
        if order: order=[j for j,v in zip(order,iou(b[i:i+1],b[order])[0]) if v<=t]
    return keep

def smooth_l1(r,beta=1.):
    r=finite(r,'residual')
    if not np.isscalar(beta) or not np.isfinite(beta) or beta<=0: raise ValueError('beta must be positive finite')
    a=np.abs(r);small=a<beta;loss=np.empty_like(r);grad=np.empty_like(r)
    loss[small]=.5*r[small]*(r[small]/beta);grad[small]=r[small]/beta
    loss[~small]=a[~small]-beta/2;grad[~small]=np.sign(r[~small])
    return loss,grad

def sigmoid(z):
    z=finite(z,'logit');return np.exp(-np.logaddexp(0,-z))

def bce(z,y,q=1.):
    z=finite(z,'logits');y=finite(y,'targets')
    if z.shape!=y.shape or z.size==0 or not np.isin(y,[0,1]).all() or not np.isscalar(q) or not np.isfinite(q) or not 0<q<=1e3: raise ValueError('BCE domain: same nonempty shape, binary y, 0<q<=1000')
    # (1-y)*softplus(z)+q*y*softplus(-z), stable even for |z|=1000.
    loss=(1-y)*np.logaddexp(0,z)+q*y*np.logaddexp(0,-z)
    grad=((1-y)*sigmoid(z)-q*y*sigmoid(-z))/z.size
    return float(loss.mean()),grad

def mask_metrics(pred,target):
    p=np.asarray(pred);y=np.asarray(target)
    if p.ndim!=3 or p.shape!=y.shape or not p.size or p.size>2_000_000 or p.dtype.kind not in 'bifu' or y.dtype.kind not in 'bifu' or not np.isin(p,[0,1]).all() or not np.isin(y,[0,1]).all():raise ValueError('metrics need equal nonempty NHW binary masks')
    p=p.astype(bool);y=y.astype(bool);axes=(1,2)
    tp=(p&y).sum(axes);fp=(p&~y).sum(axes);fn=(~p&y).sum(axes);tn=(~p&~y).sum(axes);u=tp+fp+fn
    per=np.divide(tp,u,out=np.ones(len(y),float),where=u>0)
    total=int(u.sum());fg=float(tp.sum()/total) if total else 1.
    empty=~y.any(axes)
    return {'foreground_iou_micro':fg,'foreground_iou_macro_empty1':float(per.mean()),'mean_class_iou_micro':float((fg+(tn.sum()/(tn+fp+fn).sum() if (tn+fp+fn).sum() else 1.))/2),'pixel_accuracy':float((tp+tn).sum()/y.size),'empty_false_positive_rate':float(p[empty].any(axes).mean()) if empty.any() else None,'per_image_iou':per.tolist(),'tp':tp.tolist(),'fp':fp.tolist(),'fn':fn.tolist(),'tn':tn.tolist()}

def detection_ap(gt,pred,threshold=.5):
    """Single class, explicit image IDs, no ignore/crowd/maxDets; global stable scores.
    gt: list of {image_id, box}; pred additionally score. AP=None if no GT.
    """
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('invalid IoU threshold')
    for r in gt+pred:
        if not isinstance(r,dict) or not isinstance(r.get('image_id'),str) or not r['image_id']:raise ValueError('nonempty string image_id required')
        boxes(np.asarray([r['box']]))
    scores=finite([r['score'] for r in pred],'scores',1)
    order=np.argsort(-scores,kind='stable');used=set();tp=[];matches=[]
    for k in order:
        r=pred[int(k)];candidates=[j for j,g in enumerate(gt) if g['image_id']==r['image_id'] and j not in used]
        vals=iou([r['box']],[gt[j]['box'] for j in candidates])[0] if candidates else []
        best=int(np.argmax(vals)) if len(vals) else None
        hit=best is not None and vals[best]>=threshold
        j=candidates[best] if hit else None
        if hit:used.add(j)
        tp.append(int(hit));matches.append(j)
    cum=np.cumsum(tp);fp=np.arange(1,len(tp)+1)-cum;prec=cum/np.arange(1,len(tp)+1);rec=cum/len(gt) if gt else np.zeros(len(tp))
    if not gt:ap=None;ap101=None
    else:
        mr=np.r_[0,rec,1];mp=np.r_[0,prec,0];mp=np.maximum.accumulate(mp[::-1])[::-1];ids=np.flatnonzero(mr[1:]!=mr[:-1])+1;ap=float(np.sum((mr[ids]-mr[ids-1])*mp[ids]))
        ap101=float(np.mean([max(prec[rec>=r],default=0.) for r in np.linspace(0,1,101)]))
    return {'order':order.tolist(),'tp':tp,'fp':(1-np.array(tp,dtype=int)).tolist(),'matched_gt':matches,'precision':prec.tolist(),'recall':rec.tolist(),'ap_all_point':ap,'ap_101_point':ap101,'num_gt':len(gt)}

def hand_ledger():
    x=np.array([[0.,1.],[1.,2.]]);y=np.array([[0.,1.],[1.,0.]]);theta=np.array([np.log(2),0.]);rounds=[]
    for step in range(3):
        z=theta[0]*x+theta[1];p=sigmoid(z);loss,g=bce(z,y);each=[]
        for i in range(2):
            pixels=[]
            for j in range(2):
                # d mean loss / d pixel probability and each local chain.
                dp=(-y[i,j]/p[i,j]+(1-y[i,j])/(1-p[i,j]))/4
                pixels.append({'x':x[i,j],'y':y[i,j],'z':z[i,j],'p':p[i,j],'loss':float(np.logaddexp(0,z[i,j])-y[i,j]*z[i,j]),'dL_dp':dp,'dp_dz':p[i,j]*(1-p[i,j]),'dL_dz':g[i,j],'dz_dw':x[i,j],'dz_db':1.,'dw_contribution':g[i,j]*x[i,j],'db_contribution':g[i,j],'dL_dx':g[i,j]*theta[0]})
            each.append({'pixels':pixels,'image_mean_loss':float(np.mean([v['loss'] for v in pixels]))})
        grad=np.array([(g*x).sum(),g.sum()]);rounds.append({'step':step,'theta':theta.tolist(),'samples':each,'loss':loss,'gradient':grad.tolist(),'next_theta':(theta-.6*grad).tolist() if step<2 else None});theta=theta-.6*grad
    return rounds

def cnn_reference(params,x,y,q=1.):
    """Explicit NumPy forward + weight/input reverse for conv3x3/ReLU/conv1x1."""
    k,b,v,c=[finite(a,'parameter') for a in params];x=finite(x,'x',4);y=finite(y,'y',4)
    if x.shape!=y.shape or x.shape[1]!=1 or k.ndim!=4 or k.shape[1:]!=(1,3,3) or b.shape!=(len(k),) or v.shape!=(1,len(k),1,1) or c.shape!=(1,):raise ValueError('unsupported CNN dimensions')
    xp=np.pad(x,((0,0),(0,0),(1,1),(1,1)));patch=np.lib.stride_tricks.sliding_window_view(xp,(3,3),axis=(2,3));z=np.einsum('ncrsab,ocab->nors',patch,k)+b[None,:,None,None];h=np.maximum(z,0);out=np.einsum('nohw,o->nhw',h,v.ravel())[:,None]+c.reshape(1,1,1,1);loss,g=bce(out,y,q)
    dv=np.einsum('nhw,nohw->o',g[:,0],h).reshape(v.shape);dc=np.array([g.sum()]);dz=g*v.ravel()[None,:,None,None]*(z>0);dk=np.einsum('nors,ncrsab->ocab',dz,patch);db=dz.sum((0,2,3));dxp=np.zeros_like(xp)
    for a in range(3):
        for d in range(3): dxp[:,:,a:a+x.shape[2],d:d+x.shape[3]]+=np.einsum('nohw,oc->nchw',dz,k[:,:,a,d])
    return loss,[dk,db,dv,dc],dxp[:,:,1:-1,1:-1],out

def encode_boxes(anchor,target):
    a=boxes(anchor);b=boxes(target)
    if a.shape!=b.shape:raise ValueError('anchor and target shapes must agree')
    aw=a[:,2:]-a[:,:2];bw=b[:,2:]-b[:,:2];ac=(a[:,2:]+a[:,:2])/2;bc=(b[:,2:]+b[:,:2])/2
    return finite(np.concatenate([(bc-ac)/aw,np.log(bw/aw)],axis=1),'encoded deltas',2)

def decode_boxes(anchor,delta):
    a=boxes(anchor);t=finite(delta,'deltas',2)
    if a.shape!=t.shape or np.any(np.abs(t[:,2:])>20):raise ValueError('delta shape mismatch or |log-scale|>20')
    aw=a[:,2:]-a[:,:2];ac=(a[:,2:]+a[:,:2])/2;bc=ac+t[:,:2]*aw;bw=aw*np.exp(t[:,2:]);out=np.concatenate([bc-bw/2,bc+bw/2],axis=1)
    return boxes(out)
