"""Independent scalar/NumPy references. Never import torch."""
import math
import numpy as np

def hand_round(theta):
    theta=np.asarray(theta,dtype=np.float64)
    if theta.shape!=(4,) or not np.isfinite(theta).all() or np.max(np.abs(theta))>100:raise ValueError('four finite parameters in [-100,100]')
    w,b,a,c=theta;rows=[]
    for x,t in [([0.,1.],0),([1.,2.],1)]:
        h=[max(0,w*q+b) for q in x];z=[w*q+b for q in x];m=sum(h)/2;logit=a*m+c;e=math.exp(-abs(logit));p=1/(1+e) if logit>=0 else e/(1+e);loss=max(logit,0)-logit*t+math.log1p(math.exp(-abs(logit)))
        g=(p-t)/2;dh=g*a/2;dz=[dh*(v>0) for v in z];grad=[sum(v*q for v,q in zip(dz,x)),sum(dz),g*m,g]
        rows.append(dict(x=x,y=t,z=z,h=h,m=m,logit=logit,p=p,loss=loss,dlogit=g,dh=[dh,dh],dz=dz,gradient=grad))
    gradient=np.sum([r['gradient'] for r in rows],axis=0)
    return dict(theta=theta.tolist(),rows=rows,loss=sum(r['loss'] for r in rows)/2,gradient=gradient.tolist(),next_theta=(theta-.2*gradient).tolist())
def hand_ledger():
    first=hand_round([.5,.2,.8,-.1]);second=hand_round(first['next_theta']);third=hand_round(second['next_theta'])
    return {'parameter_order':['w','b','a','c'],'learning_rate':.2,'rounds':[first,second,third]}
def metrics(logits,y):
    logits=np.asarray(logits);y=np.asarray(y)
    if logits.ndim!=2 or logits.shape[1]!=2 or len(logits)==0 or logits.dtype.kind!='f' or not np.isfinite(logits).all() or np.max(np.abs(logits))>1e6:raise ValueError('finite N-by-2 floating logits, magnitude <= 1e6')
    if y.shape!=(len(logits),) or y.dtype.kind not in 'iu' or np.any((y<0)|(y>1)):raise ValueError('integer labels 0 or 1, shape N')
    z=logits.astype(np.float64);z=z-z.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True));p=np.exp(logp);pred=z.argmax(1)
    return {'loss':float(-logp[np.arange(len(y)),y].mean()),'accuracy':float((pred==y).mean()),'probabilities':p,'pred':pred}
def conv_relu_pool_reference(x,w,b):
    """3x3 zero-padded cross-correlation + ReLU + 2x2 average pooling."""
    x=np.asarray(x,dtype=float);w=np.asarray(w,dtype=float);b=np.asarray(b,dtype=float)
    n,ci,h,ww=x.shape;co=w.shape[0];p=np.pad(x,((0,0),(0,0),(1,1),(1,1)));z=np.zeros((n,co,h,ww))
    for i in range(h):
        for j in range(ww):
            for o in range(co):z[:,o,i,j]=(p[:,:,i:i+3,j:j+3]*w[o]).sum((1,2,3))+b[o]
    a=np.maximum(z,0);return a.reshape(n,co,h//2,2,ww//2,2).mean((3,5))
