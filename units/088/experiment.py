"""DL088: CPU NumPy dual encoders, explicit gradients and honest retrieval.
Run: OPENBLAS_NUM_THREADS=1 python experiment.py
Synthetic data only. No external weights, API, or files are required.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dl088-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import csv, hashlib, json, platform, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
SEEDS = [11, 23, 37]
METHODS = ['image_to_text', 'symmetric', 'semantic_multi_positive']
COLORS = ['red', 'green', 'blue']
SHAPES = ['horizontal', 'vertical', 'square']

def make_data(per_class, seed):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(18), per_class)
    rng.shuffle(labels)
    images = rng.normal(0, .055, (len(labels), 8, 8, 3))
    text = np.zeros((len(labels), 10))
    captions = []
    for i, c in enumerate(labels):
        color, shape, side = int(c // 6), int(c % 6 // 2), int(c % 2)
        x, y = (2 if side == 0 else 5), 4
        if shape == 0: pixels = [(y, x-1), (y, x), (y, x+1)]
        elif shape == 1: pixels = [(y-1, x), (y, x), (y+1, x)]
        else: pixels = [(y-1,x-1),(y-1,x),(y,x-1),(y,x)]
        brightness = rng.uniform(.72, 1.0)
        for r,k in pixels: images[i,r,k,color] += brightness
        style = int(rng.integers(2))
        text[i, color] = text[i, 3+shape] = text[i, 6+side] = text[i, 8+style] = 1
        captions.append(('a ' if style == 0 else 'there is a ') + COLORS[color]+' '+SHAPES[shape]+' on the '+['left','right'][side])
    return {'x':images.reshape(len(labels),-1), 't':text, 'y':labels, 'captions':np.array(captions)}

def normalize(h):
    norm = np.sqrt(np.sum(h*h, axis=1, keepdims=True)+1e-12)
    return h/norm, norm

def norm_backward(g, z, norm):
    return (g-z*np.sum(g*z, axis=1, keepdims=True))/norm

def softmax(a, axis=1):
    z = a-a.max(axis=axis,keepdims=True)
    z = np.exp(z)
    return z/z.sum(axis=axis,keepdims=True)

def objective(params, x, t, y, method, tau=.2, gradients=True):
    # Bx192 -> Bx16; Bx10 -> Bx16. Both towers have learned weights.
    u, nu = normalize(x@params['wi'])
    v, nv = normalize(t@params['wt'])
    s = u@v.T/tau                 # BxB, image rows and text columns
    b = len(x)
    target = np.eye(b)
    if method == 'semantic_multi_positive':
        target = (y[:,None] == y[None,:]).astype(float)
        target /= target.sum(axis=1,keepdims=True)
    p = softmax(s, 1)
    loss_i = -np.sum(target*np.log(np.maximum(p,1e-300)))/b
    ds = (p-target)/b
    loss = loss_i
    if method != 'image_to_text':
        q = softmax(s, 0)
        # Targets must be normalized along the corresponding candidate axis.
        target_t = target.T
        loss_t = -np.sum(target_t*np.log(np.maximum(q,1e-300)))/b
        loss = .5*(loss_i+loss_t)
        ds = .5*(ds+(q-target_t)/b)
    if not gradients: return float(loss)
    gu, gv = ds@v/tau, ds.T@u/tau
    grads = {'wi':x.T@norm_backward(gu,u,nu), 'wt':t.T@norm_backward(gv,v,nv)}
    return float(loss), grads

def initial(seed):
    rng = np.random.default_rng(seed)
    return {'wi':rng.normal(0,.08,(192,16)), 'wt':rng.normal(0,.12,(10,16))}

def train(data, method, seed, tau=.2, steps=240):
    p = initial(seed)
    m = {k:np.zeros_like(v) for k,v in p.items()}
    v = {k:np.zeros_like(z) for k,z in p.items()}
    rng = np.random.default_rng(seed+1000)
    history=[]
    start=time.perf_counter()
    for step in range(1,steps+1):
        idx=rng.choice(len(data['y']),size=72,replace=False)
        loss,g=objective(p,data['x'][idx],data['t'][idx],data['y'][idx],method,tau)
        for k in p:
            m[k]=.9*m[k]+.1*g[k]
            v[k]=.999*v[k]+.001*g[k]**2
            p[k]-=.012*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        if step==1 or step%10==0: history.append({'step':step,'loss':loss})
    return p,history,time.perf_counter()-start

def scores(p,data):
    return normalize(data['x']@p['wi'])[0]@normalize(data['t']@p['wt'])[0].T

def retrieval(s,y):
    # Stable index tie break is declared. Pair IDs are the common row index.
    result={}
    for direction,a in [('i2t',s),('t2i',s.T)]:
        order=np.argsort(-a,axis=1,kind='stable')
        pair_rank=np.argmax(order==np.arange(len(y))[:,None],axis=1)+1
        semantic=(y[order]==y[:,None])
        semantic_rank=np.argmax(semantic,axis=1)+1
        for k in [1,5]:
            result[f'{direction}_pair_r{k}']=float(np.mean(pair_rank<=k))
            result[f'{direction}_semantic_r{k}']=float(np.mean(semantic_rank<=k))
        result[f'{direction}_pair_mrr']=float(np.mean(1/pair_rank))
    return result

def finite_difference():
    d=make_data(1,17); p=initial(29); x,t,y=d['x'][:7],d['t'][:7],d['y'][:7]
    # Deliberately duplicate one semantic label to check multi-positive derivatives.
    y=y.copy(); y[1]=y[0]
    errors={}
    for method in METHODS:
        _,g=objective(p,x,t,y,method)
        diffs=[]
        for k,ij in [('wi',(8,3)),('wi',(104,7)),('wt',(3,2)),('wt',(8,1))]:
            old=p[k][ij]; eps=1e-5
            p[k][ij]=old+eps; plus=objective(p,x,t,y,method,gradients=False)
            p[k][ij]=old-eps; minus=objective(p,x,t,y,method,gradients=False)
            p[k][ij]=old
            diffs.append(abs((plus-minus)/(2*eps)-g[k][ij]))
        errors[method]=max(diffs)
    return errors

def audit(train_data,test_data):
    hashes=lambda x:{hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in x}
    y=train_data['y']; same=y[:,None]==y[None,:]
    fn=int(same.sum()-len(y)); total=len(y)*(len(y)-1)
    tx=hashes(train_data['x']); vx=hashes(test_data['x'])
    # Nearest-image cosine across splits is a diagnostic, never a leakage verdict.
    sim=normalize(test_data['x'])[0]@normalize(train_data['x'])[0].T
    return {'exact_image_overlap':len(tx&vx),'unique_train_captions':len(set(train_data['captions'])),
      'caption_overlap_across_splits':len(set(train_data['captions'])&set(test_data['captions'])),
      'off_diagonal_semantic_positive_count':fn,'off_diagonal_count':total,
      'semantic_false_negative_fraction':fn/total,
      'cross_split_max_image_cosine':float(sim.max()),'cross_split_mean_nearest_image_cosine':float(sim.max(axis=1).mean()),
      'meaning':'Caption reuse is expected: independent image instances share semantic labels. No instance-ID signal is supplied.'}

def plot_all(train_data,test_data,records,histories,p_example):
    out=ROOT/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,3.4)); ax.axis('off')
    boxes=[(.02,.58,'Images\nB x 192'),(.26,.58,'Image tower\n192 x 16'),(.51,.58,'Unit vectors U\nB x 16'),(.02,.1,'Text attributes\nB x 10'),(.26,.1,'Text tower\n10 x 16'),(.51,.1,'Unit vectors V\nB x 16'),(.77,.34,'S = U V^T / tau\nB x B')]
    for x,y,txt in boxes: ax.text(x,y,txt,bbox={'boxstyle':'round,pad=.6','fc':'#e3f2fd','ec':'#315d90'},va='center')
    for a,b in [((.18,.58),(.25,.58)),((.43,.58),(.50,.58)),((.18,.1),(.25,.1)),((.43,.1),(.50,.1)),((.68,.58),(.77,.4)),((.68,.1),(.77,.27))]: ax.annotate('',b,a,arrowprops={'arrowstyle':'->','lw':2})
    fig.tight_layout(); fig.savefig(out/'01_dual_encoder.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(10,5))
    for i,ax in enumerate(axs.flat):
        ax.imshow(np.clip(test_data['x'][i].reshape(8,8,3),0,1)); ax.axis('off')
        c=int(test_data['y'][i]); ax.set_title(f'{COLORS[c//6]} {SHAPES[c%6//2]}\n'+['left','right'][c%2])
    fig.suptitle('Independent synthetic image instances (8 x 8 RGB)'); fig.tight_layout(); fig.savefig(out/'02_pairs.png',dpi=160); plt.close(fig)
    s=scores(p_example,test_data); idx=np.argsort(test_data['y'],kind='stable')[:72]
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(s[np.ix_(idx,idx)],vmin=-1,vmax=1,cmap='coolwarm'); fig.colorbar(im,ax=ax,label='cosine similarity')
    ax.set(xlabel='Text candidates (semantic-group order)',ylabel='Image queries',title='Semantic blocks, not unique instance matches')
    fig.tight_layout();fig.savefig(out/'03_similarity.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for method in METHODS:
        h=histories[f'{method}_11']; axs[0].plot([x['step'] for x in h],[x['loss'] for x in h],label=method)
    axs[0].set(xlabel='Updates',ylabel='Minibatch loss',title='Loss objectives have different interpretations');axs[0].legend(fontsize=7)
    for j,key in enumerate(['i2t_pair_r1','i2t_semantic_r1','t2i_pair_r1','t2i_semantic_r1']):
        means=[np.mean([r[key] for r in records if r['method']==m]) for m in METHODS]
        sd=[np.std([r[key] for r in records if r['method']==m],ddof=1) for m in METHODS]
        axs[1].bar(np.arange(3)+j*.19,means,.18,yerr=sd,label=key)
    axs[1].set(xticks=np.arange(3)+.28,xticklabels=['one-way','two-way','multi-positive'],ylabel='Recall@1',ylim=(0,1.12),title='All 3 seeds; error bars = sample SD');axs[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'04_training_retrieval.png',dpi=160);plt.close(fig)
    logits=np.array([1.,.8,.1])
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for i,tau in enumerate([.05,.2,1.]):axs[0].bar(np.arange(3)+i*.25,softmax((logits/tau)[None])[0],.24,label=f'tau={tau}')
    axs[0].set(xticks=[.25,1.25,2.25],xticklabels=['paired','same meaning','different'],ylabel='Row probability');axs[0].legend()
    gx=np.array([0,.1,.3,.5]); axs[1].plot(gx,np.log(1+gx*71),'o-');axs[1].set(xlabel='Hypothetical false-negative rate',ylabel='log(1 + 71 * rate)',title='Illustrative ambiguity scale, not measured loss')
    fig.tight_layout();fig.savefig(out/'05_temperature.png',dpi=160);plt.close(fig)

def main():
    started=time.perf_counter(); (ROOT/'outputs').mkdir(exist_ok=True)
    train_d=make_data(24,8801); val_d=make_data(6,8802); test_d=make_data(12,8803)
    np.savez_compressed(ROOT/'outputs/data.npz',**{f'{split}_{k}':v for split,d in [('train',train_d),('val',val_d),('test',test_d)] for k,v in d.items()})
    grad=finite_difference()
    if max(grad.values())>1e-6:raise RuntimeError(f'Gradient check failed: {grad}')
    records=[]; histories={}; predictions=[]; sample=None
    for method in METHODS:
        for seed in SEEDS:
            p,h,seconds=train(train_d,method,seed)
            s=scores(p,test_d); r={'method':method,'seed':seed,'tau':.2,'steps':240,'batch':72,'parameters':3232,'seconds':seconds,**retrieval(s,test_d['y'])}
            r['validation_semantic_r1']=retrieval(scores(p,val_d),val_d['y'])['i2t_semantic_r1']
            path=ROOT/f'outputs/{method}_seed{seed}.npz';np.savez_compressed(path,**p)
            with np.load(path) as z: restored={k:z[k] for k in p}
            r['reload_max_error']=float(np.max(np.abs(scores(restored,test_d)-s)))
            records.append(r);histories[f'{method}_{seed}']=h
            for direction,a in [('i2t',s),('t2i',s.T)]:
                top=np.argsort(-a,axis=1,kind='stable')[:,0]
                for q,k in enumerate(top):predictions.append({'method':method,'seed':seed,'direction':direction,'query_id':q,'query_class':int(test_d['y'][q]),'top_id':int(k),'top_class':int(test_d['y'][k]),'pair_correct':int(q==k),'semantic_correct':int(test_d['y'][q]==test_d['y'][k])})
            if method=='symmetric' and seed==11:sample=p
    temperatures=[]
    for tau in [.05,1.]:
        p,h,seconds=train(train_d,'symmetric',11,tau)
        temperatures.append({'tau':tau,'seed':11,'method':'symmetric',**retrieval(scores(p,test_d),test_d['y']),'seconds':seconds})
    plot_all(train_d,test_d,records,histories,sample)
    result={'unit':'088','environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},'protocol':{'train':432,'validation':108,'test':216,'data_seeds':[8801,8802,8803],'train_seeds':SEEDS,'primary_methods':METHODS,'steps':240,'batch':72,'tau':.2,'learning_rate':.012,'optimizer':'Adam, hand-written gradients','test_used_for_selection':False,'gallery':'216 images / 216 paired captions; same 18 classes; stable index tie break','temperature_ablation':'seed11 only, exploratory diagnostic, not used to choose primary tau'},'gradient_max_absolute_errors':grad,'audit':audit(train_d,test_d),'records':records,'temperature_ablation':temperatures,'training_history':histories,'total_seconds':time.perf_counter()-started}
    (ROOT/'outputs/results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for filename,rows in [('metrics.csv',records),('predictions.csv',predictions)]:
        with (ROOT/'outputs'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'audit':result['audit'],'gradient_check':grad,'mean_recall':{m:{k:float(np.mean([r[k] for r in records if r['method']==m])) for k in ['i2t_pair_r1','i2t_semantic_r1','t2i_pair_r1','t2i_semantic_r1']} for m in METHODS},'total_seconds':result['total_seconds']},indent=2))
    return result

if __name__=='__main__':main()
