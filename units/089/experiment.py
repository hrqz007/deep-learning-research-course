"""DL089: controlled VQA with hand-differentiated cross-attention, CPU NumPy.
Toy answer classifier, NOT a reproduction of a large pretrained VLM.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl089-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import csv,json,platform,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
SEEDS=[11,23,37]
METHODS=['text_only','image_only','concat_biased','attention_biased','attention_random_hint']
DOMAINS=['clean','conflict_hint','blank_hint','target_occluded','target_occluded_blank_hint','other_occluded','image_blank','question_flip']

def make_data(n,seed):
    rng=np.random.default_rng(seed)
    # All 8 combinations of two colors and requested side, balanced exactly.
    c=np.resize(np.arange(8),n);rng.shuffle(c)
    colors=np.column_stack([c//4,(c//2)%2]); side=c%2
    y=colors[np.arange(n),side]
    hint=np.where(rng.random(n)<.95,y,1-y)
    q=np.eye(2)[side]; h=np.eye(2)[hint]
    images=rng.normal(0,.16,(n,8,8,3))
    for i in range(n):
        for j,left in enumerate([0,4]):
            channel=0 if colors[i,j]==0 else 2
            images[i,2:5,left+1:left+4,channel]+=rng.uniform(.75,1.0)
    return {'images':images,'q':q,'h':h,'y':y,'colors':colors,'side':side,'hint':hint}

def tokenize(images):
    # A deliberately simple fixed visual encoder: spatial pooling of each half.
    # Nx8x8x3 -> Nx2x3 RGB, with fixed 2-dimensional position appended.
    rgb=np.stack([images[:,:,:4,:].mean(axis=(1,2)),images[:,:,4:,:].mean(axis=(1,2))],axis=1)*(32/9)
    pos=np.broadcast_to(np.eye(2),(len(images),2,2))
    return np.concatenate([rgb,pos],axis=2)

def intervene(data,domain):
    x={k:v.copy() for k,v in data.items()}
    if domain=='conflict_hint':x['h']=np.eye(2)[1-data['y']]
    elif domain=='blank_hint':x['h'][:]=0
    elif domain=='image_blank':x['images'][:]=0
    elif domain in ['target_occluded','target_occluded_blank_hint','other_occluded']:
        for i,side in enumerate(data['side']):
            s=int(side if domain.startswith('target_occluded') else 1-side)
            x['images'][i,:,s*4:(s+1)*4,:]=0
        if domain=='target_occluded_blank_hint':x['h'][:]=0
    elif domain=='question_flip':
        x['q']=data['q'][:,::-1].copy();x['side']=1-data['side']
        x['y']=data['colors'][np.arange(len(data['y'])),x['side']]
        # Keep the old hint to create a paired counterfactual, explicitly declared.
    elif domain!='clean':raise ValueError(domain)
    return x

def softmax(x):
    z=np.exp(x-x.max(axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)

def initial(method,seed):
    rng=np.random.default_rng(seed)
    inp=4 if method=='text_only' else 10 if method=='image_only' else 14 if method=='concat_biased' else 10
    p={'w1':rng.normal(0,.2,(inp,16)),'b1':np.zeros(16),'w2':rng.normal(0,.2,(16,2)),'b2':np.zeros(2)}
    if method.startswith('attention'):
        p.update({'wq':rng.normal(0,.3,(2,4)),'wk':rng.normal(0,.3,(5,4)),'wv':rng.normal(0,.3,(5,6))})
    return p

def forward(p,tokens,q,h,method):
    cache={}
    if method=='text_only': z=np.concatenate([q,h],axis=1)
    elif method=='image_only':z=tokens.reshape(len(q),-1)
    elif method=='concat_biased':z=np.concatenate([tokens.reshape(len(q),-1),q,h],axis=1)
    else:
        query=q@p['wq'];key=tokens@p['wk'];value=tokens@p['wv']
        a=softmax(np.einsum('nd,nkd->nk',query,key)/2) # sqrt(d)=2
        r=np.einsum('nk,nkv->nv',a,value)
        z=np.concatenate([r,q,h],axis=1)
        cache.update(query=query,key=key,value=value,a=a)
    hidden=np.tanh(z@p['w1']+p['b1']); logits=hidden@p['w2']+p['b2']
    cache.update(z=z,hidden=hidden)
    return logits,cache

def objective(p,tokens,q,h,y,method,gradients=True):
    logits,c=forward(p,tokens,q,h,method);prob=softmax(logits)
    loss=-np.log(np.maximum(prob[np.arange(len(y)),y],1e-300)).mean()
    if not gradients:return float(loss)
    dl=prob.copy();dl[np.arange(len(y)),y]-=1;dl/=len(y)
    dh=(dl@p['w2'].T)*(1-c['hidden']**2)
    g={'w2':c['hidden'].T@dl,'b2':dl.sum(axis=0),'w1':c['z'].T@dh,'b1':dh.sum(axis=0)}
    if method.startswith('attention'):
        dr=(dh@p['w1'].T)[:,:6]
        da=np.einsum('nv,nkv->nk',dr,c['value'])
        dv=c['a'][:,:,None]*dr[:,None,:]
        ds=c['a']*(da-(da*c['a']).sum(axis=1,keepdims=True))
        dq=np.einsum('nk,nkd->nd',ds,c['key'])/2
        dk=ds[:,:,None]*c['query'][:,None,:]/2
        g.update(wq=q.T@dq,wk=np.einsum('nki,nkd->id',tokens,dk),wv=np.einsum('nki,nkv->iv',tokens,dv))
    return float(loss),g

def train(data,method,seed,steps=300):
    p=initial(method,seed);m={k:np.zeros_like(v) for k,v in p.items()};v={k:np.zeros_like(z) for k,z in p.items()}
    rng=np.random.default_rng(seed+2000);hint_rng=np.random.default_rng(seed+3000);tokens=tokenize(data['images']);history=[];start=time.perf_counter()
    for step in range(1,steps+1):
        ids=rng.choice(len(data['y']),96,replace=False);h=data['h'][ids]
        if method=='attention_random_hint':h=np.eye(2)[hint_rng.integers(2,size=len(ids))]
        loss,g=objective(p,tokens[ids],data['q'][ids],h,data['y'][ids],method)
        for k in p:
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2
            p[k]-=.01*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        if step==1 or step%10==0:history.append({'step':step,'loss':loss})
    return p,history,time.perf_counter()-start

def predict(p,data,method):
    logits,c=forward(p,tokenize(data['images']),data['q'],data['h'],method)
    return softmax(logits),c

def gradient_check():
    d=make_data(16,91);t=tokenize(d['images']);method='attention_biased';p=initial(method,43)
    _,g=objective(p,t,d['q'],d['h'],d['y'],method)
    result={}
    for name in p:
        ij=tuple(0 for _ in p[name].shape);old=p[name][ij];eps=1e-5
        p[name][ij]=old+eps;a=objective(p,t,d['q'],d['h'],d['y'],method,False)
        p[name][ij]=old-eps;b=objective(p,t,d['q'],d['h'],d['y'],method,False)
        p[name][ij]=old;result[name]=abs((a-b)/(2*eps)-g[name][ij])
    return result

def make_figures(test,records,histories,example):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,4,figsize=(11,3))
    for ax,dom in zip(axs,['clean','conflict_hint','target_occluded','other_occluded']):
        d=intervene(test,dom);i=0;ax.imshow(np.clip(d['images'][i],0,1));ax.axis('off')
        ax.set_title(dom+'\nAsk: '+['left','right'][d['side'][i]]+'; hint: '+['red','blue'][d['h'][i].argmax()])
    fig.suptitle('Same image/question base; one controlled change');fig.tight_layout();fig.savefig(out/'01_interventions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4));ax.axis('off')
    boxes=[(.02,.7,'8 x 8 RGB\nfixed spatial pooling'),(.28,.7,'2 visual tokens\nV: 2 x 5'),(.28,.2,'Question: 1 x 2\nQ = question Wq'),(.55,.7,'K = V Wk: 2 x 4\nValue = V Wv: 2 x 6'),(.57,.2,'softmax(Q K^T / 2)\n1 x 2 attention'),(.81,.45,'Weighted value\n1 x 6\n+ question + hint\nMLP -> 2 answers')]
    for x,y,s in boxes:ax.text(x,y,s,bbox={'boxstyle':'round,pad=.55','fc':'#e8f5e9','ec':'#336d51'},va='center',fontsize=9)
    for a,b in [((.22,.7),(.27,.7)),((.47,.7),(.54,.7)),((.48,.2),(.56,.2)),((.67,.58),(.67,.32)),((.77,.2),(.84,.32)),((.77,.7),(.84,.59))]:ax.annotate('',b,a,arrowprops={'arrowstyle':'->','lw':2})
    fig.tight_layout();fig.savefig(out/'02_fusion_shapes.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    mat=np.array([[np.mean([r['accuracy'] for r in records if r['method']==m and r['domain']==d]) for d in DOMAINS] for m in METHODS])
    im=axs[0].imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    axs[0].set(xticks=np.arange(len(DOMAINS)),xticklabels=DOMAINS,yticks=np.arange(len(METHODS)),yticklabels=METHODS,title='Mean accuracy, three training seeds');axs[0].tick_params(axis='x',rotation=65,labelsize=8);axs[0].tick_params(axis='y',labelsize=8)
    for i in range(len(METHODS)):
        for j in range(len(DOMAINS)):axs[0].text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>.8 else 'black',fontsize=8)
    fig.colorbar(im,ax=axs[0],fraction=.03)
    for m in METHODS:
        h=histories[f'{m}_11'];axs[1].plot([x['step'] for x in h],[x['loss'] for x in h],label=m)
    axs[1].set(xlabel='Updates',ylabel='Training loss',title='Different models may fit the easy hint');axs[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'03_accuracy_training.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for m in ['attention_biased','attention_random_hint']:
        p=example[m];prob,c=predict(p,test,m)
        mass=c['a'][np.arange(len(test['y'])),test['side']]
        axs[0].hist(mass,bins=np.linspace(0,1,21),alpha=.5,label=m)
    axs[0].set(xlabel='Attention mass on requested side',ylabel='Count',title='Attention is a diagnostic, not proof');axs[0].legend(fontsize=7)
    n=np.array([16,64,256,576]);l=128
    axs[1].plot(n,l*n,'o-',label='Cross attention L*N');axs[1].plot(n,(l+n)**2,'s-',label='Unified full attention (L+N)^2');axs[1].set(xlabel='Visual tokens N',ylabel='Attention score elements / head',yscale='log');axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'04_attention_budget.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for j,dom in enumerate(['conflict_hint','target_occluded','other_occluded']):
        means=[np.mean([r['accuracy'] for r in records if r['method']==m and r['domain']==dom]) for m in METHODS]
        sd=[np.std([r['accuracy'] for r in records if r['method']==m and r['domain']==dom],ddof=1) for m in METHODS]
        ax.bar(np.arange(len(METHODS))+.24*j,means,.23,yerr=sd,label=dom)
    ax.set(xticks=np.arange(len(METHODS))+.24,xticklabels=['text','image','concat','attn biased','attn random'],ylim=(0,1.08),ylabel='Accuracy',title='Occluding evidence and contradicting hints ask different questions');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'05_ablation.png',dpi=160);plt.close(fig)

def main():
    started=time.perf_counter();(ROOT/'outputs').mkdir(exist_ok=True)
    train_d=make_data(960,8901);val_d=make_data(240,8902);test_d=make_data(480,8903)
    np.savez_compressed(ROOT/'outputs/data.npz',**{f'{s}_{k}':v for s,d in [('train',train_d),('val',val_d),('test',test_d)] for k,v in d.items()})
    grad=gradient_check()
    if max(grad.values())>1e-6:raise RuntimeError(f'Gradient check failed: {grad}')
    records=[];predictions=[];histories={};example={};reload_errors=[]
    for method in METHODS:
        for seed in SEEDS:
            p,h,seconds=train(train_d,method,seed);histories[f'{method}_{seed}']=h
            params=sum(x.size for x in p.values());path=ROOT/f'outputs/{method}_seed{seed}.npz';np.savez_compressed(path,**p)
            with np.load(path) as z:restored={k:z[k] for k in p}
            reload_errors.append(float(np.max(np.abs(predict(restored,test_d,method)[0]-predict(p,test_d,method)[0]))))
            vprob,_=predict(p,val_d,method);valacc=float((vprob.argmax(axis=1)==val_d['y']).mean())
            if seed==11 and method.startswith('attention'):example[method]=p
            base_pred=predict(p,test_d,method)[0].argmax(axis=1)
            for domain in DOMAINS:
                d=intervene(test_d,domain);prob,c=predict(p,d,method);yp=prob.argmax(axis=1)
                r={'method':method,'seed':seed,'domain':domain,'accuracy':float((yp==d['y']).mean()),'nll':float(-np.log(prob[np.arange(len(yp)),d['y']]+1e-300).mean()),'prediction_change_vs_clean':float((yp!=base_pred).mean()),'parameters':params,'train_seconds':seconds,'validation_clean_accuracy':valacc}
                if method.startswith('attention'):r['attention_target_mass']=float(c['a'][np.arange(len(yp)),d['side']].mean())
                else:r['attention_target_mass']=None
                records.append(r)
                for i in range(len(yp)):predictions.append({'method':method,'seed':seed,'domain':domain,'sample_id':i,'target':int(d['y'][i]),'prediction':int(yp[i]),'p_red':float(prob[i,0]),'p_blue':float(prob[i,1])})
    make_figures(test_d,records,histories,example)
    result={'unit':'089','environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},'protocol':{'data_seeds':[8901,8902,8903],'train':960,'validation':240,'test':480,'seeds':SEEDS,'methods':METHODS,'domains':DOMAINS,'input_shape':[8,8,3],'visual_token_count':2,'visual_token_width':5,'question_token_count':1,'hint_token_count':1,'attention_key_width':4,'attention_value_width':6,'answer_count':2,'steps':300,'batch':96,'learning_rate':.01,'train_hint_agreement':float(np.mean(train_d['hint']==train_d['y'])),'test_hint_agreement':float(np.mean(test_d['hint']==test_d['y'])),'selection':'No test selection or early stopping; all variants/seeds reported','scope':'Fixed spatial pooling and 2-class answer heads; no pretrained language model; not LLaVA/Flamingo reproduction'},'gradient_max_absolute_errors':grad,'reload_max_error':max(reload_errors),'records':records,'training_history':histories,'total_seconds':time.perf_counter()-started}
    (ROOT/'outputs/results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for filename,rows in [('metrics.csv',records),('predictions.csv',predictions)]:
        with (ROOT/'outputs'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'mean_accuracy':{m:{d:float(np.mean([r['accuracy'] for r in records if r['method']==m and r['domain']==d])) for d in DOMAINS} for m in METHODS},'gradient_check':grad,'reload_error':max(reload_errors),'total_seconds':result['total_seconds']},indent=2))
    return result

if __name__=='__main__':main()
