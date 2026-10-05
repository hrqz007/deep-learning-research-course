"""DL054: target construction is part of the probability model.

Uses a small untrained Transformer to probe packing equivalence, not task quality.
"""
from pathlib import Path
import argparse,json,math,platform,hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from generate_data import VOCAB,corpus,main as generate
ROOT=Path(__file__).resolve().parent
PAD,BOS,EOS,MASK=0,1,2,3
IGNORE=-100

def setup(seed=5401):
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)

def check_documents(documents):
    if not documents:raise ValueError('at least one document required')
    for doc in documents:
        if not doc or any(type(t) is not int or not 4<=t<len(VOCAB) for t in doc):
            raise ValueError('each document must contain normal vocabulary IDs, not special tokens')

def ar_padded(documents):
    """Input BOS+w1...wn predicts w1...wn+EOS, independently for each document."""
    check_documents(documents);b=len(documents);length=max(map(len,documents))+1
    x=torch.full((b,length),PAD,dtype=torch.long);y=torch.full_like(x,IGNORE)
    positions=torch.zeros_like(x);valid=torch.zeros_like(x,dtype=torch.bool)
    allow=torch.zeros((b,length,length),dtype=torch.bool)
    for i,doc in enumerate(documents):
        n=len(doc)+1;x[i,:n]=torch.tensor([BOS]+doc);y[i,:n]=torch.tensor(doc+[EOS])
        valid[i,:n]=True;positions[i,:n]=torch.arange(n)
        allow[i,:n,:n]=torch.ones(n,n,dtype=torch.bool).tril()
        # Ignored padding queries attend only themselves to avoid all-masked softmax.
        for j in range(n,length):allow[i,j,j]=True
    return {'x':x,'y':y,'position':positions,'valid':valid,'allow':allow}

def ar_packed(documents):
    """No truncation: concatenate pre-shifted samples; reset positions and mask segments."""
    check_documents(documents);xs=[];ys=[];segments=[];positions=[]
    for i,doc in enumerate(documents):
        n=len(doc)+1;xs.extend([BOS]+doc);ys.extend(doc+[EOS]);segments.extend([i]*n);positions.extend(range(n))
    x=torch.tensor([xs]);y=torch.tensor([ys]);seg=torch.tensor(segments);length=len(xs)
    allow=(seg[:,None]==seg[None,:]) & torch.ones(length,length,dtype=torch.bool).tril()
    return {'x':x,'y':y,'position':torch.tensor([positions]),'valid':torch.ones_like(x,dtype=torch.bool),
            'allow':allow[None],'segment':seg}

def mlm_padded(documents,seed=5410,rate=.3):
    """All selected normal tokens become MASK; simplified MLM, not BERT80/10/10."""
    check_documents(documents)
    if not 0<rate<=1:raise ValueError('mask rate in (0,1] required')
    gen=torch.Generator().manual_seed(seed);length=max(map(len,documents))+2;b=len(documents)
    x=torch.full((b,length),PAD,dtype=torch.long);y=torch.full_like(x,IGNORE)
    pos=torch.zeros_like(x);valid=torch.zeros_like(x,dtype=torch.bool);allow=torch.zeros((b,length,length),dtype=torch.bool)
    selections=[]
    for i,doc in enumerate(documents):
        n=len(doc)+2;original=torch.tensor([BOS]+doc+[EOS]);x[i,:n]=original
        chosen=torch.rand(len(doc),generator=gen)<rate
        # Tiny texts can randomly select no token; force one, and disclose this policy.
        if not chosen.any():chosen[torch.randint(len(doc),(1,),generator=gen).item()]=True
        indices=chosen.nonzero().flatten()+1;y[i,indices]=original[indices];x[i,indices]=MASK
        valid[i,:n]=True;pos[i,:n]=torch.arange(n);allow[i,:n,:n]=True
        for j in range(n,length):allow[i,j,j]=True
        selections.append(indices.tolist())
    return {'x':x,'y':y,'position':pos,'valid':valid,'allow':allow,'selected':selections}

def masked_nll(logits,labels):
    """Return sum, count, mean. Select first, avoiding invalid ignored label gather."""
    if logits.ndim!=3 or labels.shape!=logits.shape[:2] or labels.dtype!=torch.long:
        raise ValueError('expected logits B,L,V and int64 labels B,L')
    keep=labels!=IGNORE
    if not keep.any():raise ValueError('zero effective targets; no defined mean loss')
    chosen=labels[keep]
    if (chosen<0).any() or (chosen>=logits.shape[-1]).any():raise ValueError('target out of vocabulary')
    selected_logits=logits[keep]
    if not torch.isfinite(selected_logits).all():raise ValueError('nonfinite effective logits')
    losses=-F.log_softmax(selected_logits,dim=-1).gather(1,chosen[:,None]).squeeze(1)
    return losses.sum(),int(keep.sum()),losses.mean()

class TinyProbe(nn.Module):
    """Two independent-initialized pre-norm layers; learned absolute positions, no dropout."""
    def __init__(self,d=12,heads=3,max_length=128):
        super().__init__();self.heads=heads;self.max_length=max_length
        self.embed=nn.Embedding(len(VOCAB),d,dtype=torch.float64)
        self.position=nn.Embedding(max_length,d,dtype=torch.float64)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,heads,2*d,dropout=0,
            batch_first=True,norm_first=True,dtype=torch.float64) for _ in range(2)])
        self.norm=nn.LayerNorm(d,dtype=torch.float64);self.head=nn.Linear(d,len(VOCAB),dtype=torch.float64)
    def forward(self,batch,allow=None,position=None):
        p=batch['position'] if position is None else position
        if p.min()<0 or p.max()>=self.max_length:raise ValueError('position exceeds embedding table')
        a=batch['allow'] if allow is None else allow
        if a.dtype!=torch.bool or not a.any(-1).all():raise ValueError('nonempty boolean allow required')
        h=self.embed(batch['x'])+self.position(p)
        # Official bool mask=True is disallowed; [B*H,L,L] repeats per sample/head.
        blocked=(~a)[:,None].expand(-1,self.heads,-1,-1).reshape(-1,a.shape[-2],a.shape[-1])
        for layer in self.layers:h=layer(h,src_mask=blocked)
        return self.head(self.norm(h))

def nll_ledger():
    """Three-class independent logits, two supervised positions and one ignored one."""
    logits=torch.tensor([[[math.log(2),0.,0.],[0.,math.log(3),0.],[9.,-4.,1.]]],dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([[0,1,IGNORE]]);records=[]
    for step in range(3):
        logits.grad=None;total,count,mean=masked_nll(logits,labels);mean.backward()
        records.append({'step':step,'logits':logits.detach().tolist(),'probabilities':logits.softmax(-1).detach().tolist(),
            'sum_nll':total.item(),'count':count,'mean_nll':mean.item(),'gradient':logits.grad.tolist()})
        if step<2:
            with torch.no_grad():logits.add_(logits.grad,alpha=-.2)
    return records

def serialize(batch):
    return {k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}

def run(output):
    setup();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    corpus_obj=generate(out/'data');documents=[r['tokens'] for r in corpus_obj['documents'] if r['split']=='train']
    padded=ar_padded(documents);packed=ar_packed(documents);mlm=mlm_padded(documents)
    model=TinyProbe();model.eval()
    # Keep autograd available here; gradient equivalence is tested separately.
    lp=model(padded);lk=model(packed);lm=model(mlm)
    ps,pc,pl=masked_nll(lp,padded['y']);ks,kc,kl=masked_nll(lk,packed['y']);ms,mc,ml=masked_nll(lm,mlm['y'])
    effective=lp[padded['valid']];packed_logits=lk[0]
    leaky=torch.ones_like(packed['allow']).tril()
    altered={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in packed.items()}
    first=packed['segment']==0;normal=first & (packed['x'][0]>=4)
    altered['x'][0,normal]=4+(packed['x'][0,normal]-4+1)%(len(VOCAB)-4)
    later=packed['segment']>0
    correct_change=(model(altered)[0,later]-lk[0,later]).abs().max().item()
    wrong_change=(model(altered,allow=leaky)[0,later]-model(packed,allow=leaky)[0,later]).abs().max().item()
    no_reset=torch.arange(packed['x'].shape[1])[None]
    no_reset_diff=(model(packed,position=no_reset)[0]-effective).abs().max().item()
    # Diagnostic copy logits: same-position labels are artificially easy even with causal attention.
    copied=torch.full((*padded['x'].shape,len(VOCAB)),-6.,dtype=torch.float64)
    copied.scatter_(-1,padded['x'][...,None],6.)
    wrong_labels=padded['x'].masked_fill(~padded['valid'],IGNORE)
    _,_,bad_loss=masked_nll(copied,wrong_labels);_,_,correct_loss=masked_nll(copied,padded['y'])
    # Hand example has two unequal-length groups of token losses.
    group_sums=[.2,12.];group_counts=[2,6]
    result={'hand_nll':nll_ledger(),'corpus_counts':{s:sum(r['split']==s for r in corpus_obj['documents']) for s in ['train','validation','test']},
       'duplicates':corpus_obj['duplicates'],'vocab_size':len(VOCAB),
       'ar':{'sum_nll':ps.item(),'count':pc,'mean_nll':pl.item(),'perplexity':math.exp(pl.item())},
       'packed':{'sum_nll':ks.item(),'count':kc,'mean_nll':kl.item()},
       'mlm':{'sum_nll':ms.item(),'count':mc,'mean_nll':ml.item(),'selected':mlm['selected']},
       'packing_max_logit_error':(effective-packed_logits).abs().max().item(),
       'packing_sum_nll_error':abs(ps.item()-ks.item()),
       'blocked_cross_document_change':correct_change,'leaky_cross_document_change':wrong_change,
       'missing_position_reset_max_error':no_reset_diff,
       'copy_diagnostic':{'wrong_unshifted_nll':bad_loss.item(),'correct_shifted_nll':correct_loss.item()},
       'aggregation':{'token_weighted':sum(group_sums)/sum(group_counts),'wrong_mean_of_means':sum(s/n for s,n in zip(group_sums,group_counts))/2},
       'batches':{'ar_padded':serialize(padded),'ar_packed':serialize(packed),'mlm':serialize(mlm)},
       'scope':'untrained structural probe; objective loss magnitudes do not rank objectives'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'dtype':'float64','device':'CPU','threads':1},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='batches'},ensure_ascii=False,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'outputs'));a=p.parse_args();run(a.output)
