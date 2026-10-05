"""DL049: deterministic byte-BPE, embedding scatter and boundary-safe bigram training.

Educational byte-BPE without language-specific pretokenization. It may merge spaces,
never merges across document boundaries and never merges special control IDs.
"""
from pathlib import Path
from fractions import Fraction as F
from numbers import Integral
from collections import Counter
import argparse,gzip,hashlib,json,math,os,platform,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PAD,BOS,EOS,UNK=0,1,2,3

def text_bytes(text):
    if not isinstance(text,str):raise ValueError('text must be a Unicode string')
    raw=text.encode('utf-8',errors='strict')
    if len(raw)>4096:raise ValueError('teaching document byte limit4096')
    return raw

def ids_checked(ids,vocab):
    if not isinstance(vocab,Integral) or isinstance(vocab,(bool,np.bool_)) or not 4<=vocab<=4096:raise ValueError('vocabulary size must be integer4..4096')
    if not isinstance(ids,(list,tuple,np.ndarray)):raise ValueError('IDs must be a sequence')
    if isinstance(ids,np.ndarray) and ids.ndim!=1:raise ValueError('IDs must be one-dimensional')
    if len(ids)>4098:raise ValueError('sequence length limit')
    if any(not isinstance(i,Integral) or isinstance(i,(bool,np.bool_)) or i<0 or i>=vocab for i in ids):raise ValueError('IDs must be integers within vocabulary')
    return [int(i) for i in ids]

class ByteBPE:
    def __init__(self,merges=None):
        if merges is not None and (not isinstance(merges,(list,tuple)) or len(merges)>64):raise ValueError('at most64 serialized merge pairs')
        self.pieces=[None]*4+[bytes([b]) for b in range(256)];self.merges=[]
        for pair in merges or []:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise ValueError('merge pair required')
            a,b=ids_checked(pair,len(self.pieces))
            if min(a,b)<4:raise ValueError('special IDs cannot be merged')
            if len(self.pieces[a])+len(self.pieces[b])>4096:raise ValueError('merged piece exceeds document byte limit')
            self.merges.append((a,b));self.pieces.append(self.pieces[a]+self.pieces[b])
    @staticmethod
    def replace(seq,pair,new):
        out=[];j=0
        while j<len(seq):
            if j+1<len(seq) and (seq[j],seq[j+1])==pair:out.append(new);j+=2
            else:out.append(seq[j]);j+=1
        return out
    @classmethod
    def fit(cls,documents,num_merges=16):
        if not isinstance(documents,(list,tuple)) or not 1<=len(documents)<=512:raise ValueError('1..512 training documents required')
        if not isinstance(num_merges,Integral) or isinstance(num_merges,bool) or not 0<=num_merges<=64:raise ValueError('merge budget0..64')
        raws=[text_bytes(t) for t in documents]
        if sum(map(len,raws))>200000:raise ValueError('total corpus byte limit')
        model=cls();seqs=[[b+4 for b in raw] for raw in raws];history=[]
        for step in range(num_merges):
            counts=Counter(pair for seq in seqs for pair in zip(seq,seq[1:]))
            if not counts:break
            pair=min(counts,key=lambda p:(-counts[p],p))
            if counts[pair]<2:break
            new=len(model.pieces);seqs=[cls.replace(seq,pair,new) for seq in seqs];model.merges.append(pair);model.pieces.append(model.pieces[pair[0]]+model.pieces[pair[1]])
            history.append({'step':step,'pair':list(pair),'frequency_count_including_overlap':counts[pair],'new_id':new,'new_piece_hex':model.pieces[new].hex(),'content_tokens_after':sum(map(len,seqs))})
        return model,history
    def encode(self,text,boundaries=True):
        if not isinstance(boundaries,bool):raise ValueError('boundaries must be bool')
        seq=[b+4 for b in text_bytes(text)]
        for j,pair in enumerate(self.merges):seq=self.replace(seq,pair,260+j)
        return [BOS]+seq+[EOS] if boundaries else seq
    def decode_bytes(self,ids,boundaries=True):
        if not isinstance(boundaries,bool):raise ValueError('boundaries must be bool')
        seq=ids_checked(ids,len(self.pieces))
        if boundaries:
            if len(seq)<2 or seq[0]!=BOS or seq[-1]!=EOS:raise ValueError('one BOS/EOS pair required')
            seq=seq[1:-1]
        if any(i<4 for i in seq):raise ValueError('control tokens forbidden inside byte content')
        return b''.join(self.pieces[i] for i in seq)
    def decode(self,ids,boundaries=True):return self.decode_bytes(ids,boundaries).decode('utf-8',errors='strict')
    def metadata(self):return {'kind':'educational byte-BPE','vocab_size':len(self.pieces),'special_ids':{'PAD':0,'BOS':1,'EOS':2,'UNK':3},'merges':[list(p) for p in self.merges],'pieces_hex':[p.hex() if p is not None else None for p in self.pieces],'normalization':'none','pretokenization':'none; may merge spaces; never cross documents'}

def pad_batch(sequences,vocab,pad_to=None):
    if not isinstance(sequences,(list,tuple)) or not 1<=len(sequences)<=512:raise ValueError('nonempty sequence batch <=512')
    seqs=[ids_checked(s,vocab) for s in sequences]
    if any(len(s)<2 or s[0]!=BOS or s[-1]!=EOS or any(i<4 for i in s[1:-1]) for s in seqs):raise ValueError('each sequence is BOS + content IDs>=4 + EOS')
    lengths=np.array([len(s)-1 for s in seqs],np.int64);T=int(lengths.max())
    if pad_to is not None:
        if not isinstance(pad_to,Integral) or isinstance(pad_to,bool) or pad_to<T:raise ValueError('pad_to must be integer >= longest input; no truncation')
        T=int(pad_to)
    if T>4097 or len(seqs)*T>200000:raise ValueError('padded batch teaching limit')
    x=np.full((len(seqs),T),PAD,np.int64);y=x.copy();mask=np.zeros_like(x,dtype=bool)
    for i,s in enumerate(seqs):x[i,:lengths[i]]=s[:-1];y[i,:lengths[i]]=s[1:];mask[i,:lengths[i]]=True
    return x,y,mask,lengths

def hand_ledger():
    # Six rows use global special IDs: PAD fixed0; BOS/EOS/UNK unused in this pooling example; A=4 repeated, B=5 shared.
    table=[F(0),F(1,5),F(2,5),F(3,10),F(1,2),F(-1,4)];w=F(2);b=F(1,10);samples_ids=[[4,4,5],[5]];targets=[F(1),F(0)];states=[]
    for step in range(3):
        samples=[];g=[F(0)]*8
        for ids,y in zip(samples_ids,targets):
            h=[table[t] for t in ids];m=sum(h)/len(h);pred=w*m+b;r=pred-y;up=r/2;per=up*w/len(ids);contrib=[F(0)]*8
            for t in ids:contrib[t]+=per
            contrib[6]=up*m;contrib[7]=up;g=[a+c for a,c in zip(g,contrib)];samples.append({'ids':ids,'lookup':h,'length':len(ids),'mean':m,'prediction':pred,'residual':r,'loss':r*r/2,'prediction_upstream':up,'each_lookup_upstream':per,'contribution':contrib})
        states.append({'step':step,'theta':table+[w,b],'samples':samples,'gradient':g,'loss':sum(s['loss'] for s in samples)/2})
        if step<2:
            table=[a-F(1,10)*dg for a,dg in zip(table,g[:6])];w-=F(1,10)*g[6];b-=F(1,10)*g[7]
    def encode(v):
        if isinstance(v,F):return {'exact':str(v),'float':float(v)}
        if isinstance(v,list):return [encode(x) for x in v]
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
        return v
    return encode(states)

def numpy_objective_gradient(E,W,b,ids,targets,mask):
    E=np.asarray(E);W=np.asarray(W);b=np.asarray(b);ids=np.asarray(ids);targets=np.asarray(targets);mask=np.asarray(mask)
    if E.ndim!=2 or min(E.shape)<=0 or W.shape!=(E.shape[1],E.shape[0]) or b.shape!=(E.shape[0],):raise ValueError('embedding/head shapes mismatch')
    if any(a.dtype.kind not in 'fiu' or not np.isfinite(a).all() for a in [E,W,b]):raise ValueError('parameters must be finite real')
    E,W,b=(a.astype(np.float64,copy=False) for a in [E,W,b])
    if any(not np.isfinite(a).all() for a in [E,W,b]):raise ValueError('parameters must be representable in float64')
    if ids.ndim!=2 or targets.shape!=ids.shape or mask.shape!=ids.shape or ids.dtype.kind not in 'iu' or targets.dtype.kind not in 'iu' or mask.dtype!=np.bool_:raise ValueError('integer IDs/targets and boolean mask of identical BxT required')
    if ids.size==0 or ids.size*len(b)>2000000 or not mask.any():raise ValueError('empty valid set or teaching logit-size limit')
    if ids.min()<0 or targets.min()<0 or ids.max()>=len(b) or targets.max()>=len(b):raise ValueError('token ID out of range')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        h=E[ids];z=h@W+b;mx=z.max(-1,keepdims=True);exp=np.exp(z-mx);prob=exp/exp.sum(-1,keepdims=True);count=mask.sum();losses=(np.log(exp.sum(-1))-np.take_along_axis(z-mx,targets[...,None],axis=-1)[...,0]);loss=float(losses[mask].sum()/count)
        dz=prob;rr,cc=np.indices(targets.shape);dz[rr,cc,targets]-=1;dz*=mask[...,None]/count;gW=np.einsum('btd,btv->dv',h,dz);gb=dz.sum((0,1));dh=dz@W.T;gE=np.zeros_like(E);np.add.at(gE,ids,dh);gE[PAD]=0
        return loss,(gE,gW,gb)

def metrics(E,W,b,documents,tokenizer):
    seq=[tokenizer.encode(t) for t in documents];x,y,mask,lengths=pad_batch(seq,len(tokenizer.pieces));xt=torch.tensor(x);yt=torch.tensor(y)
    with torch.no_grad():h=torch.nn.functional.embedding(xt,E,padding_idx=PAD);z=h@W+b;loss=torch.nn.functional.cross_entropy(z.reshape(-1,len(tokenizer.pieces)),yt.reshape(-1),ignore_index=PAD,reduction='none').reshape(y.shape);nll=loss.sum(1).numpy();total=float(nll.sum())
    bytes_n=sum(len(text_bytes(t)) for t in documents);token_n=int(mask.sum())
    return {'total_nll':total,'target_tokens_including_EOS':token_n,'utf8_content_bytes':bytes_n,'mean_nll_per_target':total/token_n,'token_perplexity':math.exp(total/token_n),'bits_per_utf8_byte':total/(bytes_n*math.log(2)) if bytes_n else None,'per_document_nll':nll.tolist(),'target_lengths':lengths.tolist(),'content_token_counts':[len(s)-2 for s in seq]}

def tokenizer_audit(corpus,byte,bpe,history):
    chars=sorted(set(''.join(corpus['train'])));char_to_id={c:i+4 for i,c in enumerate(chars)};probes=[]
    for text in corpus['probes']:
        raw=text_bytes(text);bc=byte.encode(text);bp=bpe.encode(text);char_ids=[char_to_id.get(c,UNK) for c in text]
        probes.append({'text':text,'unicode_codepoints':len(text),'utf8_bytes':len(raw),'character_ids':char_ids,'character_unknowns':sum(i==UNK for i in char_ids),'byte_content_tokens':len(bc)-2,'bpe_content_tokens':len(bp)-2,'byte_ids_with_boundaries':bc,'bpe_ids_with_boundaries':bp,'byte_roundtrip':byte.decode(bc)==text,'bpe_roundtrip':bpe.decode(bp)==text})
    return {'byte':byte.metadata(),'bpe':bpe.metadata(),'bpe_history':history,'character_vocab':char_to_id,'character_scope':'train-only codepoint tokenizer demonstration, no normalization, explicit UNK; not used for comparable loss experiment','probes':probes}

def run(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);raw=(ROOT/'data/corpus.json').read_bytes();expected=json.loads((ROOT/'data/data_manifest.json').read_text())['corpus_sha256']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('fixed corpus hash mismatch')
    data=json.loads(raw);byte=ByteBPE();bpe,history=ByteBPE.fit(data['train'],data['protocol']['bpe_merges']);audit=tokenizer_audit(data,byte,bpe,history);runs=[];traces=[];max_ref=0.;final_states=[];baselines={}
    for kind,tok in [('byte',byte),('byte_bpe',bpe)]:
        V=len(tok.pieces);x,y,mask,lengths=pad_batch([tok.encode(t) for t in data['train']],V);xt=torch.tensor(x);yt=torch.tensor(y);valid=int(mask.sum());counts=np.bincount(y[mask],minlength=V);p=(counts+1)/(counts.sum()+V);baseline={}
        for split in ['validation','test']:
            seqs=[tok.encode(t) for t in data[split]];total=float(sum(-math.log(p[target]) for seq in seqs for target in seq[1:]));num=sum(len(seq)-1 for seq in seqs);nbytes=sum(len(text_bytes(t)) for t in data[split]);baseline[split]={'mean_nll_per_target':total/num,'bits_per_utf8_byte':total/(nbytes*math.log(2)),'target_count':num}
        baselines[kind]={'kind':'add-one smoothed unigram of training targets includingEOS','alpha':1,'metrics':baseline}
        for seed in data['protocol']['seeds']:
            rng=np.random.default_rng(seed);initE=rng.normal(0,.12,(V,8));initE[PAD]=0;E=torch.tensor(initE,dtype=torch.float64,requires_grad=True);W=torch.tensor(rng.normal(0,.08,(8,V)),dtype=torch.float64,requires_grad=True);b=torch.zeros(V,dtype=torch.float64,requires_grad=True);trace=[]
            for step in range(data['protocol']['steps']+1):
                h=torch.nn.functional.embedding(xt,E,padding_idx=PAD);logits=h@W+b;loss=torch.nn.functional.cross_entropy(logits.reshape(-1,V),yt.reshape(-1),ignore_index=PAD,reduction='sum')/valid
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                trace.append({'step':step,'train_nll_per_target':float(loss.detach()),'E':E.detach().tolist(),'W':W.detach().tolist(),'b':b.detach().tolist()})
                if step==data['protocol']['steps']:break
                loss.backward()
                if step in [0,59,119]:
                    nl,ng=numpy_objective_gradient(E.detach().numpy(),W.detach().numpy(),b.detach().numpy(),x,y,mask);max_ref=max(max_ref,abs(nl-float(loss.detach())),*(float(np.max(np.abs(g-p.grad.numpy()))) for g,p in zip(ng,[E,W,b])))
                with torch.no_grad():
                    for parameter in [E,W,b]:
                        if not torch.isfinite(parameter.grad).all():raise FloatingPointError('nonfinite gradient')
                        parameter-=data['protocol']['learning_rate']*parameter.grad;parameter.grad=None
            runs.append({'kind':kind,'seed':seed,'vocab_size':V,'parameter_coordinates':17*V,'fixed_pad_coordinates':8,'train_target_tokens':valid,'metrics':{split:metrics(E,W,b,data[split],tok) for split in ['train','validation','test']}});traces.append({'kind':kind,'seed':seed,'states':trace});final_states.append({'kind':kind,'seed':seed,'E':E.detach().tolist(),'W':W.detach().tolist(),'b':b.detach().tolist()})
    summary={kind:{metric:float(np.mean([r['metrics']['test'][metric] for r in runs if r['kind']==kind])) for metric in ['mean_nll_per_target','token_perplexity','bits_per_utf8_byte']} for kind in ['byte','byte_bpe']}
    results={'protocol':data['protocol'],'tokenizer_audit':audit,'hand':hand_ledger(),'runs':runs,'summary':summary,'unigram_baselines':baselines,'numpy_reference_max_abs_gap':max_ref,'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU float64'}}
    out=Path(out);out.mkdir(parents=True,exist_ok=True);payloads={}
    for name,obj in [('training_traces.json.gz',traces),('final_states.json.gz',final_states)]:
        raw=(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n').encode();compressed=gzip.compress(raw,mtime=0);payloads[name]=compressed;results[name]={'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'compressed_sha256':hashlib.sha256(compressed).hexdigest()}
    payloads['results.json']=(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    for name,content in payloads.items():
        with tempfile.NamedTemporaryFile(dir=out,delete=False) as f:f.write(content);tmp=Path(f.name)
        os.replace(tmp,out/name)
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');args=p.parse_args();r=run(args.output);print(json.dumps({'summary':r['summary'],'baselines':r['unigram_baselines'],'numpy_reference_max_abs_gap':r['numpy_reference_max_abs_gap'],'bpe_vocab':r['tokenizer_audit']['bpe']['vocab_size']},ensure_ascii=False,indent=2))
