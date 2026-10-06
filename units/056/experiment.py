"""DL056: fixed-checkpoint decoding, audited cache equality, and bounded timing."""
from pathlib import Path
import argparse,hashlib,json,math,platform,time
import numpy as np
import torch
from model import TinyLM,load_model,V,MAX_LEN,LAYERS,HEADS
ROOT=Path(__file__).resolve().parent
BOS,EOS=1,2

def distribution(logits,temperature=1.,top_k=None,top_p=1.):
    """Temperature -> exact stable top-k -> nucleus on the remaining distribution.

    The first token that reaches/exceeds p is retained. Ties use token ID order.
    -inf is allowed for an excluded token; NaN,+inf,or all-excluded inputs fail.
    """
    if logits.ndim!=1 or logits.numel()==0 or not logits.is_floating_point(): raise ValueError('one floating logit vector required')
    if not math.isfinite(temperature) or temperature<=0: raise ValueError('temperature must be finite and positive; use greedy explicitly')
    if not math.isfinite(top_p) or not 0<top_p<=1: raise ValueError('top_p in(0,1]')
    if top_k is not None and (isinstance(top_k,bool) or not isinstance(top_k,int) or not 1<=top_k<=logits.numel()): raise ValueError('top_k integer in[1,V]')
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any(): raise ValueError('invalid logits')
    z=logits.double()/temperature; order=torch.argsort(z,descending=True,stable=True)
    keep=order if top_k is None else order[:top_k]
    p=z[keep].softmax(0)
    # Previous cumulative mass is strictly below threshold => keep crossing token.
    included=(p.cumsum(0)-p)<top_p
    if top_p==1.: included=torch.ones_like(included)
    kept=keep[included]; result=torch.zeros_like(z)
    result[kept]=z[kept].softmax(0); return result

@torch.no_grad()
def generate(model,prompt,config,seed=5601,use_cache=True,max_new=6):
    if not isinstance(max_new,int) or isinstance(max_new,bool) or max_new<1: raise ValueError('max_new positive integer')
    if not prompt or any(not isinstance(t,int) or t<0 or t>=V for t in prompt): raise ValueError('invalid prompt')
    if EOS in prompt: raise ValueError('prompt must be an unfinished prefix without EOS')
    if len(prompt)+max_new-1>MAX_LEN: raise ValueError('generation would exceed learned position capacity')
    model.eval(); ids=list(prompt); cache=None; rng=torch.Generator().manual_seed(seed); traces=[]; start=time.perf_counter()
    for i in range(max_new):
        x=torch.tensor([ids if (not use_cache or i==0) else [ids[-1]]],dtype=torch.long)
        if use_cache: logits,cache=model.cached(x,cache)
        else: logits=model(x)
        z=logits[0,-1].clone(); z[:2]=-torch.inf # PAD/BOS are forbidden output symbols.
        if config.get('greedy',False): token=int(z.argmax())
        else: token=int(torch.multinomial(distribution(z,**config),1,generator=rng))
        traces.append(z.tolist()); ids.append(token)
        if token==EOS: break
    elapsed=time.perf_counter()-start; new=ids[len(prompt):]
    pairs=list(zip(new,new[1:])); repetition=0. if not pairs else 1-len(set(pairs))/len(pairs)
    return {'tokens':new,'stop_reason':'eos' if new[-1]==EOS else 'max_new_tokens',
       'new_tokens':len(new),'seconds':elapsed,'repeated_bigram_fraction':repetition,
       'adjacent_repeat_count':sum(a==b for a,b in pairs)},traces,cache

def cache_bytes(cache): return sum(t.numel()*t.element_size() for pair in cache for t in pair)

def run(output):
    torch.set_num_threads(1);torch.manual_seed(5600);torch.use_deterministic_algorithms(True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    model=load_model(ROOT/'data/weights.npz');corpus=json.loads((ROOT/'data/corpus.json').read_text());vocab=corpus['vocab']
    configs={'greedy':{'greedy':True},'T0.7':{'temperature':.7},'T1.3':{'temperature':1.3},
       'top-k3':{'temperature':1.,'top_k':3},'top-p0.8':{'temperature':1.,'top_p':.8}}
    rows=[r for r in corpus['documents'] if r['split']=='train'][:4]
    prompts=[[BOS,r['tokens'][0]] for r in rows]
    proto={'seed_range':list(range(5601,5609)),'prompt_document_ids':[r['id'] for r in rows],
       'prompts':prompts,'configs':configs,'max_new':6,'forbidden_outputs':[0,1],
       'weight_sha256':hashlib.sha256((ROOT/'data/weights.npz').read_bytes()).hexdigest(),
       'benchmark':'fixed7 input tokens from one training document; 3 warmups,20 interleaved repetitions; no sampling,no EOS stop',
       'scope':'4 training prefixes x8 seeds x5 decoding rules; fixed model; no model selection'}
    (out/'protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2)+'\n')
    all_rows=[];max_error=0.
    for name,cfg in configs.items():
        for pi,prompt in enumerate(prompts):
            for seed in proto['seed_range']:
                a,at,cache=generate(model,prompt,cfg,seed,True)
                b,bt,_=generate(model,prompt,cfg,seed,False)
                error=max(float(np.max(np.abs(np.asarray(x[2:])-np.asarray(y[2:])))) for x,y in zip(at,bt))
                max_error=max(max_error,error)
                if a['tokens']!=b['tokens'] or error>2e-5: raise RuntimeError('cache/full decode mismatch')
                all_rows.append({'config':name,'prompt_index':pi,'seed':seed,**a,
                    'text':' '.join(vocab[t] for t in a['tokens']),'uncached_seconds':b['seconds'],
                    'cache_bytes':cache_bytes(cache),'max_logit_abs_error':error})
    sequence=torch.tensor([[BOS]+rows[0]['tokens']],dtype=torch.long)
    timings={'full':[],'cache':[]}
    @torch.no_grad()
    def fixed_work(kind):
        kv=None;begin=time.perf_counter()
        for i in range(1,8):
            if kind=='full': model(sequence[:,:i])
            else: _,kv=model.cached(sequence[:,i-1:i],kv)
        return time.perf_counter()-begin
    for _ in range(3):fixed_work('full');fixed_work('cache')
    for i in range(20):
        for kind in (['full','cache'] if i%2==0 else ['cache','full']):timings[kind].append(fixed_work(kind))
    summaries=[]
    for name in configs:
        rr=[r for r in all_rows if r['config']==name]
        summaries.append({'config':name,'n':len(rr),'mean_length':float(np.mean([r['new_tokens'] for r in rr])),
            'eos_fraction':float(np.mean([r['stop_reason']=='eos' for r in rr])),
            'unique_continuations':len({(r['prompt_index'],tuple(r['tokens'])) for r in rr}),
            'mean_repeat_fraction':float(np.mean([r['repeated_bigram_fraction'] for r in rr])),
            'adjacent_repeat_total':sum(r['adjacent_repeat_count'] for r in rr),
            'median_cached_ms':float(np.median([r['seconds'] for r in rr])*1000)})
    hand={}
    z=torch.log(torch.tensor([.4,.3,.2,.1],dtype=torch.float64))
    for name,cfg in {'T1':{},'T0.5':{'temperature':.5},'k2':{'top_k':2},'p0.6':{'top_p':.6},'p0.7':{'top_p':.7}}.items():hand[name]=distribution(z,**cfg).tolist()
    result={'protocol':proto,'summary':summaries,'samples':all_rows,'hand_calculation':hand,'max_cache_logit_abs_error':max_error,
        'fixed_workload_seconds':timings,'fixed_workload_median_ms':{k:1000*float(np.median(v)) for k,v in timings.items()},
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':1,'device':'cpu'},
        'limits':'short7-position model; repeated bigrams are descriptive not semantic quality; timing is local and includes Python overhead'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args();r=run(a.output)
    print(json.dumps({'summary':r['summary'],'cache_error':r['max_cache_logit_abs_error'],'fixed_timing':r['fixed_workload_median_ms']},ensure_ascii=False,indent=2))
