"""DL055: a bounded, fully local CPU language-model training experiment.

12 predeclared runs; each gets a fresh process for per-worker peak RSS measurement.
No downloading, no hidden hyperparameter search, no test-set scoring.
"""
from pathlib import Path
import argparse,copy,json,math,platform,resource,subprocess,sys,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from generate_data import VOCAB,BOS,EOS,main as generate
ROOT=Path(__file__).resolve().parent
V=len(VOCAB);L=7;BATCH=16;LAYERS=2;HEADS=4

def setup(seed):
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)

def protocol():
    return {'version':'1','device':'cpu','dtype':'float32','threads':1,'steps':80,'batch_size':BATCH,
       'sequence_length':L,'heads':HEADS,'layers':LAYERS,'ffn_multiplier':2,'dropout':0,
       'lr':.003,'weight_decay':.01,'adam_betas':[.9,.999],'grad_clip':1.,'warmup_timing_steps':5,
       'checkpoints':[0,20,40,60,74,80],'effective_token_limit_per_run':80*BATCH*L,
       'aggregate_effective_token_limit':12*80*BATCH*L,'training_timeout_seconds_per_worker':180,
       'declared_scope':'2 widths x2 distinct training pools x3 initializations; no scaling-law fit',
       'test_policy':'test split retained but not scored or used for tuning',
       'configs':[{'d':d,'pool':pool,'seed':seed,'name':f'd{d}_n{pool}_s{seed}'}
         for d in [16,32] for pool in [32,128] for seed in [5501,5502,5503]]}

class DecoderBlock(nn.Module):
    """Pre-norm, explicit multihead masked attention and ReLU FFN."""
    def __init__(self,d):
        super().__init__();self.d=d
        self.norm1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d)
        self.norm2=nn.LayerNorm(d);self.fc1=nn.Linear(d,2*d);self.fc2=nn.Linear(2*d,d)
    def forward(self,x):
        b,length,d=x.shape;z=self.norm1(x)
        q,k,v=[t.reshape(b,length,HEADS,d//HEADS).transpose(1,2) for t in self.qkv(z).chunk(3,-1)]
        scores=q@k.transpose(-1,-2)/math.sqrt(d//HEADS)
        forbidden=torch.ones(length,length,dtype=torch.bool,device=x.device).triu(1)
        weights=scores.masked_fill(forbidden,-torch.inf).softmax(-1)
        merged=(weights@v).transpose(1,2).contiguous().reshape(b,length,d)
        x=x+self.proj(merged)
        return x+self.fc2(F.relu(self.fc1(self.norm2(x))))
class TinyLM(nn.Module):
    def __init__(self,d=16):
        super().__init__()
        if d<4 or d%HEADS:raise ValueError('width must be positive and divisible by four heads')
        self.d=d;self.embed=nn.Embedding(V,d);self.position=nn.Embedding(L,d)
        self.blocks=nn.ModuleList([DecoderBlock(d) for _ in range(LAYERS)])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,V) # Deliberately untied output embeddings.
    def forward(self,x):
        if x.ndim!=2 or x.dtype!=torch.long or min(x.shape)==0 or x.shape[1]>L:
            raise ValueError('nonempty B,L int64 input; L<=7')
        if (x<0).any() or (x>=V).any():raise ValueError('token out of vocabulary')
        h=self.embed(x)+self.position(torch.arange(x.shape[1],device=x.device))[None]
        for block in self.blocks:h=block(h)
        return self.head(self.norm(h))

def parameter_formula(d):
    # token+position embedding, two blocks, final LN, untied output weight and bias.
    f=2*d
    return V*d+L*d+LAYERS*(4*d*d+2*d*f+9*d+f)+2*d+d*V+V

def forward_matmul_flops(d,b=BATCH,length=L):
    """Dense multiply+add counted as2, excludes LN/softmax/ReLU/embedding/optimizer."""
    f=2*d
    return LAYERS*(8*b*length*d*d+4*b*length*d*f+4*b*length*length*d)+2*b*length*d*V

def tensors(rows):
    x=torch.tensor([[BOS]+r['tokens'] for r in rows],dtype=torch.long)
    y=torch.tensor([r['tokens']+[EOS] for r in rows],dtype=torch.long)
    return x,y

def nll(logits,y):
    return F.cross_entropy(logits.reshape(-1,V),y.reshape(-1),reduction='mean')

@torch.no_grad()
def evaluate(model,x,y):
    model.eval();logits=model(x);losses=F.cross_entropy(logits.reshape(-1,V),y.reshape(-1),reduction='none').reshape_as(y)
    return {'nll':losses.mean().item(),'sum_nll':losses.sum().item(),'effective_tokens':y.numel(),
            'position_nll':losses.mean(0).tolist()}

def baselines(train_rows,val_rows):
    tx,ty=tensors(train_rows);vx,vy=tensors(val_rows)
    unigram=torch.bincount(ty.flatten(),minlength=V).double()+1
    up=unigram/unigram.sum();unll=-up[vy].log().mean().item()
    counts=torch.ones(V,V,dtype=torch.float64)
    for a,b in zip(tx.flatten(),ty.flatten()):counts[a,b]+=1
    bp=counts/counts.sum(1,keepdim=True);bnll=-bp[vx,vy].log().mean().item()
    return {'uniform_nll':math.log(V),'train_unigram_add1_nll':unll,'train_bigram_add1_nll':bnll,
            'smoothing':'add-one over declared V=20 classes, fit training pool only'}

def run_one(config,out,proto,corpus_obj):
    setup(config['seed']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train=[r for r in corpus_obj['documents'] if r['split']=='train'][:config['pool']]
    val=[r for r in corpus_obj['documents'] if r['split']=='validation']
    tx,ty=tensors(train);vx,vy=tensors(val)
    model=TinyLM(config['d']);actual=sum(p.numel() for p in model.parameters())
    if actual!=parameter_formula(config['d']):raise RuntimeError('parameter accounting mismatch')
    opt=torch.optim.AdamW(model.parameters(),lr=proto['lr'],betas=tuple(proto['adam_betas']),weight_decay=proto['weight_decay'])
    generator=torch.Generator().manual_seed(5590) # Same order policy across widths/seeds.
    order=torch.randperm(len(train),generator=generator);cursor=0;seen=set();trace=[];durations=[];step_logs=[]
    initial=evaluate(model,vx,vy);trace.append({'step':0,'seen_tokens':0,'unique_docs_seen':0,**initial})
    start_all=time.perf_counter()
    for step in range(1,proto['steps']+1):
        start=time.perf_counter()
        if cursor+BATCH>len(train):order=torch.randperm(len(train),generator=generator);cursor=0
        ids=order[cursor:cursor+BATCH];cursor+=BATCH;seen.update(ids.tolist())
        xb,yb=tx[ids],ty[ids];model.train();opt.zero_grad(set_to_none=True)
        logits=model(xb);loss=nll(logits,yb)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward()
        diagnostic=None
        if step<=2:
            # A single real parameter ties CE -> bias gradient -> clipping -> AdamW.
            raw=model.head.bias.grad[3].item();before=model.head.bias[3].item()
            with torch.no_grad():
                analytic=(logits.softmax(-1)[...,3]-(yb==3).float()).mean().item()
            diagnostic={'parameter':'head.bias[3]','before':before,'raw_gradient':raw,'analytic_ce_gradient':analytic}
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),proto['grad_clip'],error_if_nonfinite=True)
        if diagnostic is not None:diagnostic['clipped_gradient']=model.head.bias.grad[3].item()
        opt.step()
        if diagnostic is not None:
            state=opt.state[model.head.bias]
            diagnostic.update(after=model.head.bias[3].item(),m=state['exp_avg'][3].item(),v=state['exp_avg_sq'][3].item())
        duration=time.perf_counter()-start;durations.append(duration)
        step_logs.append({'step':step,'nll_before_update':loss.item(),'gradient_norm_before_clip':gn.item(),
           'train_step_seconds':duration,'effective_tokens':yb.numel(),'optimizer_example':diagnostic,'document_ids':[train[i]['id'] for i in ids.tolist()]})
        if step in proto['checkpoints']:
            trace.append({'step':step,'seen_tokens':step*BATCH*L,'unique_docs_seen':len(seen),**evaluate(model,vx,vy)})
        if time.perf_counter()-start_all>proto['training_timeout_seconds_per_worker']:
            (out/'partial-results.json').write_text(json.dumps({'config':config,'status':'time_budget_exceeded','steps':step_logs,'trace':trace},ensure_ascii=False,indent=2)+'\n')
            raise RuntimeError('declared training time budget exceeded; partial-results.json retained')
    final_train=evaluate(model,tx,ty);final_val=evaluate(model,vx,vy)
    # Linux getrusage is KiB. Fresh subprocess gives worker-local peak, not isolated tensor memory.
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 if sys.platform.startswith('linux') else None
    burn=proto['warmup_timing_steps'];steady_seconds=sum(durations[burn:]);steady_tokens=(proto['steps']-burn)*BATCH*L
    result={'config':config,'parameter_count':actual,'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
       'effective_tokens':proto['steps']*BATCH*L,'processed_input_tokens':proto['steps']*BATCH*L,
       'unique_train_documents_available':len(train),'unique_train_documents_seen':len(seen),
       'unique_target_positions_seen':len(seen)*L,'repeat_exposure_ratio':proto['steps']*BATCH/len(seen),
       'forward_matmul_flops_per_step':forward_matmul_flops(config['d']),
       'estimated_train_matmul_flops':3*forward_matmul_flops(config['d'])*proto['steps'],
       'training_step_seconds_sum':sum(durations),'steady_training_step_seconds':steady_seconds,
       'steady_effective_tokens':steady_tokens,'steady_tokens_per_second':steady_tokens/steady_seconds,
       'timing_excludes':'first5 updates from steady metric; all evaluation, model construction, file writing and process startup',
       'peak_worker_rss_bytes':peak_rss_bytes,'rss_scope':'fresh Linux worker peak includes interpreter, model, optimizer, training and validation; not just tensors',
       'gpu_peak_memory_bytes':None,'gpu_memory_status':'not measured: CPU-only experiment',
       'trace':trace,'steps':step_logs,'final_train':final_train,'final_validation':final_val,'baselines':baselines(train,val),
       'test_set_evaluated':False}
    # Public, pickle-free inference checkpoint; optimizer/RNG not stored, so no resume claim.
    np.savez(out/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True);obj=generate(out/'data');proto=protocol()
    # Persist design before observing validation results; no adaptive run selection follows.
    (out/'protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
    records=[];started=time.perf_counter()
    for i,cfg in enumerate(proto['configs']):
        command=[sys.executable,str(Path(__file__).resolve()),'--worker-index',str(i),'--output',str(out)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=240)
        (out/f'worker-{i:02}.txt').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(f'worker {i} failed; inspect its log')
        r=json.loads((out/'runs'/cfg['name']/'results.json').read_text());records.append(r)
        print(cfg['name'],'validation',round(r['final_validation']['nll'],6),flush=True)
    runtime={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
        'device':'CPU','dtype':'float32','threads':1,'fresh_worker_per_configuration':True,'wall_seconds':time.perf_counter()-started}
    cpu=Path('/proc/cpuinfo')
    if cpu.exists():runtime['cpu_model']=next((line.split(':',1)[1].strip() for line in cpu.read_text().splitlines() if line.startswith('model name')),'unknown')
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    summary={'protocol':proto,'runs':[{k:v for k,v in r.items() if k!='steps'} for r in records],
       'total_effective_training_tokens':sum(r['effective_tokens'] for r in records),
       'negative_results':'all configurations retained; compare baselines and validation, not training loss alone',
       'scope':'descriptive tiny synthetic grammar; no fitted scaling law or frontier extrapolation'}
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'outputs'));p.add_argument('--worker-index',type=int);a=p.parse_args()
    if a.worker_index is None:run(a.output)
    else:
        out=Path(a.output);proto=json.loads((out/'protocol.json').read_text());obj=json.loads((out/'data/corpus.json').read_text());c=proto['configs'][a.worker_index]
        r=run_one(c,out/'runs'/c['name'],proto,obj);print(json.dumps({'config':c,'final_validation_nll':r['final_validation']['nll']}))
