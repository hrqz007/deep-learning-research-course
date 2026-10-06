"""DL057: train from scratch, freeze a grouped test, and retain every failure.

All data are original finite synthetic symbols. No external LM or paid API.
"""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,math,platform,time
import numpy as np
import torch
import torch.nn.functional as F
from model import TinyLM,V
ROOT=Path(__file__).resolve().parent
VOCAB=['PAD','BOS','EOS','红','蓝','绿','黄','猫','狗','鸟','鱼','看','拿','找','推','球','书','盒','杯','。']

def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def make_data():
    groups=list(itertools.product(range(4),repeat=3));order=np.random.default_rng(5700).permutation(64);docs=[]
    for rank,i in enumerate(order):
        a,b,c=groups[i];split='train' if rank<40 else ('validation' if rank<52 else 'test')
        for obj in range(4):
            docs.append({'id':f'g{i:02}_o{obj}','group':int(i),'split':split,'tokens':[3+a,7+b,11+c,3+(a+b+c)%4,15+obj,19]})
    return docs

def contamination(train,test):
    full={tuple(r['tokens']) for r in train};prefix={tuple(r['tokens'][:3]) for r in train}
    return {'exact_document_overlap':sum(tuple(r['tokens']) in full for r in test),
       'prefix_group_overlap':len({tuple(r['tokens'][:3]) for r in test}&prefix),
       'token_vocabulary_overlap':len({t for r in train for t in r['tokens']}&{t for r in test for t in r['tokens']}),
       'test_documents':len(test),'test_groups':len({r['group'] for r in test})}

def tensors(rows):
    return torch.tensor([[1]+r['tokens'] for r in rows]),torch.tensor([r['tokens']+[2] for r in rows])

def wilson(k,n,z=1.96):
    if not isinstance(n,int) or n<=0 or not 0<=k<=n:raise ValueError('require0<=k<=n and n>0')
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

@torch.no_grad()
def evaluate(model,rows,details=False):
    model.eval();x,y=tensors(rows);logits=model(x)
    losses=F.cross_entropy(logits.reshape(-1,V),y.reshape(-1),reduction='none').reshape_as(y)
    # Exactly one rule task per independent prefix group; object alternatives share a prompt.
    group_rows={r['group']:r for r in rows};predictions=[]
    for g,row in group_rows.items():
        prefix=[1]+row['tokens'][:3];generated=[]
        for _ in range(4):
            z=model(torch.tensor([prefix+generated]))[0,-1].clone();z[:2]=-torch.inf
            token=int(z.argmax());generated.append(token)
            if token==2:break
        expected_color=row['tokens'][3]
        format_ok=len(generated)==4 and 3<=generated[0]<=6 and 15<=generated[1]<=18 and generated[2:]==[19,2]
        rule_ok=generated[0]==expected_color
        # A single-reference EM is computed separately for each of4 valid object alternatives.
        references=[r for r in rows if r['group']==g]
        em=sum(generated==r['tokens'][3:]+[2] for r in references)
        predictions.append({'group':g,'prompt':prefix,'gold_color':expected_color,'prediction':generated,
            'rule_correct':rule_ok,'format_correct':format_ok,'task_success':rule_ok and format_ok,
            'exact_match_count':em,'references':len(references),
            'error':'success' if rule_ok and format_ok else ('wrong_rule' if format_ok else 'format_or_stop_failure')})
    n=len(predictions);k=sum(r['task_success'] for r in predictions);rule=sum(r['rule_correct'] for r in predictions)
    out={'sum_nll':float(losses.sum()),'effective_tokens':int(losses.numel()),'nll':float(losses.mean()),
       'perplexity':float(losses.mean().exp()),'position_nll':losses.mean(0).tolist(),
       'rule_accuracy':rule/n,'format_rate':sum(r['format_correct'] for r in predictions)/n,
       'task_success_rate':k/n,'task_success_count':k,'independent_prompt_groups':n,'wilson95_task_success':wilson(k,n),
       'single_reference_exact_match':sum(r['exact_match_count'] for r in predictions)/len(rows),
       'error_counts':{key:sum(r['error']==key for r in predictions) for key in ['success','wrong_rule','format_or_stop_failure']}}
    if details:out['predictions']=predictions;out['document_nll']=[{'id':r['id'],'group':r['group'],'sum_nll':float(losses[i].sum())} for i,r in enumerate(rows)]
    return out

def run(output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);docs=make_data()
    train=[r for r in docs if r['split']=='train'];val=[r for r in docs if r['split']=='validation'];test=[r for r in docs if r['split']=='test']
    proto={'seeds':[5701,5702,5703],'width':16,'steps':120,'batch_size':16,'learning_rate':.003,'weight_decay':.01,
       'gradient_clip':1.,'validation_checkpoints':[0,40,120],'test_checkpoint':120,
       'selection':'fixed final120 steps; no best-validation selection, no test-based tuning',
       'split':'64 prefix groups shuffled by5700:40train,12validation,12test;4 objects each',
       'task':'given BOS and3 rule input symbols, generate color,object,period,EOS; greedy, PAD/BOS suppressed',
       'budget_effective_tokens_per_seed':120*16*7,'max_train_seconds_per_seed':120,
       'human_evaluation_status':'not performed; blank blinded rating sheet only',
       'uncertainty':'Wilson95 on12 prompt groups for each seed, approximate iid-superpopulation interpretation; fixed finite synthetic split'}
    write_json(out/'protocol.json',proto);write_json(out/'corpus.json',{'vocab':VOCAB,'documents':docs,'rule':'color2=(color1+animal+action)mod4'})
    audit={'train_validation':contamination(train,val),'train_test':contamination(train,test),'validation_test':contamination(val,test)}
    if any(a['exact_document_overlap'] or a['prefix_group_overlap'] for a in audit.values()):raise RuntimeError('split leakage')
    write_json(out/'contamination.json',audit);tx,ty=tensors(train);records=[];started=time.perf_counter()
    for seed in proto['seeds']:
        torch.manual_seed(seed);model=TinyLM();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        rng=torch.Generator().manual_seed(5790);trace=[{'step':0,'validation':evaluate(model,val)}];steps=[];begin=time.perf_counter()
        for step in range(1,121):
            ids=torch.randperm(len(train),generator=rng)[:16];model.train();opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(tx[ids]).reshape(-1,V),ty[ids].reshape(-1));loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            steps.append({'step':step,'nll':loss.item(),'gradient_norm':gn.item(),'document_ids':[train[i]['id'] for i in ids.tolist()]})
            if step in [40,120]:trace.append({'step':step,'validation':evaluate(model,val)})
            if time.perf_counter()-begin>120:raise RuntimeError('training budget exceeded; no automatic extension')
        # All design choices are already fixed. The final test is opened once per frozen seed model.
        rr={'seed':seed,'training':evaluate(model,train),'validation_trace':trace,'test':evaluate(model,test,True),'training_steps':steps}
        records.append(rr);np.savez(out/f'weights_s{seed}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    # Fit smoothing baseline using training only, then evaluate it on test tokens.
    counts=torch.ones(V,V,dtype=torch.float64)
    for a,b in zip(tx.flatten(),ty.flatten()):counts[a,b]+=1
    probs=counts/counts.sum(1,keepdim=True);sx,sy=tensors(test);baseline=-probs[sx,sy].log().mean().item()
    # Produce an anonymized worksheet; these are NOT fabricated human scores.
    cells=[];key=[];shuffle=np.random.default_rng(5710)
    items=[(r['seed'],p) for r in records for p in r['test']['predictions']]
    for i,j in enumerate(shuffle.permutation(len(items))):
        seed,pred=items[j];cid=f'C{i+1:03}';key.append({'candidate_id':cid,'seed':seed,'group':pred['group']})
        cells.append({'candidate_id':cid,'prompt':' '.join(VOCAB[t] for t in pred['prompt']),
           'answer':' '.join(VOCAB[t] for t in pred['prediction']),'rule_correct_0_or_1':'','format_correct_0_or_1':'','comment':''})
    with (out/'blind_rating_sheet.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    write_json(out/'blind_key.json',key)
    result={'protocol':proto,'runs':records,'contamination':audit,'baselines':{'uniform_nll':math.log(V),'train_bigram_add1_test_nll':baseline},
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':1,'wall_seconds':time.perf_counter()-started},
        'scope':'synthetic compositional split only; no claims about factual natural-language QA, no human or LLM judges run'}
    write_json(out/'results.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');args=p.parse_args();result=run(args.output)
    for r in result['runs']:print(r['seed'],{k:v for k,v in r['test'].items() if k not in ['predictions','document_nll']})
