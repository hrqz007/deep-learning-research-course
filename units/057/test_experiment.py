"""Metric denominators, grouped leakage, stops, and checkpoint readback."""
import math,json
import numpy as np
import torch
from experiment import make_data,contamination,wilson,evaluate,ROOT
from model import TinyLM,load_model

def check(ok,msg):
    if not ok:raise AssertionError(msg)
def main():
    torch.set_num_threads(1);rows=make_data();train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
    check(len(rows)==256 and len(test)==48,'split sizes')
    check(contamination(train,test)['prefix_group_overlap']==0,'group holdout')
    check(contamination(train,test+[train[0]])['exact_document_overlap']==1,'injected duplicate detected')
    lo,hi=wilson(0,12);check(abs(lo)<1e-12 and .24<hi<.25,'zero success has nonzero upper uncertainty')
    check(np.allclose(wilson(6,12),[.253778,.746222],atol=1e-5),'Wilson hand reference')
    try:wilson(0,0)
    except ValueError:pass
    else:raise AssertionError('empty denominator accepted')
    torch.manual_seed(5701);result=evaluate(TinyLM(),test,True)
    check(result['effective_tokens']==336 and result['independent_prompt_groups']==12,'token versus group denominator')
    check(math.isclose(math.exp(result['nll']),result['perplexity'],rel_tol=1e-5),'perplexity')
    expected=[ROOT/'outputs'/f'weights_s{seed}.npz' for seed in [5701,5702,5703]]
    check(all(path.is_file() for path in expected),'all three released checkpoints required')
    published=json.loads((ROOT/'outputs/results.json').read_text())
    check([r['seed'] for r in published['runs']]==[5701,5702,5703],'three fixed seeds')
    check(sum(p.numel() for p in TinyLM().parameters())==5252,'architecture parameter count')
    for path,record in zip(expected,published['runs']):
        rr=evaluate(load_model(path),test,True)
        check(sum(rr['error_counts'].values())==12,'exhaustive errors')
        check(abs(rr['nll']-record['test']['nll'])<1e-6,'released weights reproduce NLL')
        check(rr['predictions']==record['test']['predictions'],'released weights reproduce every prediction')
    print('PASS:256 documents, heldout prompt groups, duplicate detection, Wilson, denominators, safe NPZ loading')
if __name__=='__main__':main()
