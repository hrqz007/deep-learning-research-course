"""Post-protocol counterfactual diagnostics; never used to pick a model."""
import json
import numpy as np
from experiment import ROOT,params_from_numpy,tensor,evaluate,NAMES

def main():
    r=json.loads((ROOT/'outputs/results.json').read_text());w=np.load(ROOT/'outputs/trained_parameters.npz',allow_pickle=False);d=np.load(ROOT/'data/delayed_sign.npz',allow_pickle=False);out=[]
    for run in r['runs']:
        if run['status']!='complete':continue
        p=params_from_numpy({k:w[run['id']+'_'+k] for k in NAMES});delay=run['delay']
        x=d[f'd{delay}_test_x'].copy();y=d[f'd{delay}_test_y'].copy()
        flipped=x.copy();flipped[:,0,0]*=-1
        erased=x.copy();erased[:,0,0]=0.
        out.append({'id':run['id'],'flipped_signal_and_label':evaluate(p,tensor(flipped),tensor(1-y)),'erased_first_signal':evaluate(p,tensor(erased),tensor(y))})
    payload={'status':'post-hoc mechanistic diagnostic, no checkpoint or hyperparameter selection','constant_zero_logit_bce':float(np.log(2)),'constant_class_accuracy':.5,'runs':out}
    (ROOT/'outputs/counterfactual.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
