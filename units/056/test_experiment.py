"""Focused correctness tests; explicit failures also work under python -O."""
import math
import numpy as np
import torch
from experiment import distribution,generate,cache_bytes,ROOT
from model import load_model

def check(ok,message):
    if not ok: raise AssertionError(message)
def rejects(fn):
    try: fn()
    except ValueError:return
    raise AssertionError('expected ValueError')
def main():
    torch.set_num_threads(1);m=load_model(ROOT/'data/weights.npz')
    z=torch.log(torch.tensor([.4,.3,.2,.1],dtype=torch.float64))
    check(torch.allclose(distribution(z,.5),torch.tensor([16,9,4,1],dtype=torch.float64)/30),'temperature power')
    check(torch.allclose(distribution(z,top_p=.6),torch.tensor([4/7,3/7,0,0],dtype=torch.float64)),'crossing token must remain')
    check(distribution(torch.zeros(4),top_k=2).tolist()==[.5,.5,0,0],'tie policy')
    for kwargs in [{'temperature':0},{'top_p':0},{'top_k':0},{'top_k':5}]:rejects(lambda:distribution(z,**kwargs))
    rejects(lambda:distribution(torch.full((4,),-math.inf)))
    seq=torch.tensor([[1,3,7,11,4,15,19],[1,4,8,12,6,17,19]])
    with torch.no_grad():
        full=m(seq); first,cache=m.cached(seq[:,:3]);last,cache=m.cached(seq[:,3:],cache)
        check(torch.allclose(torch.cat([first,last],1),full,atol=2e-5,rtol=1e-5),'chunked B2 equality')
        check(cache_bytes(cache)==2*2*2*7*16*4,'cache payload formula')
        changed=seq.clone();changed[:,-1]=18
        check(torch.allclose(m(changed)[:,:-1],full[:,:-1],atol=1e-6),'causal invariance')
    rejects(lambda:m.cached(torch.tensor([[3]]),cache))
    rejects(lambda:generate(m,[1,3],{},max_new=7))
    rejects(lambda:generate(m,[1,2],{},max_new=1))
    a,*_=generate(m,[1,3],{'top_p':.8},seed=5602)
    b,*_=generate(m,[1,3],{'top_p':.8},seed=5602,use_cache=False)
    check(a['tokens']==b['tokens'],'seeded cached sample')
    print('PASS: probability filters, tie/boundary policy, causal mask, B2 chunk cache, memory, stops')
if __name__=='__main__':main()
