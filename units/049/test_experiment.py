"""Deterministic tokenization, boundaries, sparse accumulation and gradients."""
from pathlib import Path
from collections import Counter
import gzip,json,unittest,tempfile
from unittest.mock import patch
import experiment
import numpy as np
import torch
from experiment import ByteBPE,pad_batch,numpy_objective_gradient,hand_ledger,PAD,BOS,EOS,UNK,text_bytes
import generate_data
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)
def close(a,b):np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11)
def fs(xs):return np.array([x['float'] for x in xs])
class Contracts(unittest.TestCase):
    def test_bpe_known_pairs_overlap_tie_and_boundaries(self):
        m,h=ByteBPE.fit(['abab','abab'],2);self.assertEqual(m.merges,[(101,102),(260,260)]);self.assertEqual(m.encode('abab'),[BOS,261,EOS]);self.assertEqual(m.decode([BOS,261,EOS]),'abab')
        m,h=ByteBPE.fit(['aaaa'],4);self.assertEqual(h[0]['frequency_count_including_overlap'],3);self.assertEqual(m.encode('aaaa',False),[260,260]);self.assertEqual(len(m.merges),1)
        tie,_=ByteBPE.fit(['abac','abac'],1);self.assertEqual(tie.merges,[(101,102)])
        split,_=ByteBPE.fit(['a','b'],10);self.assertEqual(split.merges,[])
    def test_roundtrips_special_literal_and_unseen_unicode(self):
        m,_=ByteBPE.fit(['样品 A stable.','样品 A stable.'],16)
        for t in ['','材料A','café','cafe\u0301','👩\u200d🔬','<PAD>','\x00\n','🧪 α']:
            ids=m.encode(t);self.assertEqual(m.decode(ids),t);self.assertTrue(all(v>=4 for v in ids[1:-1]));self.assertEqual(m.decode_bytes(ids),t.encode('utf-8'))
        rebuilt=ByteBPE(m.metadata()['merges']);self.assertEqual(rebuilt.encode('样品 A'),m.encode('样品 A'))
        with self.assertRaises(UnicodeDecodeError):ByteBPE().decode([BOS,259,EOS])
        with self.assertRaises(UnicodeEncodeError):m.encode('\ud800')
    def test_padding_target_shift_and_empty_document(self):
        x,y,mask,lens=pad_batch([[BOS,4,5,EOS],[BOS,EOS]],260,pad_to=5);self.assertEqual(lens.tolist(),[3,1]);self.assertEqual(x.tolist(),[[1,4,5,0,0],[1,0,0,0,0]]);self.assertEqual(y.tolist(),[[4,5,2,0,0],[2,0,0,0,0]]);self.assertEqual(mask.sum(),4)
        self.assertNotIn((EOS,BOS),[(int(a),int(b)) for a,b in zip(x[mask],y[mask])])
    def test_padding_invariant_loss_and_all_gradients(self):
        rng=np.random.default_rng(4951);E=rng.normal(size=(7,2));E[PAD]=0;W=rng.normal(size=(2,7));b=rng.normal(size=7)
        a=pad_batch([[BOS,4,4,5,EOS],[BOS,6,EOS]],7);c=pad_batch([[BOS,4,4,5,EOS],[BOS,6,EOS]],7,pad_to=9);la,ga=numpy_objective_gradient(E,W,b,*a[:3]);lc,gc=numpy_objective_gradient(E,W,b,*c[:3]);self.assertAlmostEqual(la,lc,places=14)
        for x,y in zip(ga,gc):close(x,y)
        et=torch.tensor(E,requires_grad=True);wt=torch.tensor(W,requires_grad=True);bt=torch.tensor(b,requires_grad=True);xt=torch.tensor(a[0]);yt=torch.tensor(a[1]);z=torch.nn.functional.embedding(xt,et,padding_idx=PAD)@wt+bt;L=torch.nn.functional.cross_entropy(z.reshape(-1,7),yt.reshape(-1),ignore_index=PAD,reduction='sum')/a[2].sum();L.backward();self.assertAlmostEqual(float(L.detach()),la,places=14)
        for x,y in zip(ga,[et.grad,wt.grad,bt.grad]):close(x,y.numpy())
    def test_all_active_small_parameters_finite_difference(self):
        rng=np.random.default_rng(4952);E=rng.normal(0,.2,(7,2));E[PAD]=0;W=rng.normal(0,.2,(2,7));b=rng.normal(0,.1,7);x,y,mask,_=pad_batch([[1,4,4,5,2],[1,6,2]],7);loss,grad=numpy_objective_gradient(E,W,b,x,y,mask)
        for ai,(a,g) in enumerate(zip([E,W,b],grad)):
            for idx in np.ndindex(a.shape):
                if ai==0 and idx[0]==PAD:continue
                old=a[idx];a[idx]=old+1e-6;plus=numpy_objective_gradient(E,W,b,x,y,mask)[0];a[idx]=old-1e-6;minus=numpy_objective_gradient(E,W,b,x,y,mask)[0];a[idx]=old;self.assertAlmostEqual(g[idx],(plus-minus)/2e-6,delta=3e-9)
    def test_repeated_rows_and_exact_hand_states(self):
        ids=torch.tensor([[4,4,5],[5,0,0]]);mask=(ids!=0).double();length=mask.sum(1);target=torch.tensor([1.,0.],dtype=torch.float64)
        for state in hand_ledger():
            theta=fs(state['theta']);E=torch.tensor(theta[:6,None],requires_grad=True);w=torch.tensor(theta[6],requires_grad=True);b=torch.tensor(theta[7],requires_grad=True);h=torch.nn.functional.embedding(ids,E,padding_idx=0)[...,0];m=(h*mask).sum(1)/length;p=w*m+b;L=((p-target)**2).mean()/2;L.backward();close(p.detach().numpy(),[x['prediction']['float'] for x in state['samples']]);close(np.r_[E.grad.numpy().ravel(),w.grad.numpy(),b.grad.numpy()],fs(state['gradient']));self.assertAlmostEqual(float(L.detach()),state['loss']['float'],places=14)
        scatter=np.zeros(6);np.add.at(scatter,[4,4],[1.,2.]);self.assertEqual(scatter[4],3)
    def test_onehot_lookup_id_renumbering_and_integer_parameters(self):
        rng=np.random.default_rng(4963);E=rng.normal(size=(7,2));W=rng.normal(size=(2,7));b=rng.normal(size=7);ids=np.array([[1,4,4,5],[1,6,0,0]]);target=np.array([[4,4,5,2],[6,2,0,0]]);mask=target!=0
        close(np.eye(7)[ids]@E,E[ids])
        perm=np.array([0,1,2,3,6,4,5]);Ep=np.empty_like(E);Ep[perm]=E;Wp=np.empty_like(W);Wp[:,perm]=W;bp=np.empty_like(b);bp[perm]=b
        l,g=numpy_objective_gradient(E,W,b,ids,target,mask);lp,gp=numpy_objective_gradient(Ep,Wp,bp,perm[ids],perm[target],mask);close(l,lp);close(g[0],gp[0][perm]);close(g[1],gp[1][:,perm]);close(g[2],gp[2][perm])
        integer=[np.ones((7,2),int),np.arange(14).reshape(2,7),np.zeros(7,int)];li,gi=numpy_objective_gradient(*integer,ids,target,mask);lf,gf=numpy_objective_gradient(*[a.astype(float) for a in integer],ids,target,mask);close(li,lf)
        for a,b in zip(gi,gf):close(a,b)
    def test_retained_all_final_models_with_independent_scalar_nll(self):
        r=json.loads((ROOT/'outputs/results.json').read_text());c=json.loads((ROOT/'data/corpus.json').read_text());states=json.loads(gzip.decompress((ROOT/'outputs/final_states.json.gz').read_bytes()))
        for state in states:
            tok=ByteBPE(r['tokenizer_audit']['bpe']['merges'] if state['kind']=='byte_bpe' else []);E=np.array(state['E']);W=np.array(state['W']);b=np.array(state['b']);nll=[]
            for text in c['test']:
                seq=tok.encode(text);total=0.
                for a,target in zip(seq,seq[1:]):
                    z=E[a]@W+b;shift=z-z.max();total+=np.log(np.exp(shift).sum())-shift[target]
                nll.append(total)
            record=next(v for v in r['runs'] if (v['kind'],v['seed'])==(state['kind'],state['seed']))['metrics']['test'];close(nll,record['per_document_nll']);close(sum(nll),record['total_nll']);close(E[0],0)
    def test_invalid_domains(self):
        m=ByteBPE()
        for ids in [[1,-1,2],[1,True,2],[1,1.5,2],[1,260,2],[1,0,2]]:
            with self.assertRaises(ValueError):m.decode(ids)
        for corpus,k in [([],1),(['a'],True),(['a'],-1),(['a'],65)]:
            with self.assertRaises(ValueError):ByteBPE.fit(corpus,k)
        with self.assertRaises(ValueError):m.encode('a'*4097)
        with self.assertRaises(ValueError):ByteBPE([[1,4]])
        with self.assertRaises(ValueError):ByteBPE([[4,4]]*65)
        repeated=[[4,4]]+[[259+i,259+i] for i in range(1,14)]
        with self.assertRaises(ValueError):ByteBPE(repeated)
        with self.assertRaises(ValueError):pad_batch([[1,4,2]],260,pad_to=1)
        with self.assertRaises(ValueError):pad_batch([[1,2]],260,pad_to=100000)
        with self.assertRaises(ValueError):numpy_objective_gradient(np.ones((7,2)),np.ones((2,7)),np.zeros(7),np.ones((1,2),int),np.ones((1,2),int),np.zeros((1,2),bool))
        with self.assertRaises(FloatingPointError):numpy_objective_gradient(np.full((7,2),1e308),np.full((2,7),1e308),np.zeros(7),np.ones((1,1),int),np.ones((1,1),int),np.ones((1,1),bool))
    def test_training_failure_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);names=['results.json','training_traces.json.gz','final_states.json.gz']
            for name in names:(out/name).write_bytes(('old '+name).encode())
            with patch.object(torch,'isfinite',return_value=torch.tensor(False)):
                with self.assertRaises(FloatingPointError):experiment.run(out)
            for name in names:self.assertEqual((out/name).read_bytes(),('old '+name).encode())
    def test_data_and_tokenizer_train_only(self):
        c=json.loads((ROOT/'data/corpus.json').read_text());self.assertEqual(c,generate_data.generate());self.assertTrue(set(c['train']).isdisjoint(c['validation']));self.assertTrue(set(c['train']).isdisjoint(c['test']))
        t,h=ByteBPE.fit(c['train'],16);self.assertEqual(len(t.pieces),276)
        for text in c['train']+c['validation']+c['test']:self.assertEqual(t.decode(t.encode(text)),text)
if __name__=='__main__':unittest.main(verbosity=2)
