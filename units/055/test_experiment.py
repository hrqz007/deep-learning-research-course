"""Accounting, causal invariance, real optimizer steps and checkpoint replay."""
import unittest,tempfile,copy,math,json
from pathlib import Path
import numpy as np
import torch
from experiment import *
from generate_data import corpus
class Tests(unittest.TestCase):
    def test_data_and_shift(self):
        rows=corpus();self.assertEqual(len(rows),256);self.assertEqual(len({r['text'] for r in rows}),256)
        self.assertEqual([sum(r['split']==s for r in rows) for s in ['train','validation','test']],[128,64,64])
        x,y=tensors(rows[:2]);self.assertEqual(tuple(x.shape),(2,7));self.assertTrue((x[:,0]==BOS).all());self.assertTrue((y[:,-1]==EOS).all())
        self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))
        for r in rows:
            c,a,v,c2,o,dot=r['tokens'];self.assertEqual(c2-3,((c-3)+(a-7)+(v-11))%4)
    def test_parameters_and_flops(self):
        setup(5501)
        for d,n,f in [(16,5252,1089536),(32,18676,4014080)]:
            m=TinyLM(d);self.assertEqual(sum(p.numel() for p in m.parameters()),n)
            self.assertEqual(parameter_formula(d),n);self.assertEqual(forward_matmul_flops(d),f)
        self.assertEqual(protocol()['aggregate_effective_token_limit'],107520)
    def test_causal_and_length(self):
        setup(5501);m=TinyLM(16).eval();x,y=tensors(corpus()[:2]);changed=x.clone();changed[:,4:]=3
        torch.testing.assert_close(m(x)[:,:4],m(changed)[:,:4],rtol=0,atol=1e-6)
        for length in [1,3,7]:torch.testing.assert_close(m(x[:,:length]),m(x)[:,:length],rtol=0,atol=1e-6)
        with self.assertRaises(ValueError):m(torch.ones(1,8,dtype=torch.long))
        with self.assertRaises(ValueError):TinyLM(15)
    def test_uniform_loss(self):
        y=torch.tensor([[3,7,11,6,15,19,2]])
        self.assertAlmostEqual(nll(torch.zeros(1,7,V),y).item(),math.log(V),places=6)
    def test_training_checkpoint_and_adam(self):
        proto=protocol();proto.update(steps=8,checkpoints=[0,4,8])
        config={'d':16,'pool':32,'seed':5501,'name':'smoke'}
        with tempfile.TemporaryDirectory() as directory:
            obj=generate(Path(directory)/'data');r=run_one(config,Path(directory)/'run',proto,obj)
            self.assertEqual(r['effective_tokens'],896);self.assertEqual(r['steady_effective_tokens'],336)
            self.assertEqual(r['unique_train_documents_seen'],32);self.assertFalse(r['test_set_evaluated'])
            previous_m=previous_v=0.
            for row in r['steps'][:2]:
                e=row['optimizer_example'];t=row['step'];g=e['clipped_gradient']
                self.assertAlmostEqual(e['raw_gradient'],e['analytic_ce_gradient'],places=6)
                self.assertAlmostEqual(g,e['raw_gradient']*min(1.,1./(row['gradient_norm_before_clip']+1e-6)),places=7)
                self.assertAlmostEqual(e['m'],.9*previous_m+.1*g,places=7)
                self.assertAlmostEqual(e['v'],.999*previous_v+.001*g*g,places=9)
                expected=e['before']*(1-.003*.01)-.003*(e['m']/(1-.9**t))/(math.sqrt(e['v']/(1-.999**t))+1e-8)
                self.assertAlmostEqual(e['after'],expected,places=6);previous_m=e['m'];previous_v=e['v']
            m=TinyLM(16)
            with np.load(Path(directory)/'run/weights.npz',allow_pickle=False) as a:
                m.load_state_dict({k:torch.from_numpy(a[k]) for k in a.files})
            vx,vy=tensors([x for x in obj['documents'] if x['split']=='validation'])
            self.assertAlmostEqual(evaluate(m,vx,vy)['nll'],r['final_validation']['nll'],places=7)
            self.assertGreater(r['steady_tokens_per_second'],0.)
    def test_packaged_checkpoints(self):
        # Verify all public checkpoints, counters, complete run matrix and held-out policy.
        results=ROOT/'outputs/results.json'
        self.assertTrue(results.exists(),'run experiment.py once before testing delivered checkpoints')
        r=json.loads(results.read_text());obj=json.loads((ROOT/'data/corpus.json').read_text())
        vx,vy=tensors([x for x in obj['documents'] if x['split']=='validation'])
        self.assertEqual(len(r['runs']),12);self.assertEqual(r['total_effective_training_tokens'],107520)
        for row in r['runs']:
            c=row['config'];m=TinyLM(c['d'])
            with np.load(ROOT/'outputs/runs'/c['name']/'weights.npz',allow_pickle=False) as a:
                m.load_state_dict({k:torch.from_numpy(a[k]) for k in a.files})
            self.assertAlmostEqual(evaluate(m,vx,vy)['nll'],row['final_validation']['nll'],places=6)
            self.assertEqual(row['effective_tokens'],8960);self.assertIsNone(row['gpu_peak_memory_bytes'])
            self.assertFalse(row['test_set_evaluated'])
if __name__=='__main__':setup(5501);unittest.main(verbosity=2)
