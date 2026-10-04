"""Scientific and execution contracts. Runtime checks do not use bare assert."""
from pathlib import Path
from fractions import Fraction as F
import unittest,tempfile,json,gzip,hashlib,subprocess,sys,shutil,copy
from unittest import mock
import numpy as np
import torch
import experiment as e

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.plan=e.verify_inputs();cls.train=e.load_split('train');cls.val=e.load_split('validation')
    def test_hand_two_complete_updates(self):
        r=e.hand_trace();self.assertEqual(r['steps'][0]['loss'],F(29,800));self.assertEqual(r['steps'][1]['loss'],F(43901069,3200000000))
        self.assertLess(r['third_loss'],r['steps'][1]['loss']);self.assertEqual(r['steps'][0]['after'],r['steps'][1]['theta'])
        for st in r['steps']:
            self.assertEqual([sum(row['sample_parameter_contributions'][j] for row in st['rows']) for j in range(7)],st['gradient'])
            self.assertEqual([p-st['lr']*g for p,g in zip(st['theta'],st['gradient'])],st['after'])
    def test_hand_per_sample_autograd(self):
        for st in e.hand_trace()['steps']:
            for row in st['rows']:
                p=torch.tensor([float(v) for v in st['theta']],dtype=torch.float64,requires_grad=True)
                h=torch.relu(p[:2]*float(row['x'])+p[2:4]);pred=(h*p[4:6]).sum()+p[6]
                loss=(pred-float(row['y']))**2/4;g=torch.autograd.grad(loss,p)[0]
                np.testing.assert_allclose(g,[float(v) for v in row['sample_parameter_contributions']],atol=2e-16,rtol=0)
    def test_hand_next_forward_uses_new_weights(self):
        st=e.hand_trace()['steps'][0];new=st['after'];n=e.hand_step(new)
        self.assertEqual(n['rows'][0]['prediction'],F(-5957,40000));self.assertEqual(n['rows'][1]['prediction'],F(47233,40000))
        wrong=[p-st['lr']*2*g for p,g in zip(st['theta'],st['gradient'])]
        self.assertNotEqual(wrong,e.hand_trace()['steps'][1]['after'])
    def test_numpy_torch_forward(self):
        for width in (0,2,8):
            p=e.initialize(width,4001);x=np.linspace(-2,2,7);t,_=e.forward(torch.tensor(p),torch.tensor(x),width)
            np.testing.assert_allclose(t,e.numpy_forward(p,x,width),atol=2e-15,rtol=1e-14)
    def test_weight_only_penalty(self):
        p=torch.ones(7,dtype=torch.float64,requires_grad=True);g=torch.autograd.grad(e.penalty(p,2),p)[0]
        np.testing.assert_array_equal(g,[1,1,0,0,1,1,0])
    def test_initialization_is_paired(self):
        self.assertEqual(e.initialize(8,4001).tolist(),e.initialize(8,4001).tolist());self.assertNotEqual(e.initialize(8,4001).tolist(),e.initialize(8,4002).tolist())
    def test_all_split_ids_disjoint(self):
        test=e.load_split('test');self.assertFalse(set(self.train['ids'])&set(self.val['ids']));self.assertFalse(set(test['ids'])&(set(self.train['ids'])|set(self.val['ids'])))
    def test_training_only_normalization(self):
        c=dict(self.plan['search'][0],budget=2);v=copy.deepcopy(self.val);v['x']=v['x']+100
        a=e.fit(self.train,self.val,c,4001);b=e.fit(self.train,v,c,4001)
        self.assertEqual(a['normalization'],b['normalization']);self.assertEqual(a['parameters'],b['parameters']);self.assertNotEqual(a['curve'][-1]['validation_data_loss'],b['curve'][-1]['validation_data_loss'])
    def test_short_run_deterministic_and_counts(self):
        c=dict(self.plan['search'][0],budget=3);a=e.fit(self.train,self.val,c,4001);b=e.fit(self.train,self.val,c,4001)
        self.assertEqual(a,b);self.assertEqual(a['updates_applied'],3);self.assertEqual(len(a['parameters']),4);self.assertEqual(a['sample_presentations'],144)
    def test_search_does_not_load_test(self):
        p=copy.deepcopy(self.plan);p['search']=[dict(c,budget=2) for c in p['search'][:2]];p['seeds']=[4001]
        with mock.patch.object(e,'load_split',side_effect=RuntimeError('unauthorized data access')):
            r=e.search(self.train,self.val,p)
        self.assertFalse(r['test_data_used']);self.assertEqual(len(r['runs']),2)
    def test_tie_rule_uses_protocol_order(self):
        p=copy.deepcopy(self.plan);c=dict(p['search'][0],budget=2);p['search']=[dict(c,id='z_first'),dict(c,id='a_second')];p['seeds']=[4001]
        r=e.search(self.train,self.val,p);self.assertEqual(r['selected_id'],'z_first')
    def test_failure_kept_without_padding(self):
        c=next(c for c in self.plan['diagnostics'] if c['id']=='unstable');r=e.fit(self.train,self.val,c,4001)
        self.assertEqual(r['status'],'stopped_candidate_loss_limit');self.assertEqual(r['updates_applied'],2);self.assertEqual(len(r['curve']),3);self.assertGreater(r['failure']['candidate_objective'],1e6)
        self.assertEqual(r['parameters'][-1],r['final_parameters']);self.assertEqual(r['failure']['attempted_update'],3)
    def test_boolean_string_nonfinite_rejected(self):
        for v in [[True,1],['1',2],[0,float('nan')],[],[[1],[2]]]:
            with self.assertRaises(ValueError):e.vector(v,'test')
    def test_config_domain_checks(self):
        for key,value in [('lr',True),('lr',float('nan')),('width',1.5),('budget',0),('l2',-1),('input_scale',0),('label_mode','mystery')]:
            c=dict(self.plan['search'][0]);c[key]=value
            with self.assertRaises(ValueError):e.validate_config(c)
    def test_shape_and_constant_feature_rejected(self):
        for train in [{'x':[1,1],'y':[1,2]},{'x':[1,2],'y':[1]}]:
            with self.assertRaises(ValueError):e.fit(train,self.val,self.plan['search'][0],4001)
    def test_paired_bootstrap_identity(self):
        r=e.paired_bootstrap([-.2,-.2,-.2],100,4);np.testing.assert_allclose(r['interval_percentile_95'],[-.2,-.2],atol=1e-15)
        self.assertEqual(r,e.paired_bootstrap([-.2,-.2,-.2],100,4))
    def test_bernoulli_selection_exact_enumeration(self):
        import itertools
        for k in (1,2,4,8):
            vals=[min(a) for a in itertools.product((0,1),repeat=k)]
            self.assertEqual(sum(vals)/len(vals),2.**(-k))
    def test_lossless_deterministic_gzip(self):
        raw=e.json_bytes({'x':[-0.,1.2345678901234567]});a=e.gzip_bytes(raw);self.assertEqual(a,e.gzip_bytes(raw));self.assertEqual(gzip.decompress(a),raw);self.assertEqual(a[:10].hex(),'1f8b08000000000002ff')
    def test_modified_fixture_before_output_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'unit';shutil.copytree(e.ROOT/'data',root/'data');out=Path(tmp)/'output';out.mkdir();sentinel=out/'results.json';sentinel.write_text('SENTINEL')
            for name in e.FIXTURES:
                p=root/'data'/name;original=p.read_bytes();p.write_bytes(original+b' ')
                with mock.patch.object(e,'ROOT',root):
                    with self.assertRaises(ValueError):e.run(out)
                    with self.assertRaises(ValueError):e.run(Path(tmp)/'absent-output')
                self.assertEqual(sentinel.read_text(),'SENTINEL');self.assertFalse((Path(tmp)/'absent-output').exists());p.write_bytes(original)
    def test_missing_fixture_before_output_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'unit';shutil.copytree(e.ROOT/'data',root/'data');(root/'data/train.csv').unlink();out=Path(tmp)/'out'
            with mock.patch.object(e,'ROOT',root):
                with self.assertRaises(FileNotFoundError):e.run(out)
            self.assertFalse(out.exists())
    def test_serialization_rejects_nonfinite(self):
        with self.assertRaises(ValueError):e.json_bytes({'bad':float('nan')})
    def test_atomic_write_replaces_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'OLD');e.atomic_write(p,b'NEW');self.assertEqual(p.read_bytes(),b'NEW');self.assertEqual([f.name for f in p.parent.iterdir()],['x'])
    def test_protocol_search_budget(self):
        self.assertEqual(len(self.plan['search']),9);self.assertEqual(sum(c['budget']*len(self.plan['seeds']) for c in self.plan['search']),4860)
        self.assertEqual(len(set(c['id'] for c in self.plan['search'])),9)

if __name__=='__main__':unittest.main(verbosity=2)
