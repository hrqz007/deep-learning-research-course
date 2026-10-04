"""Executable numeric and interface tests; assertions here are unittest methods, valid under -O."""
import copy,gzip,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import experiment as e
class CourseTests(unittest.TestCase):
    def test_fixture_hashes(self):self.assertEqual(e.verify_inputs()['unit'],'042')
    def test_entity_disjoint(self):
        d=[e.load_split(s) for s in ('train','validation','test')]
        for a in range(3):
            for b in range(a):self.assertFalse(set(d[a]['groups'])&set(d[b]['groups']))
        self.assertEqual([len(x['y']) for x in d],[224,190,788])
    def test_two_estimands(self):
        groups=['A','A','B'];np.testing.assert_array_equal(e.weights(groups,'entity'),[.25,.25,.5])
        np.testing.assert_allclose(e.weights(groups,'row'),[1/3]*3)
        self.assertEqual(float(e.weights(groups,'entity')@np.array([0,0,2.])),1.)
    def test_duplicate_whole_entity_preserves_risk(self):
        p=[1.,2.,3.];y=[0.,0.,0.];g=['A','A','B']
        _,a=e.entity_losses(p,y,g);_,b=e.entity_losses([1,2,1,2,3],[0]*5,['A']*4+['B'])
        np.testing.assert_array_equal(a,b)
    def test_numeric_guards(self):
        for bad in ([True,1],['1',2],[1,float('nan')],[1,float('inf')],[1,complex(1,2)],[np.longdouble('1e400')],[]):
            with self.assertRaises(ValueError):e.array(bad,'bad',1)
        with self.assertRaises(ValueError):e.real(np.longdouble('1e400'),'bad',0,1)
        for x in [True,1.2,-1,2**32]:
            with self.assertRaises(ValueError):e.initialize(x)
    def test_shape_guards(self):
        for g in ([],['A',1],[''],np.array(['A'])):
            with self.assertRaises(ValueError):e.weights(g,'entity')
        with self.assertRaises(ValueError):e.weights(['A'],'unknown')
        with self.assertRaises(ValueError):e.validate_split({'x':[[1,2]],'y':[1,2],'groups':['A']})
    def test_normalization_train_only(self):
        t=e.load_split('train');n=e.normalization(t);q=e.weights(t['groups'],'entity');z=e.transform(t['x'],n)
        np.testing.assert_allclose(q@z,[0,0],atol=2e-16);np.testing.assert_allclose(q@(z*z),[1,1],atol=3e-15)
        bad=dict(t,x=np.ones_like(t['x']))
        with self.assertRaises(ValueError):e.normalization(bad)
    def test_hand_exact_two_rounds(self):
        hand=e.hand_trace()
        self.assertEqual(hand['steps'][0]['data_loss'],e.F(55,128));self.assertEqual(hand['steps'][0]['objective'],e.F(331,640))
        for s in hand['steps']:
            self.assertEqual(s['after'],[p-g/10 for p,g in zip(s['theta'],s['gradient'])])
            for row in s['rows']:
                theta=torch.tensor([float(x) for x in s['theta']],dtype=torch.float64,requires_grad=True)
                h=torch.relu(theta[:2]*float(row['x'])+theta[2:4]);p=h@theta[4:6]+theta[-1]
                loss=float(row['weight'])*(p-float(row['y'])).square()/2
                g=torch.autograd.grad(loss,theta)[0].numpy();np.testing.assert_allclose(g,[float(x) for x in row['contribution']],atol=2e-15,rtol=0)
        self.assertLess(hand['third_objective'],hand['steps'][1]['objective'])
    def test_full_gradient(self):
        rng=np.random.default_rng(4260);maxerr=0
        for seed in range(8):
            x=rng.normal(size=(7,2));y=rng.normal(size=7);q=e.weights(['A','A','B','B','B','C','D'],'entity');p=e.initialize(seed)
            t=torch.tensor(p,dtype=torch.float64,requires_grad=True);pred,_=e.forward(t,torch.tensor(x));loss=(torch.tensor(q)*(pred-torch.tensor(y)).square()).sum()/2+.002*e.penalty(t)
            g=torch.autograd.grad(loss,t)[0].numpy();manual=e.numpy_gradient(p,x,y,q,.002);np.testing.assert_allclose(g,manual,atol=5e-15,rtol=1e-13)
            def objective(z):return q@((e.numpy_forward(z,x)-y)**2)/2+.001*(z[:12]@z[:12]+z[18:24]@z[18:24])
            for j in range(25):
                h=1e-5;plus=p.copy();minus=p.copy();plus[j]+=h;minus[j]-=h;fd=(objective(plus)-objective(minus))/(2*h);maxerr=max(maxerr,abs(fd-g[j]))
        self.assertLess(maxerr,1e-8)
    def test_paired_initialization_and_budget(self):
        t=e.load_split('train');v=e.load_split('validation');a=e.fit(t,v,'row',.1,4211,2);b=e.fit(t,v,'entity',.1,4211,2)
        self.assertEqual(a['parameters'][0],b['parameters'][0]);self.assertEqual(a['normalization'],b['normalization']);self.assertEqual(a['row_presentations'],b['row_presentations'])
        self.assertFalse(np.array_equal(a['parameters'][1],b['parameters'][1]))
    def test_heldout_labels_do_not_change_updates(self):
        t=e.load_split('train');v=e.load_split('validation');vv=dict(v,y=-v['y']);a=e.fit(t,v,'entity',.1,4211,3);b=e.fit(t,vv,'entity',.1,4211,3)
        self.assertEqual(a['parameters'],b['parameters']);self.assertNotEqual(a['curve'][-1]['validation_entity_half_mse'],b['curve'][-1]['validation_entity_half_mse'])
    def test_ridge_stationarity(self):
        t=e.load_split('train');v=e.load_split('validation');n=e.normalization(t);r=e.ridge(t,v,3,.001,n);X,_=e.design(e.transform(t['x'],n),3);beta=np.array(r['coefficients']);pen=beta.copy();pen[0]=0
        np.testing.assert_allclose(X.T@(e.weights(t['groups'],'entity')*(X@beta-t['y']))+.001*pen,0,atol=2e-14)
    def test_bootstrap_unit(self):
        r=e.paired_bootstrap([1,1,1],10,42);self.assertEqual(r['percentile_95'],[1,1]);self.assertEqual(r['mean'],1.)
        for reps in [True,0,10.1]:
            with self.assertRaises(ValueError):e.paired_bootstrap([1,2],reps)
    def test_identity_mechanism(self):
        r=e.identity_leakage();self.assertEqual(len(r['labels']),80);self.assertAlmostEqual(r['seen_theoretical_half_mse'],.006666666666666667)
        self.assertAlmostEqual(r['unseen_theoretical_half_mse'],.5133541666666667);self.assertLess(r['seen_empirical_half_mse'],r['unseen_empirical_half_mse'])
    def test_gzip_deterministic_lossless(self):
        raw=e.json_bytes({'x':[0,1,.5]});a=e.gzip_bytes(raw);self.assertEqual(a,e.gzip_bytes(raw));self.assertEqual(gzip.decompress(a),raw);self.assertEqual(a[:10],bytes.fromhex('1f8b08000000000002ff'))
    def test_json_forbids_nonfinite(self):
        for x in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):e.json_bytes({'x':x})
    def test_output_guard_prewrite(self):
        for dest in [e.ROOT,e.ROOT/'data',e.ROOT/'figures',e.ROOT/'experiment.py']:
            with patch.object(e,'select_networks',side_effect=RuntimeError('must not train')):
                with self.assertRaises(ValueError):e.run(dest)
    def test_corrupt_fixture_prewrite(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'FIXTURES',dict(e.FIXTURES,**{'train.csv':'0'*64})):
            path=Path(tmp)/'new';
            with self.assertRaises(ValueError):e.run(path)
            self.assertFalse(path.exists())
if __name__=='__main__':unittest.main(verbosity=2)
