"""Behavioral tests remain active under python -O (unittest assertions)."""
import unittest, tempfile
from pathlib import Path
import numpy as np
import torch
import experiment as e

torch.set_num_threads(1)

class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[.9,.1],[.2,.8],[.7,.3],[.4,.6]])
        self.y=np.array([0,0,0,1])
    def test_selective_denominator(self):
        r=e.selective(self.p,self.y,.75)
        self.assertEqual(r['accepted'],2);self.assertEqual(r['risk'],.5);self.assertEqual(r['coverage'],.5)
    def test_empty_acceptance_is_undefined(self):
        r=e.selective(self.p,self.y,1.)
        self.assertIsNone(r['risk']);self.assertIsNone(r['risk_wilson95'])
    def test_all_accept_equals_error(self):
        r=e.selective(self.p,self.y,0)
        self.assertEqual(r['risk'],.25)
    def test_threshold_equality_included(self):
        self.assertEqual(e.selective(self.p,self.y,.8)['accepted'],2)
    def test_invalid_probabilities_raise(self):
        with self.assertRaises(ValueError): e.selective(self.p*2,self.y,.5)
        with self.assertRaises(ValueError): e.selective(np.full((4,2),np.nan),self.y,.5)
    def test_invalid_shapes_and_threshold(self):
        with self.assertRaises(ValueError): e.selective(self.p,self.y[:2],.5)
        with self.assertRaises(ValueError): e.selective(self.p,self.y,1.1)
    def test_softmax_temperature_preserves_argmax(self):
        z=np.array([[1.,-1.],[0.,3.]])
        np.testing.assert_array_equal(e.softmax(z,.5).argmax(1),e.softmax(z,3).argmax(1))
        with self.assertRaises(ValueError):e.softmax(z,0)
    def test_temperature_search_does_not_increase_grid_best(self):
        z=np.log(self.p);t,r=e.calibrate(z,self.y)
        self.assertAlmostEqual(e.nll(e.softmax(z,t),self.y),min(r['nll']))
    def test_curve_counts_ties_together(self):
        p=np.array([[.8,.2],[.2,.8],[.7,.3]])
        r=e.risk_coverage(p,[0,0,0])
        self.assertEqual(r['coverage'],[2/3,1.]);self.assertEqual(r['risk'],[.5,1/3])
    def test_coverage_quantile_includes_ties(self):
        tau=e.threshold_for_coverage(self.p,.5)
        self.assertGreaterEqual(e.selective(self.p,self.y,tau)['coverage'],.5)
    def test_wilson_zero_errors_nonzero_upper(self):
        interval=e.wilson(0,10)
        self.assertGreater(interval[1],0.);self.assertAlmostEqual(interval[0],0.)
    def test_ece_uses_all_bins(self):
        result=e.reliability(self.p,self.y)
        self.assertEqual(sum(x['n'] for x in result['bins']),4)
        self.assertGreaterEqual(result['ece'],0.)
    def test_pairing_and_mutation_boundary(self):
        x,y,g=e.generate(60,7);before=x.copy();domains=e.shift_domains(x,g)
        np.testing.assert_array_equal(x,before)
        np.testing.assert_array_equal(domains['shortcut_flip'][:,0],x[:,0])
        np.testing.assert_array_equal(domains['shortcut_flip'][:,1],-x[:,1])
        np.testing.assert_array_equal(domains['subgroup_flip'][g==0],x[g==0])
    def test_rng_reproducibility(self):
        a=e.generate(20,7);b=e.generate(20,7)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertFalse(np.array_equal(a[0],e.generate(20,8)[0]))
    def test_entropy_disagreement_nonnegative(self):
        p=np.array([[[.9,.1],[.5,.5]],[[.1,.9],[.5,.5]]])
        delta=e.entropy(p.mean(0))-e.entropy(p).mean(0)
        self.assertGreater(delta[0],0.);self.assertAlmostEqual(delta[1],0.)
    def test_training_and_inference_checkpoint(self):
        x,y,_=e.generate(30,3);m,losses,_=e.fit(x,y,3,steps=6)
        self.assertLess(losses[-1],losses[0])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weights.pt';torch.save({'state_dict':m.state_dict()},p)
            other=e.model();other.load_state_dict(torch.load(p,weights_only=True)['state_dict'])
            np.testing.assert_array_equal(e.logits(m,x),e.logits(other,x))

if __name__=='__main__': unittest.main()
