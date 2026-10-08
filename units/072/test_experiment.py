"""Tests exercise data, intervention, reporting and actual PyTorch training."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
import experiment as e

torch.set_num_threads(1)

class ProjectTests(unittest.TestCase):
    def test_data_shapes(self):
        x,y,g=e.generate(12,1)
        self.assertEqual(x.shape,(12,1,12,12));self.assertEqual(y.dtype,np.int64)
        self.assertTrue(np.isin(g,[0,1]).all())
    def test_data_reproducibility(self):
        for a,b in zip(e.generate(20,1),e.generate(20,1)):np.testing.assert_array_equal(a,b)
    def test_data_seed_separation(self):
        self.assertFalse(np.array_equal(e.generate(20,1)[0],e.generate(20,2)[0]))
    def test_shift_changes_only_declared_corner(self):
        x,_,_=e.generate(20,3);d=e.domains_from(x)
        np.testing.assert_array_equal(d['flip'][:,:,3:,:],x[:,:,3:,:])
        np.testing.assert_array_equal(d['flip'][:,:,:3,:3],-x[:,:,:3,:3])
    def test_shift_preserves_input(self):
        x,_,_=e.generate(20,3);copy=x.copy();e.domains_from(x)
        np.testing.assert_array_equal(x,copy)
    def test_augmentation_is_label_independent(self):
        x=torch.zeros(16,1,12,12);g=torch.Generator().manual_seed(1)
        z=e.transform(x,'random_corner',g,True)
        self.assertEqual(set(z[:,:,0,0].ravel().tolist()),{-2.2,2.2} if z.dtype==torch.float64 else set(torch.tensor([-2.2,2.2]).tolist()))
        self.assertTrue(torch.equal(x,torch.zeros_like(x)))
    def test_augmentation_preserves_core(self):
        x=torch.randn(16,1,12,12);g=torch.Generator().manual_seed(1)
        z=e.transform(x,'random_corner',g,True)
        self.assertTrue(torch.equal(z[:,:,3:10,3:10],x[:,:,3:10,3:10]))
    def test_wrong_corner_preserves_shortcut(self):
        x=torch.randn(16,1,12,12);g=torch.Generator().manual_seed(1)
        z=e.transform(x,'wrong_corner',g,True)
        self.assertTrue(torch.equal(z[:,:,:3,:3],x[:,:,:3,:3]))
    def test_mask_applied_at_inference(self):
        x=torch.ones(2,1,12,12);z=e.transform(x,'mask_corner',training=False)
        self.assertEqual(float(z[:,:,:3,:3].sum()),0.)
        self.assertEqual(float(e.transform(x,'random_corner',training=False).sum()),288.)
    def test_missing_rng_rejected(self):
        with self.assertRaises(ValueError): e.transform(torch.zeros(2,1,12,12),'random_corner',training=True)
    def test_parameter_counts(self):
        counts={v:sum(p.numel() for p in e.network(v).parameters()) for v in e.VARIANTS}
        self.assertEqual(counts['linear'],290);self.assertEqual(counts['erm'],4706)
        self.assertEqual(counts['erm'],counts['random_corner'])
    def test_metric_numerators(self):
        p=np.array([[.9,.1],[.2,.8]]);r=e.metrics(p,np.array([0,0]))
        self.assertEqual(r['errors'],1);self.assertEqual(r['error'],.5)
    def test_invalid_metrics_raise(self):
        with self.assertRaises(ValueError): e.metrics(np.array([[.9,.9]]),np.array([0]))
        with self.assertRaises(ValueError): e.metrics(np.array([[.9,.1]]),np.array([2]))
    def test_selective_empty(self):
        p=np.array([[.9,.1],[.2,.8]]);r=e.selective(p,np.array([0,1]),1.)
        self.assertEqual(r['coverage'],0.);self.assertIsNone(r['risk'])
    def test_curve_ties(self):
        p=np.array([[.8,.2],[.2,.8]]);r=e.curve(p,np.array([0,0]))
        self.assertEqual(r['coverage'],[1.]);self.assertEqual(r['risk'],[.5])
    def test_training_reproducibility(self):
        x,y,_=e.generate(30,8)
        a,l1,_=e.fit(x,y,'random_corner',9,steps=5)
        b,l2,_=e.fit(x,y,'random_corner',9,steps=5)
        self.assertEqual(l1,l2)
        np.testing.assert_array_equal(e.predict(a,x,'random_corner'),e.predict(b,x,'random_corner'))
    def test_checkpoint_roundtrip(self):
        x,y,_=e.generate(30,8);m,losses,_=e.fit(x,y,'erm',9,steps=5)
        self.assertLess(losses[-1],losses[0])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weights.pt';torch.save({'state_dict':m.state_dict()},p)
            other=e.network('erm');other.load_state_dict(torch.load(p,weights_only=True)['state_dict'])
            np.testing.assert_array_equal(e.predict(m,x,'erm'),e.predict(other,x,'erm'))

if __name__=='__main__':unittest.main()
