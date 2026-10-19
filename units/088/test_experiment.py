"""Behavioral tests use unittest checks, including under python -O."""
import unittest
import numpy as np
import experiment as e

class TestAlignment(unittest.TestCase):
    def setUp(self):self.d=e.make_data(2,5)
    def test_shapes_balance(self):
        self.assertEqual(self.d['x'].shape,(36,192));self.assertEqual(self.d['t'].shape,(36,10))
        np.testing.assert_array_equal(np.bincount(self.d['y']),np.repeat(2,18))
    def test_seed_reproducible(self):
        b=e.make_data(2,5)
        for k in self.d:np.testing.assert_array_equal(self.d[k],b[k])
    def test_independent_instances(self):
        b=e.make_data(2,6);self.assertEqual(e.audit(self.d,b)['exact_image_overlap'],0)
    def test_text_attribute_groups(self):
        for sl in [slice(0,3),slice(3,6),slice(6,8),slice(8,10)]:np.testing.assert_array_equal(self.d['t'][:,sl].sum(1),np.ones(36))
    def test_normalization(self):
        z,r=e.normalize(np.array([[3.,4.],[0.,0.]]));self.assertTrue(np.isfinite(z).all());np.testing.assert_allclose(z[0],[.6,.8],atol=1e-12)
    def test_softmax_axes(self):
        s=np.array([[1000.,998.],[999.,999.]])
        np.testing.assert_allclose(e.softmax(s,1).sum(1),1);np.testing.assert_allclose(e.softmax(s,0).sum(0),1)
    def test_exact_losses(self):
        s=np.array([[2.,0.],[1.,0.]])
        li=-np.log(np.diag(e.softmax(s,1))).mean();lt=-np.log(np.diag(e.softmax(s,0))).mean()
        self.assertAlmostEqual(li,.7200948492805977);self.assertAlmostEqual(lt,.503204434039084)
    def test_gradients(self):self.assertLess(max(e.finite_difference().values()),1e-6)
    def test_both_towers_update(self):
        before=e.initial(11);p,_,_=e.train(self.d,'image_to_text',11,steps=0)
        # The standard training batch needs >=72 examples.
        d=e.make_data(4,5);after,h,_=e.train(d,'image_to_text',11,steps=8)
        for k in before:self.assertGreater(float(np.linalg.norm(after[k]-before[k])),0)
        self.assertTrue(all(np.isfinite(x['loss']) for x in h))
    def test_parameter_count(self):self.assertEqual(sum(v.size for v in e.initial(11).values()),3232)
    def test_scoring_cannot_read_labels(self):
        p=e.initial(11);s=e.scores(p,self.d);d={k:v.copy() for k,v in self.d.items()};d['y'][:]=-999
        np.testing.assert_array_equal(s,e.scores(p,d))
    def test_retrieval_definitions(self):
        y=np.array([0,0,1]);s=np.array([[1.,2.,0.],[2.,1.,0.],[0.,0.,3.]])
        r=e.retrieval(s,y);self.assertAlmostEqual(r['i2t_pair_r1'],1/3);self.assertEqual(r['i2t_semantic_r1'],1)
    def test_stable_ties(self):
        s=np.ones((3,3));r=e.retrieval(s,np.array([0,1,2]));self.assertAlmostEqual(r['i2t_pair_r1'],1/3)
    def test_false_negative_count(self):
        a=e.audit(self.d,e.make_data(1,6));self.assertEqual(a['off_diagonal_semantic_positive_count'],36);self.assertEqual(a['off_diagonal_count'],36*35)
    def test_repeat_training(self):
        d=e.make_data(4,5);p,_,_=e.train(d,'symmetric',11,steps=3);q,_,_=e.train(d,'symmetric',11,steps=3)
        for k in p:np.testing.assert_array_equal(p[k],q[k])

if __name__=='__main__':unittest.main()
