"""Tests remain active under python -O."""
import unittest
import numpy as np
import experiment as e

class TestFusion(unittest.TestCase):
    def setUp(self):self.d=e.make_data(96,19)
    def test_balanced_data(self):
        self.assertEqual(self.d['images'].shape,(96,8,8,3));np.testing.assert_array_equal(np.bincount(self.d['y']),[48,48])
    def test_true_labels(self):np.testing.assert_array_equal(self.d['y'],self.d['colors'][np.arange(96),self.d['side']])
    def test_reproducible_data(self):
        other=e.make_data(96,19)
        for k in self.d:np.testing.assert_array_equal(self.d[k],other[k])
    def test_tokens(self):
        t=e.tokenize(self.d['images']);self.assertEqual(t.shape,(96,2,5));np.testing.assert_array_equal(t[:,:,-2:],np.broadcast_to(np.eye(2),(96,2,2)))
    def test_interventions_do_not_mutate(self):
        original={k:v.copy() for k,v in self.d.items()}
        for domain in e.DOMAINS:e.intervene(self.d,domain)
        for k in original:np.testing.assert_array_equal(original[k],self.d[k])
    def test_conflict(self):
        d=e.intervene(self.d,'conflict_hint');np.testing.assert_array_equal(d['h'].argmax(1),1-self.d['y']);np.testing.assert_array_equal(d['images'],self.d['images'])
    def test_mask_exact_region(self):
        d=e.intervene(self.d,'target_occluded')
        for i,side in enumerate(d['side']):
            self.assertEqual(float(np.abs(d['images'][i,:,side*4:(side+1)*4]).sum()),0)
            other=1-side;np.testing.assert_array_equal(d['images'][i,:,other*4:(other+1)*4],self.d['images'][i,:,other*4:(other+1)*4])
    def test_blank_hint(self):self.assertEqual(float(e.intervene(self.d,'target_occluded_blank_hint')['h'].sum()),0)
    def test_question_counterfactual(self):
        d=e.intervene(self.d,'question_flip');np.testing.assert_array_equal(d['y'],self.d['colors'][np.arange(96),1-self.d['side']]);np.testing.assert_array_equal(d['h'],self.d['h'])
    def test_gradients(self):self.assertLess(max(e.gradient_check().values()),1e-6)
    def test_parameter_counts(self):
        for m,n in [('text_only',114),('image_only',210),('concat_biased',274),('attention_biased',268),('attention_random_hint',268)]:self.assertEqual(sum(v.size for v in e.initial(m,1).values()),n)
    def test_attention_probability(self):
        p=e.initial('attention_biased',1);_,c=e.predict(p,self.d,'attention_biased');np.testing.assert_allclose(c['a'].sum(1),1);self.assertTrue((c['a']>=0).all())
    def test_text_ignores_image(self):
        p=e.initial('text_only',1);a=e.predict(p,self.d,'text_only')[0];b=e.predict(p,e.intervene(self.d,'image_blank'),'text_only')[0];np.testing.assert_array_equal(a,b)
    def test_image_ignores_text(self):
        p=e.initial('image_only',1);a=e.predict(p,self.d,'image_only')[0];b=e.predict(p,e.intervene(self.d,'question_flip'),'image_only')[0];np.testing.assert_array_equal(a,b)
    def test_predict_cannot_read_labels(self):
        p=e.initial('attention_biased',1);a=e.predict(p,self.d,'attention_biased')[0];d={k:v.copy() for k,v in self.d.items()};d['y'][:]=-100;np.testing.assert_array_equal(a,e.predict(p,d,'attention_biased')[0])
    def test_training_updates_attention(self):
        before=e.initial('attention_biased',11);p,h,_=e.train(self.d,'attention_biased',11,steps=5)
        for k in before:self.assertGreater(float(np.linalg.norm(p[k]-before[k])),0)
    def test_sample_prediction_independent_of_batch(self):
        p=e.initial('attention_biased',1);a=e.predict(p,self.d,'attention_biased')[0]
        d={k:v[:1].copy() for k,v in self.d.items()};b=e.predict(p,d,'attention_biased')[0];np.testing.assert_allclose(a[:1],b,atol=1e-14)
    def test_repeat_training(self):
        p,_,_=e.train(self.d,'attention_random_hint',11,steps=3);q,_,_=e.train(self.d,'attention_random_hint',11,steps=3)
        for k in p:np.testing.assert_array_equal(p[k],q[k])

if __name__=='__main__':unittest.main()
