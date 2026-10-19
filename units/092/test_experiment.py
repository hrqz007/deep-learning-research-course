import unittest
import numpy as np
from experiment import data,features,fit,domains,sigmoid,paired_bootstrap,METHODS
class TestEvidence(unittest.TestCase):
    def setUp(self): self.x,self.y,self.g,self.a=data(200,5)
    def test_shape(self): self.assertEqual(self.x.shape,(200,3))
    def test_reproduce(self): np.testing.assert_array_equal(self.x,data(200,5)[0])
    def test_labels(self): np.testing.assert_array_equal(self.y,(self.a*self.x[:,1]>0).astype(float))
    def test_group(self): self.assertEqual(set(self.g),{0,1})
    def test_features(self):
        self.assertEqual([features(self.x,m).shape[1] for m in METHODS],[2,3,4,5,5])
    def test_interaction(self): np.testing.assert_allclose(features(self.x,'interaction')[:,3],self.x[:,0]*self.x[:,1])
    def test_counterfactual(self):
        ds=domains(self.x,self.y,self.a);np.testing.assert_array_equal(ds['question_counterfactual'][1],1-self.y)
    def test_permutation_labels(self):
        x,y=domains(self.x,self.y,self.a)['permuted_relabelled'];np.testing.assert_array_equal(y,(self.a*x[:,1]>0).astype(float))
    def test_flip_keeps_label(self): np.testing.assert_array_equal(domains(self.x,self.y,self.a)['shortcut_flip'][1],self.y)
    def test_training_determinism(self):
        np.testing.assert_array_equal(fit(self.x,self.y,'interaction',11,20)[0],fit(self.x,self.y,'interaction',11,20)[0])
    def test_loss_decreases(self):
        _,l=fit(self.x,self.y,'interaction',11,100);self.assertLess(l[-1],l[0])
    def test_gradient(self):
        f=features(self.x,'interaction');w=np.arange(5)*.03;p=sigmoid(f@w);analytic=f.T@(p-self.y)/len(p);numeric=[]
        def loss(v): s=f@v;return np.mean(np.logaddexp(0,s)-self.y*s)
        for j in range(5):
            d=np.zeros(5);d[j]=1e-5;numeric.append((loss(w+d)-loss(w-d))/2e-5)
        np.testing.assert_allclose(analytic,numeric,atol=1e-8)
    def test_bootstrap_constant(self): np.testing.assert_array_equal(paired_bootstrap(np.zeros(20),repeats=20),[0,0])
if __name__=='__main__': unittest.main()
