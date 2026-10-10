import unittest
import numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def test_row_stochastic(self):self.assertTrue(np.allclose(e.transition(np.ones((3,3))).sum(1),1))
 def test_isolated_node(self):self.assertTrue(np.allclose(e.transition(np.zeros((3,3))),np.eye(3)))
 def test_bad_shape(self):self.assertRaises(ValueError,e.transition,np.zeros((2,3)))
 def test_negative(self):self.assertRaises(ValueError,e.transition,-np.ones((2,2)))
 def test_nan(self):self.assertRaises(ValueError,e.transition,np.full((2,2),np.nan))
 def test_reproducible(self):self.assertTrue(np.array_equal(e.make_data()[0],e.make_data()[0]))
 def test_split(self):self.assertFalse(set(range(100))&set(range(125,150)))
 def test_equivariance(self):
  a,x,_=e.make_data(n_graphs=1);p=[3,2,1,0,7,6,5,4]
  self.assertTrue(np.allclose(e.features(a[0][p][:,p],x[0][p]),e.features(a[0],x[0])[p]))
 def test_gradient(self):
  z=np.array([[.2,.4],[-.3,.1]]);y=np.array([[.3],[-.2]]);p=e.init();_,g=e.loss_grad(z,y,p)
  for k in p:
   for idx in np.ndindex(p[k].shape):
    old=p[k][idx];p[k][idx]=old+1e-6;l1=e.loss_grad(z,y,p)[0];p[k][idx]=old-1e-6;l0=e.loss_grad(z,y,p)[0];p[k][idx]=old
    self.assertAlmostEqual((l1-l0)/2e-6,g[k][idx],places=7)
 def test_training(self):
  a,x,y=e.make_data(n_graphs=3);z=np.concatenate([e.features(ai,xi) for ai,xi in zip(a,x)]);y=y.reshape(-1,1)
  p,h=e.train(z,y,steps=200);self.assertLess(h[-1],h[0])
 def test_consensus(self):
  p=e.transition(np.ones((4,4))-np.eye(4));self.assertLess(np.var(p@np.arange(4.)),1e-20)
if __name__=='__main__':unittest.main()
