import unittest
import numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def test_orthogonal(self):
  q=e.rotation(.4);self.assertTrue(np.allclose(q.T@q,np.eye(2)))
 def test_group_composition(self):self.assertTrue(np.allclose(e.rotation(.3)@e.rotation(.2),e.rotation(.5)))
 def test_origin(self):self.assertTrue(np.allclose(e.truth(np.zeros((1,2))),0))
 def test_invalid(self):self.assertRaises(ValueError,e.truth,np.zeros((3,3)))
 def test_nan(self):self.assertRaises(ValueError,e.truth,np.array([[np.nan,0]]))
 def test_rotation(self):
  x,_=e.data();q=e.rotation(.6);self.assertTrue(np.allclose(e.truth(x@q.T),e.truth(x)@q.T))
 def test_reflection(self):
  x,_=e.data();q=np.diag([-1,1]);p=e.init('equivariant');self.assertTrue(np.allclose(e.predict(x@q,p,'equivariant'),e.predict(x,p,'equivariant')@q))
 def test_learned_equivariance(self):
  x,_=e.data();q=e.rotation(.95);p=e.init('equivariant');self.assertTrue(np.allclose(e.predict(x@q.T,p,'equivariant'),e.predict(x,p,'equivariant')@q.T))
 def test_gradient(self):
  x,y=e.data(n=3)
  for kind in ['equivariant','ordinary']:
   p=e.init(kind);_,g=e.loss_grad(x,y,p,kind)
   for k in p:
    for idx in np.ndindex(p[k].shape):
     old=p[k][idx];p[k][idx]=old+1e-6;l1=e.loss_grad(x,y,p,kind)[0];p[k][idx]=old-1e-6;l0=e.loss_grad(x,y,p,kind)[0];p[k][idx]=old
     self.assertAlmostEqual((l1-l0)/2e-6,g[k][idx],places=7)
 def test_translation_relative(self):
  x,_=e.data();c=np.array([8.,-5.]);self.assertTrue(np.allclose(e.truth((x+c)-c),e.truth(x)))
 def test_training(self):
  x,y=e.data(n=32);p,h=e.train(x,y,'equivariant',steps=200);self.assertLess(h[-1],h[0])
if __name__=='__main__':unittest.main()
