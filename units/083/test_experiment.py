import unittest,numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):self.X,self.y,self.Xt,self.yt,self.Xr,self.yr,self.W=e.fixture()
 def test_softmax_rows(self):np.testing.assert_allclose(e.softmax([[1000,999]]).sum(1),1)
 def test_softmax_shift(self):np.testing.assert_allclose(e.softmax(self.W),e.softmax(self.W+100))
 def test_gradient(self):
  loss,g=e.loss_grad(self.X,self.y,self.W);h=1e-5
  for i,j in [(0,0),(3,7),(7,11)]:
   a=self.W.copy();b=a.copy();a[i,j]+=h;b[i,j]-=h
   self.assertAlmostEqual(g[i,j],(e.loss_grad(self.X,self.y,a)[0]-e.loss_grad(self.X,self.y,b)[0])/(2*h),places=7)
 def test_mask_zero(self):
  with self.assertRaises(ValueError):e.loss_grad(self.X,self.y,self.W,np.zeros(len(self.y)))
 def test_mask_selection(self):
  m=np.zeros(len(self.y));m[:7]=1
  self.assertAlmostEqual(e.loss_grad(self.X,self.y,self.W,m)[0],e.loss_grad(self.X[:7],self.y[:7],self.W)[0])
 def test_frozen(self):np.testing.assert_array_equal(e.train(self.X,self.y,self.W,'frozen')[0],self.W)
 def test_input_immutable(self):
  old=self.W.copy();e.train(self.X,self.y,self.W,steps=5);np.testing.assert_array_equal(old,self.W)
 def test_full_decreases(self):self.assertLess(e.loss_grad(self.X,self.y,e.train(self.X,self.y,self.W)[0])[0],e.loss_grad(self.X,self.y,self.W)[0])
 def test_rank(self):self.assertLessEqual(np.linalg.matrix_rank(e.train(self.X,self.y,self.W,'lora',2)[0]-self.W,tol=1e-8),2)
 def test_merge(self):
  W,_,A,B=e.train(self.X,self.y,self.W,'lora',2);np.testing.assert_allclose(self.X@W.T,self.X@self.W.T+(self.X@A.T)@B.T,atol=1e-12)
 def test_lora_gradient(self):
  rng=np.random.default_rng(4);A=rng.normal(size=(2,12));B=rng.normal(size=(8,2));G=e.loss_grad(self.X,self.y,self.W+B@A)[1];h=1e-5
  C=B.copy();D=B.copy();C[0,1]+=h;D[0,1]-=h
  numeric=(e.loss_grad(self.X,self.y,self.W+C@A)[0]-e.loss_grad(self.X,self.y,self.W+D@A)[0])/(2*h)
  self.assertAlmostEqual((G@A.T)[0,1],numeric,places=7)
 def test_disjoint(self):self.assertFalse(np.array_equal(self.X[:120],self.Xt))
 def test_repeat(self):np.testing.assert_array_equal(e.train(self.X,self.y,self.W,'lora',steps=10)[0],e.train(self.X,self.y,self.W,'lora',steps=10)[0])
if __name__=='__main__':unittest.main()
