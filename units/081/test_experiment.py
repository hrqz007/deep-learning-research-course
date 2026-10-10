import unittest,json
import numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def test_gradient(self):
  rng=np.random.default_rng(4);t=[rng.normal(size=4) for _ in range(3)]+[np.array([.1])];x=np.array([.12,.38,.79]);_,g=e.loss_grad(t,x)
  for j in range(4):
   for k in range(len(t[j])):
    eps=1e-6;t[j][k]+=eps;plus=e.loss_grad(t,x)[0];t[j][k]-=2*eps;minus=e.loss_grad(t,x)[0];t[j][k]+=eps
    self.assertAlmostEqual((plus-minus)/(2*eps),g[j][k],delta=2e-6)
 def test_input_derivative(self):
  t=[np.array([1.,-.3]),np.array([.1,.7]),np.array([.2,-.4]),np.array([.1])];x=np.linspace(.1,.9,9);eps=1e-4
  u,du,ddu=e.field(t,x)
  np.testing.assert_allclose((e.field(t,x+eps)[0]-e.field(t,x-eps)[0])/(2*eps),du,atol=1e-8)
  np.testing.assert_allclose((e.field(t,x+eps)[0]-2*u+e.field(t,x-eps)[0])/eps**2,ddu,atol=1e-7)
 def test_difference_equations(self):
  x,u=e.finite_difference(31);h=x[1]-x[0]
  np.testing.assert_allclose(2*u[1:-1]-u[:-2]-u[2:],h*h*np.pi**2*np.sin(np.pi*x[1:-1]),atol=1e-13)
 def test_saved_results(self):
  m=json.loads((e.ROOT/'outputs/metrics.json').read_text());self.assertLess(m['pinn']['relative_l2'],.001);self.assertGreater(m['pinn_sparse']['heldout_residual_rmse'],10*m['pinn']['heldout_residual_rmse'])
  self.assertLess(m['fd_127']['relative_l2'],m['fd_63']['relative_l2']/3)

class RobustnessTests(unittest.TestCase):
 def test_invalid_grid_size(self):
  for n in [0,-1,2.5]:
   with self.assertRaises(ValueError):e.finite_difference(n)
 def test_invalid_training(self):
  for n,steps in [(0,10),(3,0),(2.5,20)]:
   with self.assertRaises(ValueError):e.train(n=n,steps=steps)
 def test_nonfinite_input(self):
  t=[np.ones(2),np.ones(2),np.ones(2),np.zeros(1)]
  for x in [np.array([np.nan]),np.array([np.inf]),np.zeros((2,2))]:
   with self.assertRaises(ValueError):e.field(t,x)
 def test_bad_parameter_shapes(self):
  with self.assertRaises(ValueError):e.field([np.ones(2),np.ones(3),np.ones(2),np.zeros(1)],np.array([.5]))
 def test_field_output_shape(self):
  t=[np.ones(2),np.ones(2),np.ones(2),np.zeros(1)]
  for out in e.field(t,np.linspace(0,1,7)):self.assertEqual(out.shape,(7,));self.assertTrue(np.isfinite(out).all())
 def test_fd_zero_boundary_and_balance(self):
  x,u=e.finite_difference(63);h=x[1]-x[0]
  self.assertEqual(u[0],0.);self.assertEqual(u[-1],0.)
  self.assertAlmostEqual((u[1]+u[-2])/h,h*np.sum(np.pi**2*np.sin(np.pi*x[1:-1])),places=10)
 def test_saved_residual_not_claim_only(self):
  z=np.load(e.ROOT/'outputs/pinn_sparse_weights.npz');t=[z[k] for k in ['w','b','v','c']];x=(np.arange(5)+.5)/5
  train_r=-e.field(t,x)[2]-np.pi**2*np.sin(np.pi*x)
  grid=np.linspace(0,1,1001);u,_,ddu=e.field(t,grid);test_r=-ddu-np.pi**2*np.sin(np.pi*grid)
  self.assertLess(np.sqrt(np.mean(train_r**2)),1e-10);self.assertGreater(np.sqrt(np.mean(test_r**2)),.1)
  self.assertGreater(np.linalg.norm(u-np.sin(np.pi*grid))/np.linalg.norm(np.sin(np.pi*grid)),.001)
 def test_saved_all_finite(self):
  data=np.loadtxt(e.ROOT/'data/evaluation.csv',delimiter=',',skiprows=1)
  self.assertEqual(data.shape,(1001,6));self.assertTrue(np.isfinite(data).all())

 def test_all_parameter_groups_update(self):
  t1,_,_,_=e.train(seed=81,n=5,steps=1);t2,_,_,_=e.train(seed=81,n=5,steps=2)
  for a,b in zip(t1,t2):self.assertTrue(np.any(a!=b))

if __name__=='__main__':unittest.main()
