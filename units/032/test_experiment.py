"""DL032 independent finite-population and scalar-loop training checks."""
import copy,itertools,json,math,random,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
import experiment as e

def scalar_reference(theta,x,y):
    values=[];gradient=[0.]*13
    for row,target in zip(x,y[:,0]):
        a=[sum(float(theta[2*j+k])*float(row[k]) for k in range(2))+float(theta[6+j]) for j in range(3)]
        h=[math.tanh(z) for z in a];p=sum(h[j]*float(theta[9+j]) for j in range(3))+float(theta[12]);r=p-float(target);values.append(r*r/2)
        dp=r/len(x);gradient[12]+=dp
        for j in range(3):
            gradient[9+j]+=dp*h[j];da=dp*float(theta[9+j])*(1-h[j]*h[j]);gradient[6+j]+=da
            for k in range(2):gradient[2*j+k]+=da*float(row[k])
    return math.fsum(values)/len(x),np.array(gradient)

class SamplingAndSGDTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.x,cls.y,cls.orders,cls.cfg=e.load_inputs();cls.theta=np.array(cls.cfg['initial_parameters'],dtype=np.float64)
 def test_01_network_all_coordinates(self):
  rng=np.random.default_rng(32)
  for _ in range(80):
   theta=rng.uniform(-.7,.7,13);x=rng.uniform(-2,2,(int(rng.integers(1,17)),2));y=rng.uniform(-2,2,(len(x),1))
   loss,g,trace=e.numpy_forward_backward(theta,x,y);ref,rg=scalar_reference(theta,x,y)
   self.assertAlmostEqual(loss,ref,places=14);np.testing.assert_allclose(g,rg,atol=2e-15,rtol=2e-13)
   t=torch.tensor(theta,requires_grad=True);l=e.torch_loss(t,torch.tensor(x),torch.tensor(y));l.backward();np.testing.assert_allclose(t.grad.numpy(),rg,atol=2e-15,rtol=2e-13)
 def test_02_first_trace_finite_difference(self):
  tr=e.first_batch_trace(self.x,self.y,self.orders,self.cfg);ids=tr['ids'];x=self.x[ids];y=self.y[ids]
  for j in range(13):
   p=self.theta.copy();m=p.copy();p[j]+=1e-5;m[j]-=1e-5
   fd=(scalar_reference(p,x,y)[0]-scalar_reference(m,x,y)[0])/2e-5
   self.assertAlmostEqual(fd,tr['gradient'][j],places=9)
  l,g=scalar_reference(tr['updated'],x,y);self.assertAlmostEqual(l,tr['next_loss'],places=15)
  np.testing.assert_allclose(tr['updated'],self.theta-self.cfg['learning_rate']*tr['gradient'],atol=0,rtol=0)
 def test_03_fraction_covariance_all_schemes(self):
  rng=random.Random(32);checked=0
  for n in range(2,7):
   for _ in range(8):
    pop=[[F(rng.randint(-6,6),4),F(rng.randint(-6,6),4)] for _ in range(n)]
    mean=[sum(p[j] for p in pop)/n for j in range(2)]
    population=[[sum((p[j]-mean[j])*(p[k]-mean[k]) for p in pop)/n for k in range(2)] for j in range(2)]
    for b in range(1,min(n,3)+1):
     for replace in (False,True):
      choices=list(itertools.product(range(n),repeat=b) if replace else itertools.combinations(range(n),b))
      sums=[[sum(pop[i][j] for i in ids)/b for j in range(2)] for ids in choices]
      actual_mean,actual=e.matrix_stats(sums)
      factor=F(1,b) if replace else F(n-b,b*(n-1))
      self.assertEqual(actual_mean,mean);self.assertEqual(actual,[[z*factor for z in row] for row in population]);checked+=1
  self.assertEqual(checked,224)
 def test_04_fixed_exact_examples(self):
  s=e.sampling_exact();self.assertEqual(s['population_mean'],['0','0']);self.assertEqual(s['population_covariance'],[['5','3/2'],['3/2','5/2']])
  two=[v for v in s['cases'] if v['batch_size']==2]
  self.assertEqual(two[0]['covariance_exact'][0][0],'5/2');self.assertEqual(two[1]['covariance_exact'][0][0],'5/3')
  self.assertEqual(s['nonuniform']['unweighted_expected_gradient'],'-1');self.assertEqual(s['nonuniform']['importance_corrected_expectation'],'0')
  for row in s['reshuffle_conditional_counterexample']:self.assertNotEqual(row['next_gradient'],row['full_gradient_here'])
 def test_05_exact_noise_moments_by_path_enumeration(self):
  # All 4^t IID paths, one sample per step, compared with recurrence at rational eta.
  targets=list(map(F,[-3,-1,1,3]));eta=F(1,2)
  for steps in range(1,6):
   values=[]
   for path in itertools.product(targets,repeat=steps):
    theta=F(1)
    for a in path:theta=(1-eta)*theta+eta*a
    values.append(theta)
   mean=sum(values)/len(values);variance=sum((v-mean)**2 for v in values)/len(values)
   expected_mean=(1-eta)**steps;expected_var=eta**2*5*sum((1-eta)**(2*k) for k in range(steps))
   self.assertEqual(mean,expected_mean);self.assertEqual(variance,expected_var)
  rows=e.scalar_noise_theory()
  for row in rows:
   t=row['step'];eta=row['learning_rate'];b=row['batch_size'];expected=(1-eta)**t
   self.assertAlmostEqual(row['mean_theta'],expected,places=9)
   variance=eta*eta*5/b*sum((1-eta)**(2*k) for k in range(t))
   self.assertAlmostEqual(row['variance_theta'],variance,delta=1e-10*max(1.,variance))
 def test_06_all_main_trajectories_scalar_reference(self):
  data=e.run_report();runs=json.loads(data['trajectories.json']);flat=[i for q in self.orders for i in q];summ=json.loads(data['summary.json'])
  for row in summ['runs']:
   run=runs[row['run']];theta=self.theta.copy();b=row['batch_size'];lr=row['learning_rate']
   self.assertEqual(run['used_ids'],flat[:row['examples']]);self.assertEqual(row['examples'],row['steps']*b)
   for step in range(row['steps']):
    ids=flat[step*b:(step+1)*b];loss,g=scalar_reference(theta,self.x[ids],self.y[ids]);theta-=lr*g
    np.testing.assert_allclose(theta,run['history'][step+1]['parameters'],atol=5e-14,rtol=5e-13)
    full_loss,_=scalar_reference(theta,self.x,self.y)
    self.assertAlmostEqual(full_loss,run['history'][step+1]['full_training_loss'],places=13)
 def test_07_fixed_theta_finite_population(self):
  s=e.network_fixed_theta_sampling(self.x,self.y,self.theta)
  independent=np.stack([scalar_reference(self.theta,self.x[i:i+1],self.y[i:i+1])[1] for i in range(8)])
  np.testing.assert_allclose(independent,s['per_example_gradients'],atol=1e-15,rtol=1e-13)
  for row in s['cases']:self.assertLess(row['maximum_covariance_error'],2e-16)
  self.assertEqual(s['cases'][-1]['covariance_trace'],0.)
 def test_08_invalid_function_inputs(self):
  for theta,x,y in [(self.theta.astype('float32'),self.x,self.y),(self.theta,self.x,self.y[:,0]),(self.theta*float('nan'),self.x,self.y),(self.theta*1000,self.x,self.y)]:
   with self.assertRaises(ValueError):e.numpy_forward_backward(theta,x,y)
  for b,steps,lr in [(True,1,.02),(0,1,.02),(33,1,.02),(4,0,.02),(4,1001,.02),(4,4,0),(4,4,float('nan')),(4,4,True)]:
   with self.assertRaises(ValueError):e.train(self.x,self.y,self.orders,self.theta,b,steps,lr)
  bad=copy.deepcopy(self.orders);bad[0][0]=True
  with self.assertRaises(ValueError):e.train(self.x,self.y,bad,self.theta,4,4,.02)
 def test_09_no_output_on_changed_fixed_inputs(self):
  for name in e.INPUT_HASHES:
   for exists in (False,True):
    with tempfile.TemporaryDirectory() as tmp:
     p=Path(tmp);d=p/'data';d.mkdir()
     for n in e.INPUT_HASHES:(d/n).write_bytes((e.ROOT/'data'/n).read_bytes())
     (d/name).write_bytes((d/name).read_bytes()+b' ');out=p/'out'
     if exists:out.mkdir();(out/'sentinel').write_text('keep')
     with self.assertRaises(ValueError):e.write_report(out,d)
     self.assertEqual(out.exists(),exists)
     if exists:self.assertEqual([q.name for q in out.iterdir()],['sentinel'])
if __name__=='__main__':unittest.main()
