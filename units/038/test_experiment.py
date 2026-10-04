"""Independent rational, autograd, LAPACK and spectral checks for unit 038."""
import copy,hashlib,json,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import experiment as e

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.fixture=e.load_fixture();cls.rows,cls.details=e.capacity_runs(cls.fixture)
    def test_01_exact_rational_two_steps(self):
        x=[[F(1),F(0),F(1)],[F(0),F(1),F(1)]];y=[F(1),F(2)];w=[F(0)]*3;trace=e.hand_trace()
        for record in trace['steps']:
            pred=[sum(a*b for a,b in zip(row,w)) for row in x];r=[a-b for a,b in zip(pred,y)];pieces=[[r[i]*x[i][j]/2 for j in range(3)] for i in range(2)];g=[sum(p[j] for p in pieces) for j in range(3)];after=[a-F(1,2)*b for a,b in zip(w,g)]
            for key,val in [('prediction',pred),('contributions',pieces),('gradient',g),('after',after)]:np.testing.assert_allclose(record[key],np.array(val,dtype=float),atol=1e-15)
            self.assertEqual(float(sum(z*z for z in r)/4),record['loss']);w=after
        self.assertEqual(trace['third_loss'],float(F(45,1024)))
    def test_02_all_linear_coordinate_derivatives(self):
        rng=np.random.default_rng(38100)
        for _ in range(32):
            x=rng.normal(size=(3,5));y=rng.normal(size=3);w=rng.normal(size=5);s=e.linear_step(x,y,w,.1)
            for j in range(5):
                z=w.astype(complex);z[j]+=1e-25j;g=np.imag(np.mean((x@z-y)**2)/2)/1e-25
                self.assertAlmostEqual(g,s['gradient'][j],places=12)
    def test_03_duplication_all_parameter_and_sample_gradients(self):
        for width in (1,2):
            s=e.neural_step(width);v=torch.tensor(s['before'],dtype=torch.float64,requires_grad=True);x=torch.tensor(e.X_HAND);y=torch.tensor(e.Y_HAND)
            w=v[:3*width].reshape(width,3);b=v[3*width:4*width];a=v[4*width:5*width];c=v[-1];pred=torch.relu(x@w.T+b)@a+c;losses=(pred-y)**2/4
            for i in range(2):
                g=torch.autograd.grad(losses[i],v,retain_graph=True)[0].numpy();np.testing.assert_allclose(g,s['sample_contributions'][i],atol=1e-15)
            g=torch.autograd.grad(losses.sum(),v)[0].numpy();np.testing.assert_allclose(g,s['gradient'],atol=1e-15)
            nv=v.detach()-.2*torch.tensor(g);npred=torch.relu(x@nv[:3*width].reshape(width,3).T+nv[3*width:4*width])@nv[4*width:5*width]+nv[-1]
            np.testing.assert_allclose(npred,s['next_prediction'],atol=1e-15)
    def test_04_duplicate_function_global(self):
        rng=np.random.default_rng(38101);x=rng.normal(size=(100,3));h=np.maximum(x@np.array([.5,1,0])+.5,0)
        np.testing.assert_allclose(h,.5*h+.5*h,atol=0)
        self.assertNotEqual(e.neural_step(1)['next_loss'],e.neural_step(2)['next_loss'])
    def test_05_null_component_and_limits(self):
        d=e.implicit_bias();z=d['null_direction'];np.testing.assert_allclose(e.X_HAND@z,0,atol=0);np.testing.assert_allclose(d['minimum_norm'],[0,1,1],atol=1e-14)
        for p in d['paths']:
            for row in p['rows']:self.assertAlmostEqual(row['w']@z/3,p['null_coefficient'],places=13)
            np.testing.assert_allclose(p['rows'][-1]['w'],p['limit'],atol=2e-10)
    def test_06_rescaling_changes_implicit_solution(self):
        d=e.implicit_bias();np.testing.assert_allclose(d['rescaled_function_coefficients'],[-1/3,2/3,4/3],atol=1e-14);np.testing.assert_allclose(d['rescaled_training_predictions'],e.Y_HAND,atol=1e-14)
    def test_07_all_1944_fits_independent_lstsq(self):
        for row,detail in zip(self.rows,self.details):
            x=np.array(self.fixture['draws'][row['seed']-3800]['x'])[:,:row['p']];y=np.array(detail['y']);lam=row['ridge'];p=row['p']
            if lam:x=np.vstack([x,np.sqrt(e.N*lam)*np.eye(p)]);y=np.concatenate([y,np.zeros(p)])
            ref=np.linalg.lstsq(x,y,rcond=1e-12)[0]
            np.testing.assert_allclose(detail['weights'],ref,rtol=2e-9,atol=2e-10)
    def test_08_all_risk_decompositions(self):
        beta=np.zeros(e.P_MAX);beta[:3]=[1,-1,.5]
        for r,d in zip(self.rows,self.details):
            w=np.zeros(e.P_MAX);w[:r['p']]=d['weights'];self.assertAlmostEqual(np.sum((w-beta)**2),r['clean_risk'],places=8)
            self.assertAlmostEqual(r['signal_error']+r['realized_noise_norm2']+r['cross_term'],r['clean_risk'],places=8)
            self.assertAlmostEqual(r['noisy_risk']-r['clean_risk'],r['sigma']**2,places=10)
    def test_09_ranks_and_interpolation(self):
        for row in self.rows:
            self.assertEqual(row['rank'],min(e.N,row['p']))
            if row['ridge']==0 and row['p']>=e.N:self.assertLess(row['train_mse'],1e-19)
            if row['ridge']==0 and row['sigma']==0 and 3<=row['p']<=e.N:self.assertLess(row['clean_risk'],1e-20)
    def test_10_noise_expectation_matrix_trace(self):
        for dd in self.fixture['draws'][:3]:
            for p in (3,20,24,32,96):
                x=np.array(dd['x'])[:,:p];a=np.linalg.pinv(x,rcond=1e-12)
                s=e.least_squares(x,np.zeros(e.N));self.assertAlmostEqual(np.sum(a*a),np.sum(s['filter']**2),places=8)
    def test_11_conditional_monte_carlo_is_disclosed(self):
        d=e.conditional_demo();self.assertEqual(len(d['noise_risks']),400);self.assertGreater(d['noise_mcse'],0);self.assertLess(abs(d['noise_mean']-d['noise_expected_risk']),5*d['noise_mcse']);self.assertLess(abs(d['test_clean_mse']-d['exact_fixed_fit_risk']),5*d['test_mcse'])
    def test_12_spectral_filters_actual_updates(self):
        d=e.spectral_demo();x=d['x'];y=d['y'];w=np.zeros(2);saved={r['steps']:r for r in d['gd']}
        for t in range(1001):
            if t in saved:np.testing.assert_allclose(w,saved[t]['w'],rtol=1e-12,atol=1e-12)
            w=w-d['lr']*x.T@(x@w-y)/2
    def test_13_rank_deficiency_and_rcond(self):
        x=np.array([[1.,1.,0.],[2.,2.,0.]]);y=np.array([1.,2.]);s=e.least_squares(x,y);self.assertEqual(s['rank'],1);np.testing.assert_allclose(s['w'],[.5,.5,0],atol=1e-14)
        x=np.diag([1.,1e-14]);s=e.least_squares(x,[1,1]);self.assertEqual(s['rank'],1);np.testing.assert_allclose(s['w'],[1,0],atol=1e-14)
    def test_14_invalid_numeric_inputs(self):
        bad=[(np.ones((0,2)),[],0,1e-12),(e.X_HAND,[1],0,1e-12),(e.X_HAND,[1,np.nan],0,1e-12),(e.X_HAND,e.Y_HAND,-1,1e-12),(e.X_HAND,e.Y_HAND,0,0)]
        for x,y,l,r in bad:
            with self.assertRaises(ValueError):e.least_squares(x,y,l,r)
        with self.assertRaises(ValueError):e.linear_step(e.X_HAND,e.Y_HAND,[0,0,0],float('inf'))
    def test_15_fixture_hash_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.json';p.write_text('{}')
            with self.assertRaises(ValueError):e.load_fixture(p)
    def test_16_before_write_guards(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'absent'
            with patch.object(e,'calculate',side_effect=ValueError('calculation failed')):
                with self.assertRaises(ValueError):e.run(out)
            self.assertFalse(out.exists());out.mkdir();marker=out/'keep';marker.write_text('old')
            with patch.object(e,'calculate',return_value=({'bad.json':float('nan')},[{'x':1}])):
                with self.assertRaises(ValueError):e.run(out)
            self.assertEqual(list(out.iterdir()),[marker]);self.assertEqual(marker.read_text(),'old')
    def test_17_bound_formula_and_vacuity(self):
        rows=e.bound_demo();self.assertEqual(len(rows),12);self.assertGreater(next(r['epsilon'] for r in rows if r['n']==5 and r['M']==1000000),1)
        for r in rows:self.assertAlmostEqual(2*r['M']*np.exp(-2*r['n']*r['epsilon']**2),.05,places=13)
    def test_18_fixture_generation(self):
        for d in self.fixture['draws']:
            rng=np.random.default_rng(d['seed']);np.testing.assert_array_equal(rng.normal(size=(e.N,e.P_MAX)),d['x']);np.testing.assert_array_equal(rng.normal(size=e.N),d['epsilon'])

if __name__=='__main__':unittest.main(verbosity=2)
