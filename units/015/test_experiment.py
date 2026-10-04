"""Independent scalar/Fraction oracles; no production oracle reuse."""
from fractions import Fraction as F
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import experiment as e

H = [[13, -12], [-12, 13]]


def oracle_q(x, y):
    return (13*x*x - 24*x*y + 13*y*y)/2


def oracle_point(x, y, eta, k):
    a = (x+y)*(1-eta)**k
    b = (x-y)*(1-25*eta)**k
    return (a+b)/2, (a-b)/2


class Tests(unittest.TestCase):
    def assertClose(self, a, b, atol=2e-11, rtol=2e-11):
        self.assertTrue(np.allclose(a, b, atol=atol, rtol=rtol), (a, b))

    def test_01_fraction_scalar_oracle(self):
        count = 0
        for a in range(-5, 6):
            for b in range(-4, 5):
                x, y = F(a, 3), F(b, 5)
                self.assertClose(e.quadratic([[float(x), float(y)]], H), float(oracle_q(x,y)))
                self.assertClose(e.gradient([[float(x), float(y)]], H), [[float(13*x-12*y), float(-12*x+13*y)]])
                count += 1
        self.assertEqual(count, 99)

    def test_02_fraction_trajectory_oracle(self):
        count = 0
        for eta in [F(1,50),F(1,25),F(3,40),F(2,25),F(9,100),F(0),F(-1,100)]:
            series = e.trajectory([[2,0]], H, float(eta), 40)
            for k, r in enumerate(series):
                x,y = oracle_point(F(2),F(0),eta,k)
                self.assertClose([r['x'],r['y']],[float(x),float(y)],rtol=2e-10)
                self.assertClose(r['loss'],float(oracle_q(x,y)),rtol=2e-10)
                count += 1
        self.assertEqual(count,287)

    def test_03_taylor_exact_remainders(self):
        for h in [F(1,5),F(1,10),F(1,20),F(1,40),F(-1,10)]:
            first,second=e.local_models(1.5,[[4,1]],[[12,0],[0,1]],[[float(h),float(2*h)]])
            actual=e.polynomial([[float(1+h),float(1+2*h)]])
            self.assertClose(actual-first,float(8*h*h+4*h**3+h**4),atol=1e-14)
            self.assertClose(actual-second,float(4*h**3+h**4),atol=1e-14)
        for d in [[[.1,.2]],[[-.3,.4]],[[2,-3]]]:
            p=np.array([[2.,0.]])
            _,pred=e.local_models(e.quadratic(p,H),e.gradient(p,H),H,d)
            self.assertClose(pred,e.quadratic(p+np.asarray(d),H))

    def test_04_spectrum_shapes(self):
        values,vectors=e.spectrum(H)
        self.assertClose(values,[1,25]);self.assertClose(vectors.T@vectors,np.eye(2))
        self.assertClose(vectors@np.diag(values)@vectors.T,H)
        self.assertClose(np.asarray(H)@vectors,vectors*values)
        self.assertEqual(e.gradient([[2,0]],H).shape,(1,2))
        self.assertEqual(e.spd_bounds(H)['eta_strict_upper'],.08)
        self.assertEqual(e.spd_bounds(H)['condition_number'],25)

    def test_05_step_boundaries_and_initial_directions(self):
        for eta in [.02,.04,.075,1/13]:
            losses=[r['loss'] for r in e.trajectory([[2,0]],H,eta,60)]
            self.assertTrue(all(a>b for a,b in zip(losses,losses[1:])))
        boundary=e.trajectory([[2,0]],H,.08,60)
        for k,r in enumerate(boundary):self.assertClose(r['loss'],25+.92**(2*k))
        self.assertGreater(boundary[-1]['loss'],25)
        beyond=e.trajectory([[2,0]],H,.08005,500)
        self.assertLess(beyond[1]['loss'],26)
        self.assertGreater(beyond[-1]['loss'],26)
        stable_exception=e.trajectory([[1,1]],H,.09,30)
        self.assertLess(stable_exception[-1]['loss'],.004)
        self.assertClose(e.trajectory([[2,0]],H,0,3)[-1]['loss'],26)
        self.assertGreater(e.trajectory([[2,0]],H,.09,60)[-1]['loss'],1e13)

    def test_06_scaling_and_stationarity(self):
        Q=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
        S=np.diag([1.,5.]);Si=np.diag([1.,.2])
        theta=np.array([[2.,0.]]);z=theta@Q;w=z@S
        self.assertClose(.5*np.sum(w*w),26)
        gw=e.gradient(theta,H)@Q@Si
        self.assertClose(gw,w)
        self.assertClose((w-gw)@Si@Q.T,[[0,0]])
        self.assertClose(Si@Q.T@np.asarray(H)@Q@Si,np.eye(2))
        # This independent polynomial supplies non-global local-minimum witness.
        G=lambda x,y:x*x-3*x**4+x**6+y*y
        self.assertEqual(G(0,0),0);self.assertEqual(G(1,0),-1)
        self.assertTrue(all(G(x,0)>0 for x in [.01,-.01,.1,-.1]))
        self.assertEqual(e.steepening([[0,0]]),0);self.assertEqual(e.steepening([[-1,0]]),9.5)
        self.assertClose(e.quadratic([[3,-2]],[[2,3],[3,4]]),-1)

    def test_07_hessian_finite_difference(self):
        for h in [.1,.01,.001]:
            self.assertClose(e.finite_difference_hessian(lambda p:e.gradient(p,H),[[.7,-.4]],h),H,atol=1e-8)
            found=e.finite_difference_hessian(e.polynomial_gradient,[[1,1]],h)
            self.assertClose(found,[[12+4*h*h,0],[0,1]],atol=1e-8)
        calls=[]
        with self.assertRaises(ArithmeticError):
            e.finite_difference_hessian(lambda p:calls.append(1),[[1,1]],1e308)
        self.assertEqual(calls,[])
        with self.assertRaises(ArithmeticError):e.finite_difference_hessian(e.polynomial_gradient,[[1,1]],1e-20)
        for h in [0,-.1,float('nan'),True]:
            with self.assertRaises((ValueError,TypeError)):e.finite_difference_hessian(e.polynomial_gradient,[[1,1]],h)

    def test_08_invalid_inputs(self):
        bad=[ [1,2], [[1],[2]], [[1,2,3]], [[True,1]], [[np.array(True),1]],
              np.array([[True,False]]), [[1+0j,2]], [['1','2']], [[np.nan,0]], [[np.inf,0]],
              np.array([[object(),1]],dtype=object), [[np.array([True]),1]]]
        for p in bad:
            with self.subTest(p=str(p)),self.assertRaises((ValueError,TypeError)):
                e.gradient(p,H)
        for h in [[[1,2],[3,4]], [[1,0]], [[True,0],[0,1]], [[1,np.inf],[np.inf,1]]]:
            with self.assertRaises((ValueError,TypeError)):e.gradient([[1,1]],h)
        for h in [[[1,0],[0,0]],[[-1,0],[0,2]]]:
            with self.assertRaises(ValueError):e.spd_bounds(h)
        for n in [True,1.,-1,5001]:
            with self.assertRaises((TypeError,ValueError)):e.trajectory([[1,1]],H,.01,n)
        for eta in [True,[],[.1],1+0j,np.nan,np.inf]:
            with self.assertRaises((TypeError,ValueError)):e.step([[1,1]],H,eta)
        if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
            with self.assertRaises(ValueError):e.row(np.array([['1e400','0']],dtype=np.longdouble))

    def test_09_arithmetic_range_failures(self):
        funcs=[lambda:e.quadratic([[1e308,1e308]],H),
               lambda:e.gradient([[1e308,-1e308]],H),
               lambda:e.step([[2,0]],H,1e308),
               lambda:e.polynomial([[1e100,0]]),
               lambda:e.polynomial_gradient([[1e200,0]]),
               lambda:e.polynomial_hessian([[1e200,0]]),
               lambda:e.steepening([[1e100,0]]),
               lambda:e.local_models(0,[[1e308,0]],[[1,0],[0,1]],[[2,0]])]
        for f in funcs:
            with self.assertRaises(ArithmeticError):f()
        self.assertClose(e.quadratic([[0,0]],H),0)
        # Explicitly documented finite-precision limitation, not exact optimality.
        self.assertEqual(e.quadratic([[1e-200,0]],H),0.0)

    def test_10_fresh_process_determinism_and_config_failure(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            for suffix in ['one','two']:
                subprocess.run([sys.executable,str(e.ROOT/'experiment.py'),'--output',str(base/suffix)],check=True,capture_output=True)
            for name in ['results.json','taylor.csv','trajectories.csv','environment.json']:
                self.assertEqual((base/'one'/name).read_bytes(),(base/'two'/name).read_bytes())
            config=json.loads((e.ROOT/'data/config.json').read_text())
            config['step_sizes']=[1e308]
            snapshots={name:(base/'one'/name).read_bytes() for name in ['results.json','taylor.csv','trajectories.csv','environment.json']}
            for iterations in [0, 60]:
                config['iterations']=iterations
                cf=base/'bad.json';cf.write_text(json.dumps(config))
                for target in [base/'failed',base/'one']:
                    bad=subprocess.run([sys.executable,str(e.ROOT/'experiment.py'),'--config',str(cf),'--output',str(target)],capture_output=True)
                    self.assertNotEqual(bad.returncode,0)
                    self.assertFalse((base/'failed').exists())
                    for name,contents in snapshots.items():
                        self.assertEqual((base/'one'/name).read_bytes(),contents)
            with (base/'one'/'trajectories.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),305)

    def test_11_document_numbers(self):
        text=(e.ROOT/'lecture.md').read_text()
        for fragment in ['7.2104','19.99625','25.8464','39.8906','31252','0.0801229','9.5','0.0841','0.0041']:
            self.assertIn(fragment,text)
        first=e.step([[2,0]],H,.075)
        self.assertClose(first,[[.05,1.8]])
        self.assertClose(e.quadratic(first,H),19.99625)
        self.assertLess(float(F(1601,20000)),float(F(626,7813)))
        self.assertGreater(float(F(1601,20000)),.08)


if __name__=='__main__':
    unittest.main(verbosity=2)
