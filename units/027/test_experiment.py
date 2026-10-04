"""Independent author checks. Run with python test_experiment.py (also works under -O)."""
import copy
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import experiment as e

D=Decimal

class Dual:
    """Decimal forward-mode oracle: no production backprop or CE functions."""
    def __init__(self,v,d): self.v=D(str(v)); self.d=list(d)
    def __add__(self,o):
        if not isinstance(o,Dual): o=Dual(o,[D(0)]*len(self.d))
        return Dual(self.v+o.v,[a+b for a,b in zip(self.d,o.d)])
    __radd__=__add__
    def __mul__(self,o):
        if not isinstance(o,Dual): o=Dual(o,[D(0)]*len(self.d))
        return Dual(self.v*o.v,[a*o.v+self.v*b for a,b in zip(self.d,o.d)])
    __rmul__=__mul__
    def __neg__(self): return self*(-1)
    def __sub__(self,o): return self+-o
    def exp(self):
        v=self.v.exp(); return Dual(v,[v*a for a in self.d])
    def log(self): return Dual(self.v.ln(),[a/self.v for a in self.d])
    def reciprocal(self): return Dual(1/self.v,[-a/self.v**2 for a in self.d])
    def tanh(self):
        q=(2*self).exp(); return (q-1)*(q+1).reciprocal()


def dual_reference(X,y,params,activation='tanh'):
    coords=list(e.coordinates(params)); npar=len(coords); loc={c:i for i,c in enumerate(coords)}
    def parameter(j,key,index):
        d=[D(0)]*npar;d[loc[j,key,index]]=D(1)
        return Dual(params[j][key][index],d)
    losses=[]
    with localcontext() as ctx:
        ctx.prec=70
        for x,lab in zip(X,y):
            h=[Dual(v,[D(0)]*npar) for v in x]
            for j,p in enumerate(params):
                z=[sum((h[r]*parameter(j,'W',(r,k)) for r in range(len(h))),parameter(j,'b',(k,))) for k in range(len(p['b']))]
                if j<len(params)-1:
                    h=[v.tanh() if activation=='tanh' else (v if v.v>0 else Dual(0,[D(0)]*npar)) if activation=='relu' else v for v in z]
            losses.append(sum(v.exp() for v in z).log()-z[int(lab)])
        total=sum(losses)*(D(1)/len(losses))
    return float(total.v),np.array([float(v) for v in total.d])

class Tests(unittest.TestCase):
    def setUp(self): self.ids,self.X,self.y,self.params,self.cfg=e.load_inputs()
    def test_hand_network(self):
        X=np.array([[1.,2.],[-1.,1.]])
        p=[{'W':np.array([[1.,1.],[0.,0.]]),'b':np.array([2.,2.])},
           {'W':np.array([[1.,-1.],[-1.,1.]]),'b':np.zeros(2)}]
        r=e.loss_and_grad(X,np.array([0,0]),p,'relu')
        self.assertAlmostEqual(r['loss'],np.log(2),15)
        np.testing.assert_array_equal(r['grads'][0]['W'],[[0,0],[-1.5,1.5]])
        np.testing.assert_array_equal(r['grads'][0]['b'],[-1,1])
        np.testing.assert_array_equal(r['grads'][1]['W'],[[-1,1],[-1,1]])
        np.testing.assert_array_equal(r['grads'][1]['b'],[-.5,.5])
        np.testing.assert_array_equal(r['dX'],np.zeros((2,2)))
        self.assertTrue(all(v['pass'] for v in e.finite_difference(X,[0,0],p,'relu')))
    def test_decimal_forward_mode_networks(self):
        rng=np.random.default_rng(2701)
        for trial in range(72):
            widths=[2]+[int(rng.integers(2,5)) for _ in range(trial%3)]+[3]
            p=[{'W':rng.integers(-6,7,size=(a,b))/20,'b':rng.integers(-3,4,size=b)/20} for a,b in zip(widths,widths[1:])]
            X=rng.integers(-10,11,size=(trial%5+1,2))/10;y=rng.integers(0,3,size=len(X))
            # Smooth tanh is the reference objective; some one-layer cases have no activation.
            loss,grad=dual_reference(X,y,p)
            actual=e.loss_and_grad(X,y,p)
            np.testing.assert_allclose(actual['loss'],loss,rtol=2e-14,atol=2e-14)
            np.testing.assert_allclose(e.flat(actual['grads']),grad,rtol=2e-12,atol=2e-14)
            samples=e.scalar_sample_oracle(X,y,p)
            np.testing.assert_allclose(samples.mean(0),grad,rtol=2e-12,atol=2e-14)
    def test_all_parameter_fd_and_input_fd(self):
        rng=np.random.default_rng(2702)
        for trial in range(36):
            p=copy.deepcopy(self.params)
            for layer in p:
                for k in layer: layer[k] += rng.uniform(-.08,.08,size=layer[k].shape)
            rows=e.finite_difference(self.X,self.y,p,h=1e-5)
            self.assertEqual(len(rows),26);self.assertTrue(all(r['pass'] for r in rows))
            r=e.loss_and_grad(self.X,self.y,p)
            for idx in np.ndindex(self.X.shape):
                xp=self.X.copy();xm=self.X.copy();xp[idx]+=1e-5;xm[idx]-=1e-5
                fd=(e.loss_only(xp,self.y,p)-e.loss_only(xm,self.y,p))/(2e-5)
                self.assertAlmostEqual(r['dX'][idx],fd,places=8)
    def test_reductions_permutations_duplicates_and_chunks(self):
        m=e.loss_and_grad(self.X,self.y,self.params);g=e.flat(m['grads']);n=len(self.X)
        s=e.loss_and_grad(self.X,self.y,self.params,reduction='sum')
        np.testing.assert_allclose(e.flat(s['grads']),n*g,atol=2e-15)
        for order in ([4,3,2,1,0],[1,3,0,4,2]):
            r=e.loss_and_grad(self.X[order],self.y[order],self.params)
            np.testing.assert_allclose(e.flat(r['grads']),g,atol=2e-15)
        d=e.loss_and_grad(np.repeat(self.X,3,axis=0),np.repeat(self.y,3),self.params)
        np.testing.assert_allclose(e.flat(d['grads']),g,atol=2e-15)
        for cut in range(1,n):
            a=e.flat(e.loss_and_grad(self.X[:cut],self.y[:cut],self.params)['grads'])
            b=e.flat(e.loss_and_grad(self.X[cut:],self.y[cut:],self.params)['grads'])
            np.testing.assert_allclose(cut/n*a+(n-cut)/n*b,g,atol=2e-15)
        self.assertGreater(np.max(abs((a+b)/2-g)),1e-3)
    def test_fraction_affine_and_broadcast(self):
        rng=np.random.default_rng(2703)
        for _ in range(120):
            n,d,h=map(int,rng.integers(1,5,size=3));X=rng.integers(-6,7,size=(n,d));G=rng.integers(-6,7,size=(n,h));W=rng.integers(-6,7,size=(d,h))
            reference=np.array([[float(sum(Fraction(int(X[i,r]))*int(G[i,k]) for i in range(n))) for k in range(h)] for r in range(d)])
            np.testing.assert_array_equal(X.T@G,reference)
            np.testing.assert_array_equal(e.unbroadcast(G,(h,)),[sum(int(G[i,k]) for i in range(n)) for k in range(h)])
        g=np.arange(24.).reshape(2,3,4)
        for shape in ((),(4,),(1,4),(3,1),(1,3,1),(2,1,4),(2,3,4)):
            actual=e.unbroadcast(g,shape)
            ref=np.zeros(shape)
            for index in np.ndindex(g.shape):
                aligned=(1,)*(g.ndim-len(shape))+shape
                idx=tuple(0 if a==1 else i for a,i in zip(aligned,index))
                ref[idx[-len(shape):] if shape else ()]+=g[index]
            np.testing.assert_array_equal(actual,ref)
    def test_ce_decimal_extreme(self):
        vectors=[[0,0,0],[1000,1001,-1000],[37,0,-1],[-1000,-1000,-1000],[10000,9999,9998]]
        rng=np.random.default_rng(2704)
        vectors += [rng.integers(-100,101,size=3).tolist() for _ in range(45)]
        with localcontext() as ctx:
            ctx.prec=100
            for values in vectors:
                ex=[D(v).exp() for v in values];den=sum(ex)
                for y in range(3):
                    loss,delta,_,p=e.cross_entropy([values],[y])
                    ref=den.ln()-D(values[y]);g=[v/den for v in ex];g[y]-=1
                    self.assertAlmostEqual(loss,float(ref),places=10)
                    np.testing.assert_allclose(delta[0],list(map(float,g)),rtol=2e-14,atol=1e-90)
        loss,d,_,p=e.cross_entropy([[37.,0.]],[0])
        self.assertGreater(loss,0);self.assertLess(d[0,0],0)
        self.assertEqual(float(p[0,0]-1),0.)
    def test_relu_kink_is_not_smooth_fd(self):
        self.assertEqual((max(1e-5,0)-max(-1e-5,0))/(2e-5),.5)
        self.assertEqual(float(0>0),0)
        # Away from kinks, check an entire small network, including inactive units.
        p=[{'W':np.array([[.5,-.2],[.3,.4]]),'b':np.array([2.,-2.])},self.params[-1]]
        self.assertTrue(all(r['pass'] for r in e.finite_difference(self.X,self.y,p,'relu')))
    def test_invalid_math_and_conversion(self):
        bad=[lambda:e.real_array([True,1.],'x'),lambda:e.real_array(['1'],'x'),lambda:e.real_array([np.nan],'x'),lambda:e.real_array([np.inf],'x'),lambda:e.real_array([1+2j],'x'),lambda:e.real_array([], 'x'),lambda:e.real_array(np.array([np.longdouble('1e-400')]),'x'),lambda:e.read_number('1e-400'),lambda:e.read_number('NaN'),lambda:e.unbroadcast(np.ones((2,3)),(2,)),lambda:e.unbroadcast(np.ones((2,3)),(0,)),lambda:e.unbroadcast(np.ones((2,3)),(True,3)),lambda:e.cross_entropy(np.ones((2,3)),[[0],[1]]),lambda:e.cross_entropy(np.ones((2,3)),[0.,1.]),lambda:e.cross_entropy(np.ones((2,3)),[True,1]),lambda:e.cross_entropy(np.ones((2,3)),[0,3]),lambda:e.cross_entropy(np.ones((2,3)),[0,1],'none'),lambda:e.cross_entropy(np.ones((2,1)),[0,0]),lambda:e.loss_and_grad(self.X[:0],self.y[:0],self.params),lambda:e.loss_and_grad(self.X,self.y,self.params,'sigmoid'),lambda:e.finite_difference(self.X,self.y,self.params,h=0),lambda:e.validate_params([{'W':[[1,2]],'b':[[0,0]]}]),lambda:e.validate_params([])]
        for f in bad:
            with self.assertRaises((ValueError,TypeError)): f()
        self.assertEqual(e.read_number('0e-400'),0.)
        np.testing.assert_array_equal(e.real_array([0.,-0.],'zero'),[0.,0.])
    def test_bad_input_does_not_touch_outputs(self):
        source=(e.HERE/'data/batch.csv').read_text();cfg=json.loads((e.HERE/'data/config.json').read_text())
        bad_data=[source.replace('id,x1,x2,y','id,x1,x2,label'),source.replace('s2','s1'),source.replace('-1,0.5','1e-400,0.5'),source.replace('-1,0.5','NaN,0.5'),source.replace('0.25,-0.75,1','0.25,-0.75,3'),source.replace('s1,-1,0.5,0','s1,-1,0.5,0,extra'), 'id,x1,x2,y\n',source.replace('s1,-1,0.5,0','s1,-1,0.5,0.0')]
        bad_cfg=[]
        for key,value in [('activation','bad'),('fd_h',0),('fd_h',True),('learning_rate',1),('layers',[]),('extra',1)]:
            c=copy.deepcopy(cfg);c[key]=value;bad_cfg.append(json.dumps(c))
        c=copy.deepcopy(cfg);c['layers'][0]['b']=[[0,0,0]];bad_cfg.append(json.dumps(c))
        bad_cfg += ['{"activation":"tanh","activation":"relu"}',json.dumps(cfg).replace('0.2','1e-400',1),json.dumps(cfg).replace('0.2','NaN',1)]
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d=root/'batch.csv';c=root/'config.json';out=root/'old';new=root/'new';out.mkdir()
            for name in e.OUTPUT_NAMES:(out/name).write_text('sentinel '+name)
            before={p.name:p.read_bytes() for p in out.iterdir()}
            for dat,con in [(x,json.dumps(cfg)) for x in bad_data]+[(source,x) for x in bad_cfg]:
                d.write_text(dat);c.write_text(con)
                for dest in (out,new):
                    r=subprocess.run([sys.executable,'-O',str(e.HERE/'experiment.py'),'--data',str(d),'--config',str(c),'--output',str(dest)],capture_output=True,text=True)
                    self.assertEqual(r.returncode,2,(r.stdout,r.stderr))
                    self.assertEqual({p.name:p.read_bytes() for p in out.iterdir()},before);self.assertFalse(new.exists())
    def test_calculation_failure_isolation(self):
        # Valid input parameters at the bound cannot be centrally perturbed inside the
        # finite demonstration domain; calculation fails before any final writes.
        cfg=json.loads((e.HERE/'data/config.json').read_text());cfg['layers'][0]['W'][0][0]=10
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c=root/'config.json';c.write_text(json.dumps(cfg));out=root/'old';out.mkdir()
            for n in e.OUTPUT_NAMES:(out/n).write_text('old')
            with self.assertRaises(ValueError):e.run(config=c,output=out)
            self.assertTrue(all((out/n).read_text()=='old' for n in e.OUTPUT_NAMES))
    def test_reproducible_and_no_mutation(self):
        x=self.X.copy();ps=copy.deepcopy(self.params)
        e.finite_difference(self.X,self.y,self.params)
        np.testing.assert_array_equal(x,self.X)
        for a,b in zip(ps,self.params):
            for k in a:np.testing.assert_array_equal(a[k],b[k])
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'out'
            subprocess.run([sys.executable,'-O',str(e.HERE/'experiment.py'),'--output',str(out)],cwd=t,capture_output=True,check=True)
            for n in e.OUTPUT_NAMES:self.assertEqual((out/n).read_bytes(),(e.HERE/'outputs'/n).read_bytes())
    def test_wrong_gradients_are_detected(self):
        g=e.flat(e.loss_and_grad(self.X,self.y,self.params)['grads'])
        reference=e.scalar_sample_oracle(self.X,self.y,self.params).mean(0)
        self.assertGreater(np.max(abs(5*g-reference)),.1)
        self.assertGreater(np.max(abs(g/5-reference)),.01)
        # Wrong axis can have a coincident shape when n=C, but is still different.
        _,delta,_,_=e.cross_entropy(np.zeros((3,3)),[0,0,1])
        self.assertEqual(delta.sum(0).shape,delta.sum(1).shape)
        self.assertGreater(np.max(abs(delta.sum(0)-delta.sum(1))),.3)

if __name__=='__main__': unittest.main(verbosity=2)
