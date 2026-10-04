"""Independent indexed rational adjoints, high-precision activation and VJP probes."""
from fractions import Fraction as F
from decimal import Decimal as D,localcontext
from pathlib import Path
from unittest.mock import patch
from dataclasses import FrozenInstanceError
import json,math,random,tempfile,unittest
import numpy as np
import engine as a
import experiment as e

def source_index(index,shape):
    tail=index[len(index)-len(shape):] if shape else ()
    return tuple(0 if s==1 else v for v,s in zip(tail,shape))

class Test028(unittest.TestCase):
    def test_hand_broadcast(self):
        u=a.Tensor([[1.],[2.]]);v=a.Tensor([[.5,-1.,1.5]]);seed=np.array([[1.,2.,3.],[4.,5.,6.]])
        root=u+v;g=a.vjp(root,seed)
        np.testing.assert_array_equal(g[u],[[6.],[15.]])
        np.testing.assert_array_equal(g[v],[[5.,7.,9.]])
        self.assertEqual(g[u].shape,(2,1));self.assertEqual(g[v].shape,(1,3))
        s=a.Tensor(2.);r=u*s;h=a.vjp(r,[[3.],[4.]])
        self.assertEqual(float(h[s]),11.);np.testing.assert_array_equal(h[u],[[6.],[8.]])

    def test_small_jacobian_seed_and_dot_identity(self):
        x,y=a.Tensor(2.),a.Tensor(-1.)
        root=x*y*np.array([1.,0.])+(x+y*y)*np.array([0.,1.])
        g=a.vjp(root,[3.,-1.]);np.testing.assert_array_equal(root.value,[-2.,3.])
        self.assertEqual(float(g[x]),-4);self.assertEqual(float(g[y]),8)
        self.assertEqual(float(a.vjp(root,[1.,1.])[x]),0)
        self.assertEqual(float(a.vjp(root,[1.,1.])[y]),0)
        J=np.array([[-1.,2.],[1.,-2.]])
        np.testing.assert_array_equal(J@np.array([.5,2.]),[3.5,-3.5])
        self.assertEqual(float(np.array([3.,-1.])@J@np.array([.5,2.])),14.)

    def test_360_fraction_broadcast_graphs(self):
        patterns=[((),()),((),(2,3)),((3,),(2,3)),((2,1),(1,3)),((1,3),(2,1)),((2,1,3),(1,4,1)),((1,1),(2,3)),((2,1,1,3),(1,4,2,1)),((1,),(2,3,1))]
        rng=random.Random(2801)
        for shapeA,shapeB in patterns:
            shape=np.broadcast_shapes(shapeA,shapeB)
            for _ in range(40):
                A=np.array([F(rng.randrange(-4,5),8) for _ in range(math.prod(shapeA))],dtype=object).reshape(shapeA)
                B=np.array([F(rng.randrange(-4,5),8) for _ in range(math.prod(shapeB))],dtype=object).reshape(shapeB)
                seed=np.array([F(rng.randrange(-4,5),8) for _ in range(math.prod(shape))],dtype=object).reshape(shape)
                xa=a.Tensor(np.asarray(A,float));xb=a.Tensor(np.asarray(B,float));t=xa*xb+xa;out=t*t+t;g=a.vjp(out,np.asarray(seed,float))
                da=np.full(shapeA,F(0),dtype=object);db=np.full(shapeB,F(0),dtype=object);forward=np.empty(shape,dtype=object)
                for idx in np.ndindex(shape):
                    ia=source_index(idx,shapeA);ib=source_index(idx,shapeB);va=A[ia];vb=B[ib];z=va*vb+va
                    forward[idx]=z*z+z
                    da[ia]+=seed[idx]*(2*z+1)*(vb+1);db[ib]+=seed[idx]*(2*z+1)*va
                np.testing.assert_array_equal(out.value,np.asarray(forward,float))
                np.testing.assert_array_equal(g[xa],np.asarray(da,float));np.testing.assert_array_equal(g[xb],np.asarray(db,float))

    def test_reductions_by_independent_index_mapping(self):
        X=np.arange(24.).reshape(2,3,4)/8;node=a.Tensor(X)
        for axes in [None,(),0,1,-1,(0,2),(1,2)]:
            normalized=tuple(range(3)) if axes is None else ((axes%3,) if type(axes) is int else tuple(sorted(axes)))
            for keep in [False,True]:
                out=node.sum(axis=axes,keepdims=keep);seed=np.arange(out.value.size).reshape(out.shape)/8+1
                gradient=a.vjp(out,seed)[node];expected=np.empty(X.shape)
                for idx in np.ndindex(X.shape):
                    oi=tuple(0 if k in normalized else v for k,v in enumerate(idx)) if keep else tuple(v for k,v in enumerate(idx) if k not in normalized)
                    expected[idx]=seed[oi]
                np.testing.assert_array_equal(gradient,expected)
        s=a.Tensor(3.);self.assertEqual(float(a.vjp(s.sum())[s]),1)

    def test_40_dense_jacobians_and_duality(self):
        rng=np.random.default_rng(2802)
        for _ in range(40):
            X=rng.uniform(-1,1,(2,3));w=rng.uniform(-.8,.8,3);b=np.array(rng.uniform(-.3,.3));seed=rng.uniform(-1,1,(2,3));n=e.build(X,w,b);g=a.vjp(n['z'],seed)
            values=[X,w,b];J=np.empty((6,10));col=0;h=1e-5
            for k,arr in enumerate(values):
                for idx in np.ndindex(arr.shape):
                    p=[x.copy() for x in values];m=[x.copy() for x in values];p[k][idx]+=h;m[k][idx]-=h
                    J[:,col]=((e.plain(*p)-e.plain(*m))/(2*h)).ravel();col+=1
            flat=np.concatenate([g[n[k]].reshape(-1) for k in ['X','w','b']]);np.testing.assert_allclose(flat,J.T@seed.ravel(),atol=2e-9,rtol=2e-8)
            direction=rng.uniform(-1,1,10)
            self.assertAlmostEqual(float(seed.ravel()@(J@direction)),float(flat@direction),places=8)

    def test_121_decimal_tanh_and_mean_reference(self):
        with localcontext() as context:
            context.prec=100
            for i in range(-60,61):
                x=a.Tensor(i/2);out=x.tanh();q=(-D(abs(i))).exp();slope=4*q/(1+q)**2
                self.assertTrue(math.isclose(float(a.vjp(out)[x]),float(slope),rel_tol=5e-16,abs_tol=0))
        c=e.read_config(e.HERE/'data/config.json');n=e.build(c['X'],c['w'],c['b']);g=a.vjp(n['L']);refs=e.scalar_reference(c['X'],c['w'],c['b'],np.full((2,3),1/6))
        for k,r in zip(['X','w','b'],refs):np.testing.assert_allclose(g[n[k]],r,atol=3e-16,rtol=0)

    def test_49_exact_first_and_rebuilt_second_derivatives(self):
        for i in range(-24,25):
            q=F(i,8);x=a.Tensor(float(q));f=.5*(x*x+x)*(x*x+x);df=(x*x+x)*(2*x+1)
            self.assertEqual(float(a.vjp(f)[x]),float(2*q**3+3*q*q+q))
            self.assertEqual(float(a.vjp(df)[x]),float(6*q*q+6*q+1))
        with self.assertRaises(ValueError):a.vjp(a.vjp(f)[x])

    def test_nonsmooth_composition_and_barrier(self):
        for value in [-1.,0.,1.]:
            x=a.Tensor(value);r=x.relu();composed=x.relu()-(-x).relu()
            self.assertEqual(float(composed.value),value)
            self.assertEqual(float(a.vjp(composed)[x]),1. if value else 0.)
            self.assertEqual(float(a.vjp(r)[x]),float(value>0))
        x=a.Tensor(2.);normal=x*x*x;blocked=(x*x).detach()*x
        self.assertEqual(float(normal.value),float(blocked.value));self.assertEqual(float(a.vjp(normal)[x]),12);self.assertEqual(float(a.vjp(blocked)[x]),4)
        detached=x.detach();self.assertNotIn(x,a.vjp(detached));self.assertIsNot(x,detached)

    def test_seed_identity_and_repeated_calls(self):
        x=a.Tensor([1.,2.]);y=x*x+x
        with self.assertRaises(ValueError):a.vjp(y)
        with self.assertRaises(ValueError):a.vjp(a.Tensor([1.]))
        for seed in [1.,[[1.,1.]],True,[True,False]]:
            with self.assertRaises(ValueError):a.vjp(y,seed)
        g=a.vjp(y,[1.,2.]);h=a.vjp(y,[1.,2.]);np.testing.assert_array_equal(g[x],[3.,10.]);self.assertIsNot(g[x],h[x])
        g[x][0]=999;self.assertEqual(h[x][0],3.)
        other=a.Tensor([1.,2.]);self.assertNotIn(other,g)
        np.testing.assert_array_equal(a.vjp(y,[0.,0.])[x],[0.,0.])
        for result,expected in [(np.array([2.,3.])+x,[3.,5.]),(np.array([2.,3.])*x,[2.,6.]),(np.array([2.,3.])-x,[1.,1.])]:
            self.assertIsInstance(result,a.Tensor);np.testing.assert_array_equal(result.value,expected)
        np.testing.assert_array_equal(a.vjp(np.array([2.,3.])*x,[1.,1.])[x],[2.,3.])

    def test_snapshots_and_defensive_value_copy(self):
        original=np.array([2.,3.]);x=a.Tensor(original);y=x*x;original[:]=0
        np.testing.assert_array_equal(x.value,[2.,3.]);np.testing.assert_array_equal(a.vjp(y,[1.,1.])[x],[4.,6.])
        shown=x.value
        with self.assertRaises(ValueError):shown[0]=9
        shown.setflags(write=True);shown[0]=9
        np.testing.assert_array_equal(x.value,[2.,3.])
        with self.assertRaises(FrozenInstanceError):x._value=np.zeros(2)

    def test_invalid_numeric_shapes_axes_and_underflow(self):
        bad=[True,[False],[],[[]],['1'],[1+0j],[np.nan],[np.inf],np.zeros((1,1,1,1,1)),np.zeros(100001),[1e101],np.array([D('1')],dtype=object)]
        for x in bad:
            with self.assertRaises(ValueError):a.Tensor(x)
        for args in [(True,2),(0,1e101),('1',1),(np.ones((400,1)),np.ones((1,400)))]:
            with self.assertRaises(ValueError):a.multiply(*args)
        for x in [np.longdouble('1e-400'),np.longdouble('-1e-400')]:
            if x!=0:
                with self.assertRaises(ValueError):a.Tensor(np.array([x],dtype=np.longdouble))
        self.assertEqual(float(a.Tensor(np.longdouble(0)).value),0)
        self.assertEqual(float(a.Tensor(np.nextafter(0.,1.)).value),np.nextafter(0.,1.))
        with self.assertRaises(ValueError):a.Tensor(1e-300)*1e-300
        with self.assertRaises(ValueError):a.Tensor(1e100)*2
        with self.assertRaises(ValueError):a.Tensor(101).tanh()
        with self.assertRaises(ValueError):a.Tensor(np.ones(2))+np.ones(3)
        t=a.Tensor(np.ones((2,3)))
        for axis in [True,[0],(0,0),(0,-2),2,-3,'0']:
            with self.assertRaises(ValueError):t.sum(axis=axis)
        with self.assertRaises(ValueError):t.sum(keepdims=1)
        for shape in [(3,2),(2,3,1),(0,),[2,3],(True,3)]:
            with self.assertRaises(ValueError):a.sum_to_shape(np.ones((2,3)),shape)

    def test_invalid_configs_and_late_failures_preserve_files(self):
        text=(e.HERE/'data/config.json').read_text();base=json.loads(text)
        bad=['[]','{',text.replace('"schema": 1','"schema": 1,"schema":1'),text.replace('"b": 0.125','"b": 1e-400'),text.replace('"b": 0.125','"b": -1e-400'),text.replace('"b": 0.125','"b": NaN')]
        for key,value in [('schema',True),('b',4),('b',None),('b',[0]),('w',[1,2]),('X',[[1,2,3]]),('seed',0),('seed',[[True,0,0],[0,0,0]])]:bad.append(json.dumps(base|{key:value}))
        bad.append(json.dumps(base|{'extra':1}))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';e.run(output=old);before={p.name:p.read_bytes() for p in old.iterdir()}
            for i,t in enumerate(bad):
                p=root/'bad.json';p.write_text(t)
                for out in [old,root/f'new{i}']:
                    with self.assertRaises(ValueError):e.run(p,out)
                self.assertFalse((root/f'new{i}').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
            under=json.loads(text);under['X'][0][0]=1e-300;under['w'][0]=1e-300
            cp=root/'under.json';cp.write_text(json.dumps(under))
            for out in [old,root/'under_new']:
                with self.assertRaises(ValueError):e.run(cp,out)
            self.assertFalse((root/'under_new').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
            for out in [old,root/'new_late']:
                with patch.object(e,'csv_string',side_effect=ValueError('forced late serialization failure')):
                    with self.assertRaises(ValueError):e.run(output=out)
            self.assertFalse((root/'new_late').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
            (root/'link').symlink_to(old,target_is_directory=True)
            with self.assertRaises(ValueError):e.run(output=root/'link')

    def test_iterative_chain_and_one_backward_per_node(self):
        x=a.Tensor(1.);t=x
        for _ in range(1100):t=t+0.
        self.assertEqual(float(a.vjp(t)[x]),1);self.assertEqual(len(a.topological(t)),2201)
        x=a.Tensor(2.);q=x*x;root=q+q+x
        self.assertEqual(float(a.vjp(root)[x]),9)
        self.assertEqual(len(a.topological(root)),4)

if __name__=='__main__':unittest.main(verbosity=2)
