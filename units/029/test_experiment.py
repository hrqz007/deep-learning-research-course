"""Run with python -m unittest -v test_experiment; actual torch CPU checks."""
import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
import torch
import experiment as e
import reference as r

torch.set_num_threads(1)

class FrameworkTests(unittest.TestCase):
    def setUp(self):self.X,self.y,self.layers,self.cfg=e.load_inputs()

    def test_01_complete_default_and_027_facts(self):
        actual=e.torch_result(self.X,self.y,self.layers);ref=r.numpy_reference(self.X,self.y,self.layers)
        rows=e.compare(ref,actual);self.assertTrue(all(a['pass'] for a in rows))
        self.assertAlmostEqual(float(actual['loss']),1.148978865605775,14)
        self.assertEqual(sum(p['W'].size+p['b'].size for p in self.layers),26)
        summed=e.torch_result(self.X,self.y,self.layers,reduction='sum')
        for k in actual:
            if k.startswith('d'):np.testing.assert_allclose(summed[k],5*actual[k],atol=2e-15,rtol=2e-13)

    def test_02_independent_random_matrix_references(self):
        rng=np.random.default_rng(2902)
        for act in ('tanh','relu','linear'):
            for case in range(24):
                n=int(rng.integers(1,8));width=[int(rng.integers(1,5)),int(rng.integers(1,5)),int(rng.integers(2,5))]
                X=rng.uniform(-.7,.7,(n,width[0]));y=rng.integers(width[-1],size=n)
                layers=[{'W':rng.uniform(-.6,.6,(a,b)),'b':rng.uniform(-.4,.4,b)} for a,b in zip(width[:-1],width[1:])]
                for red in ('mean','sum'):
                    rows=e.compare(r.numpy_reference(X,y,layers,act,red),e.torch_result(X,y,layers,act,red))
                    self.assertTrue(all(a['pass'] for a in rows),(act,case,red))

    def test_03_scalar_forward_finite_differences(self):
        rows=r.finite_differences(self.X,self.y,self.layers)
        self.assertEqual(len(rows),36);self.assertLess(max(a['absolute_error'] for a in rows),2e-9)
        rng=np.random.default_rng(2903)
        for _ in range(18):
            X=rng.uniform(-.5,.5,(3,2));layers=[{'W':rng.uniform(-.5,.5,(2,2)),'b':rng.uniform(-.2,.2,2)},{'W':rng.uniform(-.5,.5,(2,3)),'b':rng.uniform(-.2,.2,3)}];y=np.array([0,1,2])
            rows=r.finite_differences(X,y,layers);actual=e.torch_result(X,y,layers)
            for a in rows:
                idx=tuple(map(int,a['index'].split(',')))
                self.assertAlmostEqual(float(actual[a['tensor']][idx]),a['finite_difference'],places=8)

    def test_04_fraction_affine_hand_and_random(self):
        X=e.leaf([[1.,2.],[-1.,1.]]);W=e.leaf([[1.,0.],[0.,1.]]);b=e.leaf([.5,-.5]);z=X@W+b;z.retain_grad();loss=.5*(z*z).sum()/2;loss.backward()
        self.assertEqual(float(loss.detach()),1.25)
        for actual,expected in [(z.grad,[[.75,.75],[-.25,.25]]),(W.grad,[[1.,.5],[1.25,1.75]]),(b.grad,[.5,1.]),(X.grad,[[.75,.75],[-.25,.25]])]:np.testing.assert_array_equal(e.array(actual),expected)
        rng=np.random.default_rng(2904)
        for _ in range(100):
            xx=rng.integers(-3,4,(2,2));ww=rng.integers(-3,4,(2,2));bb=rng.integers(-3,4,2)
            zz=[[sum(Fraction(int(xx[i,k]))*int(ww[k,j]) for k in range(2))+int(bb[j]) for j in range(2)] for i in range(2)]
            dw=[[sum(Fraction(int(xx[i,k]))*zz[i][j]/2 for i in range(2)) for j in range(2)] for k in range(2)]
            x=e.leaf(xx);w=e.leaf(ww);b=e.leaf(bb);z=x@w+b;(.25*(z*z).sum()).backward()
            np.testing.assert_array_equal(e.array(w.grad),np.array(dw,dtype=float))
            np.testing.assert_array_equal(e.array(b.grad),[float(sum(zz[i][j]/2 for i in range(2))) for j in range(2)])
            np.testing.assert_array_equal(e.array(x.grad),[[float(sum(zz[i][j]*int(ww[k,j])/2 for j in range(2))) for k in range(2)] for i in range(2)])

    def test_05_storage_leaf_and_graph(self):
        p=e.semantic_probes(self.X,self.y,self.layers)
        self.assertEqual(p['storage'],{'clone_shares':False,'clone_has_history':True,'detach_shares':True,'detach_requires_grad':False,'detached_clone_shares':False})
        self.assertTrue(p['leaf']['input']);self.assertFalse(p['leaf']['float_conversion']);self.assertTrue(p['leaf']['same_dtype_returns_same_object'])
        self.assertEqual(p['numpy_alias']['gradient_after_foreign_mutation'],6.)
        self.assertEqual(p['numpy_alias']['copied_tensor_gradient'],4.)
        x=e.leaf(2.);h=x*x
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning);self.assertIsNone(h.grad)
        h.retain_grad();(3*h).backward();self.assertEqual(float(h.grad),3.);self.assertEqual(float(x.grad),12.)

    def test_06_intentional_detach_diagnosis(self):
        p=e.semantic_probes(self.X,self.y,self.layers)
        for cut in ('detach','reconstruct'):
            self.assertEqual(p['cuts'][cut]['loss_difference'],0.);self.assertEqual(p['cuts'][cut]['last_weight_gradient_difference'],0.)
            self.assertIn('dW1',p['cuts'][cut]['missing']);self.assertIn('dH0',p['cuts'][cut]['missing'])
        w=e.leaf(2.);g=torch.autograd.grad((w*w).detach()*w,w)[0];self.assertEqual(float(g),4.)
        w=e.leaf(2.);g=torch.autograd.grad((w*w).clone()*w,w)[0];self.assertEqual(float(g),12.)

    def test_07_inplace_and_lifetime(self):
        p=e.semantic_probes(self.X,self.y,self.layers)
        for key in ('second_backward_error','leaf_inplace_error','saved_inplace_error','detached_alias_error','no_grad_before_backward_error','reconstructed_scalar_error'):self.assertTrue(p[key])
        w=e.leaf(.5);h=torch.tanh(w);safe=h+1;safe.backward();self.assertAlmostEqual(float(w.grad),1-np.tanh(.5)**2,15)
        w=e.leaf(2.);l=w*w;l.backward(retain_graph=True);l.backward();self.assertEqual(float(w.grad),8.)

    def test_08_accumulation_microbatch_and_update(self):
        p=e.semantic_probes(self.X,self.y,self.layers)
        self.assertEqual(p['accumulation'],{'first':4.,'uncleared_second':8.,'after_clear':4.})
        self.assertLess(p['microbatch']['weighted'],1e-15);self.assertGreater(p['microbatch']['unweighted'],.03)
        self.assertAlmostEqual(p['update']['after'],1.1413798716255645,14);self.assertTrue(p['update']['parameters_still_leaf'])
        # All 2+3 partitions, not only the contiguous cut.
        from itertools import combinations
        full=e.torch_result(self.X,self.y,self.layers)
        for ii in combinations(range(5),2):
            jj=tuple(k for k in range(5) if k not in ii)
            parts=[e.torch_result(self.X[list(k)],self.y[list(k)],self.layers) for k in (ii,jj)]
            for name in full:
                if name.startswith(('dW','db')):np.testing.assert_allclose(.4*parts[0][name]+.6*parts[1][name],full[name],atol=1e-15,rtol=1e-13)

    def test_09_grad_modes(self):
        p=e.semantic_probes(self.X,self.y,self.layers)
        self.assertEqual(p['no_grad'],{'input_still_requires_grad':True,'result_requires_grad':False,'factory_requires_grad':True})
        self.assertEqual(p['eval']['training_output'],[0.,0.]);self.assertEqual(p['eval']['training_input_grad'],[0.,0.]);self.assertFalse(p['eval']['training_no_grad_requires_grad']);self.assertEqual(p['eval']['eval_input_grad'],[1.,1.]);self.assertTrue(p['eval']['eval_requires_grad']);self.assertFalse(p['eval']['eval_no_grad_requires_grad'])

    def test_10_vjp_and_higher_order(self):
        p=e.semantic_probes(self.X,self.y,self.layers);self.assertEqual(p['vjp']['gradient'],[6.,-4.]);self.assertEqual(p['higher_order']['first'],1.5);self.assertEqual(p['higher_order']['second'],5.5)
        for a in np.linspace(-1.3,1.2,21):
            w=e.leaf(float(a));f=.5*(w*w+w)**2;g=torch.autograd.grad(f,w,create_graph=True)[0];gg=torch.autograd.grad(g,w)[0]
            self.assertAlmostEqual(float(g.detach()),(a*a+a)*(2*a+1),12);self.assertAlmostEqual(float(gg),6*a*a+6*a+1,12)
        w=e.leaf([1.,2.]);self.assertTrue(e.expect_runtime(lambda:(w*w).backward()))
        kink=e.leaf(0.);torch.relu(kink).backward();self.assertEqual(float(kink.grad),0.)
        self.assertEqual((max(1e-5,0.)-max(-1e-5,0.))/(2e-5),.5)

    def test_11_dtype_and_rejects(self):
        p=e.semantic_probes(self.X,self.y,self.layers);self.assertTrue(p['integer_requires_grad_error']);self.assertTrue(p['mixed_matmul_error'])
        bad=[lambda:r.validate([[True,1]],[0],self.layers),lambda:r.validate([[float('nan'),0]],[0],self.layers),lambda:r.validate([],[],self.layers),lambda:r.validate(self.X,[0.,1.,2.,1.,0.],self.layers),lambda:r.validate(self.X,[False,1,2,1,0],self.layers),lambda:r.validate(self.X,self.y,[{'W':[[1]],'b':[0]}]),lambda:r.validate(self.X,self.y,self.layers,'bad'),lambda:r.validate(self.X,self.y,self.layers,reduction='none'),lambda:r.real(['1'],'string'),lambda:r.finite_differences(self.X,self.y,self.layers,h=0),lambda:e.torch_result(self.X,self.y,self.layers,dtype=torch.int64)]
        if np.finfo(np.longdouble).tiny < np.finfo(np.float64).tiny:
            bad.append(lambda:r.real(np.array([np.longdouble('1e-400')]),'tiny'))
        large=[{'W':np.full((16,16),10.),'b':np.zeros(16)} for _ in range(5)]
        bad += [lambda:e.torch_result(np.full((1,16),10.),[0],large,activation='linear'),lambda:r.scalar_loss(np.full((1,16),10.),[0],large,activation='linear')]
        for action in bad:
            with self.assertRaises((ValueError,TypeError)):action()
        np.testing.assert_array_equal(r.real([0.],'zero'),[0.])

    def test_12_fixed_input_and_output_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';data.mkdir()
            for name in e.INPUT_SHA:(data/name).write_bytes((e.HERE/'data'/name).read_bytes())
            for filename in e.INPUT_SHA:
                original=(data/filename).read_bytes()
                for changed in (original+b'\n',original.replace(b'0.2',b'1e-400',1) if filename=='config.json' else original.replace(b'-1,',b'NaN,',1)):
                    (data/filename).write_bytes(changed)
                    for old in (False,True):
                        out=root/('old' if old else 'new')
                        if old:out.mkdir(exist_ok=True);(out/'summary.json').write_text('sentinel')
                        with self.assertRaises(ValueError):e.run(data,out)
                        if old:self.assertEqual((out/'summary.json').read_text(),'sentinel')
                        else:self.assertFalse(out.exists())
                    (data/filename).write_bytes(original)

if __name__=='__main__':unittest.main(verbosity=2)
