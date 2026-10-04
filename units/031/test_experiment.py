"""Run: python -m unittest -v test_experiment.py. No assert statements required."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from decimal import Decimal, localcontext
from fractions import Fraction
import numpy as np
import torch
import experiment as E

torch.set_num_threads(1)


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X,cls.y,cls.yr,cls.cfg=E.load_inputs()

    def test_01_fraction_hand_and_quadratic_grid(self):
        # Independent rational arithmetic for 125 two-sample linear models.
        for w in [Fraction(k,4) for k in range(-2,3)]:
            for b in [Fraction(k,4) for k in range(-2,3)]:
                for a in [Fraction(k,2) for k in range(-2,3)]:
                    x=[-1,1];y=[-a,a];r=[w*xi+b-yi for xi,yi in zip(x,y)]
                    loss=sum(v*v for v in r)/4;gw=sum(v*xi for v,xi in zip(r,x))/2;gb=sum(r)/2
                    t=torch.tensor([float(w),float(b)],dtype=torch.float64,requires_grad=True)
                    tx=torch.tensor(x,dtype=torch.float64);ty=torch.tensor([float(v) for v in y],dtype=torch.float64)
                    actual=.5*((tx*t[0]+t[1]-ty)**2).mean();g=torch.autograd.grad(actual,t)[0]
                    self.assertEqual(float(actual.detach()),float(loss));self.assertEqual(g.tolist(),[float(gw),float(gb)])
        w,b=Fraction(1,2),Fraction(1,4)
        self.assertEqual(((w-1)**2+b*b)/2,Fraction(5,32))
        self.assertEqual(((w+Fraction(1,20)-1)**2+(b-Fraction(1,40))**2)/2,Fraction(81,640))

    def test_02_decimal_forward_and_derivative(self):
        # 36 independently randomized small networks, 468 70-digit central differences.
        rng=random.Random(3102)
        with localcontext() as ctx:
            ctx.prec=70
            for case in range(36):
                X=[[Decimal(rng.randint(-10,10))/10 for _ in range(2)] for _ in range(3)]
                y=[Decimal(rng.randint(-10,10))/10 for _ in range(3)]
                t=[Decimal(rng.randint(-8,8))/10 for _ in range(13)]
                def dec_loss(q):
                    total=Decimal(0)
                    for row,target in zip(X,y):
                        hs=[]
                        for j in range(3):
                            z=row[0]*q[j]+row[1]*q[3+j]+q[6+j]
                            ez=(2*z).exp();hs.append((ez-1)/(ez+1))
                        residual=sum(hs[j]*q[9+j] for j in range(3))+q[12]-target
                        total+=residual*residual/6
                    return total
                xf=np.array(X,dtype=float);yf=np.array(y,dtype=float).reshape(-1,1);tf=np.array(t,dtype=float)
                loss,g=E.autograd_values(xf,yf,tf);reference,gr=E.scalar_reference(xf,yf,tf)
                self.assertAlmostEqual(loss,float(dec_loss(t)),places=14)
                self.assertAlmostEqual(reference,loss,places=14)
                eps=Decimal('1e-24')
                for k in range(13):
                    tp=t.copy();tm=t.copy();tp[k]+=eps;tm[k]-=eps
                    expected=float((dec_loss(tp)-dec_loss(tm))/(2*eps))
                    self.assertAlmostEqual(g[k],expected,places=13)
                    self.assertAlmostEqual(gr[k],expected,places=13)

    def test_03_fd_and_framework_gradcheck(self):
        X,y=self.X[:4],self.y[:4];t=self.cfg['theta']
        _,g=E.autograd_values(X,y,t);numeric=E.central_difference(X,y,t)
        np.testing.assert_allclose(g,numeric,atol=1e-8,rtol=1e-5)
        rows=E.fd_report(X,y,t,self.cfg['fd_steps'])
        self.assertEqual(len(rows),78)
        self.assertTrue(all(r['pass'] for r in rows if r['step']==1e-6))
        f=E.fault_report(X,y,t)
        self.assertEqual(f['gradcheck'],{'healthy':True,'detached_hidden':False,'permuted_labels':True})
        self.assertEqual(f['gradient_compare']['cut_failed_coordinates'],list(range(9)))

    def test_04_one_step_all_fault_routes(self):
        probes={k:E.one_step_probe(self.X[:4],self.y[:4],k) for k in ['healthy','no_step','zero_lr','omit_W1','detach_hidden','sum_reduction']}
        healthy=probes['healthy']
        for row in healthy['parameters']:
            self.assertGreater(row['grad_norm'],0);self.assertAlmostEqual(row['update_norm'],.03*row['grad_norm'],places=14)
        for mode in ['no_step','zero_lr']:
            self.assertEqual(probes[mode]['loss'],healthy['loss'])
            self.assertTrue(all(r['update_norm']==0 and r['grad_norm']>0 for r in probes[mode]['parameters']))
        omitted=probes['omit_W1']['parameters'][0]
        self.assertFalse(omitted['in_optimizer']);self.assertFalse(omitted['grad_is_none']);self.assertEqual(omitted['update_norm'],0)
        detached=probes['detach_hidden']['parameters']
        self.assertEqual([r['grad_is_none'] for r in detached],[True,True,False,False])
        for good,scaled in zip(healthy['parameters'],probes['sum_reduction']['parameters']):
            self.assertAlmostEqual(scaled['grad_norm'],4*good['grad_norm'],places=13)
            self.assertAlmostEqual(scaled['update_norm'],4*good['update_norm'],places=13)

    def test_05_zero_none_kink_broadcast(self):
        x=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        self.assertEqual(float(torch.autograd.grad(torch.relu(x),x)[0]),0)
        for h in [1e-2,1e-6,1e-10]:
            self.assertEqual((max(h,0)-max(-h,0))/(2*h),.5)
        X,y=self.X[:4],self.y[:4];f=E.fault_report(X,y,self.cfg['theta'])
        self.assertEqual(f['none_vs_zero'],{'connected_relu_gradient':0.,'unused_gradient_is_none':True})
        self.assertEqual(f['broadcast']['incorrect_shape'],[2,2]);self.assertEqual(f['broadcast']['incorrect_loss'],.25)
        self.assertEqual(f['direction_blind_spot']['true_dot'],f['direction_blind_spot']['wrong_dot'])
        self.assertGreater(f['direction_blind_spot']['coordinate_error_norm'],1)

    def test_06_actual_fits_and_perturbations(self):
        model,trace=E.train_one_batch(self.X,self.y)
        random_model,rtrace=E.train_one_batch(self.X,self.yr)
        self.assertEqual(len(trace),1601);self.assertEqual(trace[-1]['update_norm'],0)
        self.assertLess(trace[-1]['loss_before_step'],1e-4);self.assertLess(rtrace[-1]['loss_before_step'],1e-4)
        p=E.perturbation_report(model,self.X,self.y)
        self.assertAlmostEqual(p['joint_row_permutation_loss'],p['baseline_loss'],places=30)
        self.assertGreater(p['input_only_permutation_loss'],.9)
        self.assertGreater(abs(p['small_shifts'][-1]['mean_slope']-.7),.1)
        self.assertAlmostEqual(p['mean_target_constant_loss'],.24375,places=14)
        with torch.no_grad():
            self.assertLess(float(.5*(random_model(torch.tensor(self.X))-torch.tensor(self.yr)).square().mean()),1e-4)

    def test_07_contradictory_duplicate_floor(self):
        _,trace=E.train_one_batch([[0.,0.],[0.,0.]],[[-1.],[1.]],300,.03)
        self.assertAlmostEqual(trace[-1]['loss_before_step'],.5,places=12)
        for k in range(-100,101):
            c=Fraction(k,10);loss=((c+1)**2+(c-1)**2)/4
            self.assertEqual(loss,(c*c+1)/2);self.assertGreaterEqual(loss,Fraction(1,2))

    def test_08_local_generator_reproducibility(self):
        torch.manual_seed(812);before=torch.random.get_rng_state().clone()
        a=E.Tiny();after=torch.random.get_rng_state();self.assertTrue(torch.equal(before,after))
        b=E.Tiny();self.assertEqual(sum(p.numel() for p in a.parameters()),49)
        for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
        _,a=E.train_one_batch(self.X,self.y,5);_,b=E.train_one_batch(self.X,self.y,5);self.assertEqual(a,b)

    def test_09_data_provenance(self):
        np.testing.assert_allclose(self.y[:,0],.7*self.X[:,0]-.4*self.X[:,1]+.2,atol=3e-16,rtol=0)
        rng=random.Random(31);self.assertEqual(self.yr[:,0].tolist(),[round(rng.uniform(-1,1),3) for _ in range(8)])

    def test_10_reject_helper_inputs(self):
        bad_X=[[],[1,2],[[1,2,3]],[[True,1.]],[[float('nan'),0]],[[float('inf'),0]],[[1001.,0]],[[1+0j,0]],np.array([[-2**63,0]],dtype=np.int64)]
        for x in bad_X:
            with self.subTest(x=str(x)),self.assertRaises((ValueError,TypeError)):E.validate_batch(x,[[0.]])
        for y in [[1,2],[[1.,True]],[[1.]],np.ones((8,2)),[[None]]]:
            with self.assertRaises((ValueError,TypeError)):E.validate_batch(self.X,y)
        for eps in [True,0,-1,float('nan'),float('inf'),1e-14,1.,'1e-6']:
            with self.assertRaises(ValueError):E.central_difference(self.X,self.y,self.cfg['theta'],eps)
        for steps in [True,0,5001,1.5]:
            with self.assertRaises(ValueError):E.train_one_batch(self.X,self.y,steps)
        for lr in [True,0,-.1,.2,float('nan')]:
            with self.assertRaises(ValueError):E.train_one_batch(self.X,self.y,1,lr)
        for seed in [True,-1,2**31,1.2]:
            with self.assertRaises(ValueError):E.Tiny(seed)
        with self.assertRaises(ValueError):E.one_step_probe(self.X,self.y,'unknown')
        with self.assertRaises(ValueError):E.autograd_values(self.X,self.y,self.cfg['theta'],1)
        with self.assertRaises(ValueError):E.scalar_reference(self.X,self.y,[1.]*12)

    def test_11_fixed_cli_failures_preserve_output(self):
        mutations=[('config.json',b'{}'),('config.json',b'{"steps":1e-400}'),('config.json',b'{"steps":1600.0000000000000000001}'),('config.json',b'{"steps":true}'),('config.json',b'NaN'),('batch.csv',b'id,x1,x2,target\na,NaN,0,1\n'),('batch.csv',b''),('batch.csv',b'id,x1,x2,target\na,0,0,1\n')]
        for name,data in mutations:
            for exists in (False,True):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);inp=root/'data';shutil.copytree(E.HERE/'data',inp);(inp/name).write_bytes(data)
                    dest=root/'out'
                    if exists:dest.mkdir();(dest/'sentinel').write_bytes(b'preserve');(dest/'summary.json').write_bytes(b'old')
                    before={p.name:p.read_bytes() for p in dest.iterdir()} if exists else None
                    command=[sys.executable,str(E.HERE/'experiment.py'),'--data-dir',str(inp),'--output',str(dest)]
                    done=subprocess.run(command,capture_output=True,cwd=root)
                    self.assertNotEqual(done.returncode,0)
                    if exists:self.assertEqual(before,{p.name:p.read_bytes() for p in dest.iterdir()})
                    else:self.assertFalse(dest.exists())

    def test_12_missing_input_preserves_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dest=root/'not-created'
            with self.assertRaises(OSError):E.run(root/'missing',dest)
            self.assertFalse(dest.exists())

    def test_13_committed_report_consistency(self):
        summary=json.loads((E.HERE/'outputs/summary.json').read_text())
        with (E.HERE/'outputs/training.csv').open() as file:rows=list(csv.DictReader(file))
        self.assertEqual(len(rows),3202)
        self.assertEqual(float(rows[1600]['loss_before_step']),summary['rule_final_loss'])
        self.assertEqual(float(rows[-1]['loss_before_step']),summary['random_final_loss'])
        for name in E.OUTPUT_NAMES:self.assertGreater((E.HERE/'outputs'/name).stat().st_size,0)

    def test_14_full_trace_same_fd_fixture(self):
        r=E.full_trace(self.X[:4],self.y[:4],self.cfg['theta'])
        t={name:np.array(value) for name,value in r['tensors'].items()}
        self.assertEqual(len(r['comparisons']),77)
        self.assertLess(r['comparison_max_abs'],1e-14)
        np.testing.assert_allclose(t['parameter_contributions'].sum(axis=0),t['gradient'],atol=1e-16,rtol=0)
        np.testing.assert_allclose(E.central_difference(self.X[:4],self.y[:4],self.cfg['theta']),t['gradient'],atol=1e-8,rtol=1e-5)
        np.testing.assert_allclose(t['theta_after'],t['theta_before']-.1*t['gradient'],atol=0,rtol=0)
        expected,_=E.scalar_reference(self.X[:4],self.y[:4],t['theta_after'])
        self.assertAlmostEqual(r['loss_after'],expected,places=15)
        self.assertLess(r['loss_after'],r['loss_before'])
        self.assertEqual(r['shapes']['dX'],[4,2])
        self.assertEqual(r['shapes']['parameter_contributions'],[4,13])
        for eta in [True,0,.2,float('nan')]:
            with self.assertRaises(ValueError):E.full_trace(self.X[:4],self.y[:4],self.cfg['theta'],eta)

    def test_15_fixed_figure_and_notebook_guards(self):
        first=next(cell['source'] for cell in json.loads((E.HERE/'experiment.ipynb').read_text())['cells'] if cell['cell_type']=='code')
        if isinstance(first,list):first=''.join(first)
        guard=json.loads((E.HERE/'data/figure-input-sha256.json').read_text())
        self.assertEqual(len(guard),9)
        for changed in guard:
            for target in ('figure','notebook'):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp)
                    shutil.copytree(E.HERE/'data',root/'data');shutil.copytree(E.HERE/'outputs',root/'outputs')
                    shutil.copy2(E.HERE/'make_figures.py',root/'make_figures.py')
                    (root/'figures').mkdir();(root/'figures/sentinel.png').write_bytes(b'preserve')
                    (root/changed).write_bytes((root/changed).read_bytes()+b' ')
                    before={str(p.relative_to(root)):p.read_bytes() for name in ('figures','outputs') for p in (root/name).iterdir()}
                    command=[sys.executable,str(root/'make_figures.py')] if target=='figure' else [sys.executable,'-c',first]
                    result=subprocess.run(command,cwd=root,capture_output=True)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn(b'fixed figure input changed',result.stderr)
                    after={str(p.relative_to(root)):p.read_bytes() for name in ('figures','outputs') for p in (root/name).iterdir()}
                    self.assertEqual(before,after)
                    self.assertFalse((root/'notebook_outputs').exists())

    def test_16_calculation_and_serialization_failure_preserve_output(self):
        for target in ('train_one_batch','encode'):
            for exists in (False,True):
                with tempfile.TemporaryDirectory() as temp:
                    dest=Path(temp)/'out'
                    if exists:
                        dest.mkdir();(dest/'summary.json').write_bytes(b'old');(dest/'sentinel').write_bytes(b'preserve')
                    before={p.name:p.read_bytes() for p in dest.iterdir()} if exists else None
                    with patch.object(E,target,side_effect=ValueError('injected failure before output')):
                        with self.assertRaises(ValueError):E.run(output=dest)
                    if exists:self.assertEqual(before,{p.name:p.read_bytes() for p in dest.iterdir()})
                    else:self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
