"""Independent rational/Decimal oracles plus failure-before-write regressions."""
from fractions import Fraction as F
from decimal import Decimal as D, localcontext
import contextlib, csv, importlib, io, json, math, random, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import experiment as e

class Test025(unittest.TestCase):
    def test_fraction_forward_180_networks(self):
        rng=random.Random(2501)
        for case in range(180):
            widths=[rng.randrange(1,5) for _ in range(2+case%4)]
            X=[[F(rng.randrange(-8,9),4) for _ in range(widths[0])] for _ in range(3)]
            layers=[]
            for a,b in zip(widths,widths[1:]):
                W=[[F(rng.randrange(-6,7),4) for _ in range(b)] for _ in range(a)]
                bias=[F(rng.randrange(-4,5),4) for _ in range(b)];layers.append((W,bias))
            floats=[(np.array(W,dtype=float),np.array(b,dtype=float)) for W,b in layers]
            for activation in ('relu','identity'):
                oracle=[row[:] for row in X]
                for k,(W,b) in enumerate(layers):
                    oracle=[[sum(row[j]*W[j][q] for j in range(len(W)))+b[q] for q in range(len(b))] for row in oracle]
                    if activation=='relu' and k<len(layers)-1: oracle=[[max(F(0),v) for v in row] for row in oracle]
                actual,_=e.forward(np.array(X,dtype=float),floats,activation)
                np.testing.assert_array_equal(actual,np.array(oracle,dtype=float))
                if activation=='identity':
                    W,b=e.collapse_affine(floats)
                    np.testing.assert_array_equal(np.array(X,dtype=float)@W+b,actual)
            self.assertEqual(e.parameter_count(widths),sum(len(W)*len(b)+len(b) for W,b in layers))
    def test_decimal_all_130_candidates(self):
        inputs,result,_=e.prepare(); y=[0,1,1,0]
        with localcontext() as ctx:
            ctx.prec=60
            for row in result[2]:
                if row['family']=='affine':
                    z=[row['bias'],row['w2']+row['bias'],row['w1']+row['bias'],row['w1']+row['w2']+row['bias']]
                else:z=[-row['scale'],row['scale'],row['scale'],-row['scale']]
                losses=[(D(1)+(-D((2*t-1)*s)).exp()).ln() for s,t in zip(z,y)]
                self.assertAlmostEqual(row['loss'],float(sum(losses)/4),places=14)
                self.assertEqual(row['accuracy'],sum((s>=0)==t for s,t in zip(z,y))/4)
        self.assertEqual(result[0]['affine_best_loss']['loss'],math.log(2))
        self.assertEqual(result[0]['affine_best_accuracy'],.75)
        self.assertEqual(result[0]['relu_best_loss']['scale'],8)
    def test_hand_xor_and_ambiguity(self):
        X,y,_,_=e.load_inputs(); z,tr=e.forward(X,e.xor_layers())
        np.testing.assert_array_equal(tr[0]['z'],[[0,-1],[1,0],[1,0],[2,1]])
        np.testing.assert_array_equal(tr[0]['h'],[[0,0],[1,0],[1,0],[2,1]])
        np.testing.assert_array_equal(z[:,0],[-1,1,1,-1]);self.assertEqual(e.parameter_count([2,3,2,1]),20)
        self.assertEqual(e.forward([[.25,.5]],e.xor_layers())[0][0,0],.5)
        self.assertEqual(e.forward([[.5,.5]],e.xor_layers())[0][0,0],1.)
        # A distinct 2-unit representation |x1-x2| agrees only at the four corners here.
        other=[(np.array([[1,-1],[-1,1]]),np.zeros(2)),(np.array([[2],[2]]),np.array([-1]))]
        np.testing.assert_array_equal(e.forward(X,other)[0],z)
        self.assertEqual(e.forward([[.5,.5]],other)[0][0,0],-1.)
        # Hidden-unit permutation leaves the function unchanged.
        layers=e.xor_layers(); perm=[(layers[0][0][:,::-1],layers[0][1][::-1]),(layers[1][0][::-1,:],layers[1][1])]
        np.testing.assert_array_equal(e.forward(X,perm)[0],z)
    def test_triangle_606_fraction_references(self):
        for d in range(1,7):
            x=[F(i,100) for i in range(101)]; o=x[:]
            for _ in range(d):o=[2*min(v,1-v) for v in o]
            np.testing.assert_allclose(e.triangle([float(v) for v in x],d),[float(v) for v in o],rtol=0,atol=1e-14)
    def test_stable_loss_probability(self):
        z=np.array([-1000.,-8.,0.,8.,1000.]);p=e.sigmoid(z)
        self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p+p[::-1],1,rtol=0,atol=1e-16)
        self.assertEqual(e.binary_metrics([-1000,1000],[0,1])['loss'],0)
        self.assertEqual(e.binary_metrics([-1000,1000],[1,0])['loss'],1000)
        # Classification unchanged by positive scaling; losses are not.
        self.assertEqual(e.binary_metrics([-1,1,1,-1],[0,1,1,0])['accuracy'],1)
    def test_30_api_rejections(self):
        L=e.xor_layers()
        cases=[lambda:e.forward([],L),lambda:e.forward([1,2],L),lambda:e.forward([[1]],L),lambda:e.forward([[True,0]],L),lambda:e.forward([[1+0j,0]],L),lambda:e.forward([['1','0']],L),lambda:e.forward([[np.nan,0]],L),lambda:e.forward([[np.inf,0]],L),lambda:e.forward([[1e101,0]],L),lambda:e.forward([[0,0]],L,'sigmoid'),lambda:e.forward([[0,0]],[]),lambda:e.forward([[0,0]],[(np.ones((2,2)),np.ones((1,2)))]),lambda:e.forward([[0,0]],[(np.ones((2,3)),[0,0])]),lambda:e.forward([[0,0]],[(np.ones((2,129)),np.zeros(129))]),lambda:e.forward([[0,0]],L*6),lambda:e.parameter_count([2,True,1]),lambda:e.parameter_count([2,0,1]),lambda:e.parameter_count([2.,2,1]),lambda:e.parameter_count([2]),lambda:e.sigmoid([[0]]),lambda:e.binary_metrics([0],[2]),lambda:e.binary_metrics([0,0],[1]),lambda:e.xor_layers(-1),lambda:e.xor_layers(101),lambda:e.xor_layers(True),lambda:e.triangle([1.01]),lambda:e.triangle([0],0),lambda:e.triangle([0],True),lambda:e.triangle([0],13),lambda:e.collapse_affine([])]
        self.assertEqual(len(cases),30)
        for i,fn in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):fn()
        # Legal magnitudes can still overflow after several multiplications.
        with self.assertRaises(ValueError):e.forward([[1e100]],[(np.array([[1e100]]),[0])]*4,'identity')
        with self.assertRaises(ValueError):e.collapse_affine([(np.array([[1e100]]),[0])]*4)
    def test_14_malformed_inputs_preserve_outputs(self):
        valid=e.ROOT/'data/samples.csv'; config=e.ROOT/'data/config.json'
        bad_csv=[e.SAMPLE_TEXT.replace('id,x1,x2,y','id,x1,x1,y'),e.SAMPLE_TEXT.replace('b,0,1,1','a,0,1,1'),e.SAMPLE_TEXT.replace('b,0,1,1','b,0,1'),e.SAMPLE_TEXT.replace('b,0,1,1','b,0,1,1,0'),e.SAMPLE_TEXT.replace('b,0,1,1','b,nan,1,1'),e.SAMPLE_TEXT.replace('b,0,1,1','b,inf,1,1'),e.SAMPLE_TEXT.replace('b,0,1,1','b,0,1,0')]
        cfg=config.read_text();bad_cfg=[cfg.replace('"schema": 1','"schema": 1, "schema": 1'),cfg.replace('"schema": 1','"schema": true'),cfg.replace('[0, 1, 2, 4, 8]','[0, 1, 2, 4, 16]'),cfg.replace('"schema": 1','"schema": NaN'),cfg.replace('"schema": 1','"extra": 1, "schema": 1'),cfg.replace('"schema": 1','"schema": Infinity'),'[]']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old';e.run(old); before={p.name:p.read_bytes() for p in old.iterdir()}
            for i,txt in enumerate(bad_csv+bad_cfg):
                inp=root/('bad.csv' if i<7 else 'bad.json');inp.write_text(txt)
                samples=inp if i<7 else valid;conf=config if i<7 else inp;new=root/f'new{i}'
                for out in [new,old]:
                    with self.assertRaises(ValueError):e.run(out,samples,conf)
                self.assertFalse(new.exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
    def test_late_failures_before_write_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';e.run(old);before={p.name:p.read_bytes() for p in old.iterdir()}
            for out in [root/'new',old]:
                with patch.object(e,'csv_bytes',side_effect=ValueError('forced serialization failure')):
                    with self.assertRaises(ValueError):e.run(out)
            self.assertFalse((root/'new').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
            (root/'linked').symlink_to(old,target_is_directory=True)
            with self.assertRaises(ValueError):e.run(root/'linked')
            for name in e.OUTPUT_NAMES:
                (old/name).unlink();(old/name).symlink_to(e.ROOT/'data/samples.csv')
                with self.assertRaises(ValueError):e.run(old)
                (old/name).unlink();(old/name).write_bytes(before[name])
            self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
    def test_teaching_guard_before_display_write(self):
        e.verify_teaching_artifacts()
        with patch.object(e,'calculate',return_value=({'altered':1},[{'x':1}],[{'x':1}],[{'x':1}])):
            with self.assertRaises(ValueError):e.verify_teaching_artifacts()
    def test_input_cast_underflow_and_true_zero(self):
        # The author platform's longdouble carries values smaller than binary64.
        # Already-rounded Python float zeros cannot reveal earlier information loss.
        for value in [np.longdouble('1e-400'), np.longdouble('-1e-400')]:
            if value != 0:
                with self.assertRaises(ValueError): e.forward(np.array([[value, 0]], dtype=np.longdouble), e.xor_layers())
        for value in [np.longdouble(0), np.float64(0)]:
            np.testing.assert_array_equal(e.forward(np.array([[value, 0]]),e.xor_layers())[0], [[-1]])
        smallest = np.nextafter(np.float64(0), np.float64(1))
        self.assertEqual(e.real_array([smallest], 'x', 1)[0], smallest)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old';e.run(old);before={p.name:p.read_bytes() for p in old.iterdir()}
            for tiny in ['1e-400','-1e-400']:
                changed=root/'changed.csv';changed.write_text(e.SAMPLE_TEXT.replace('a,0,0,0','a,'+tiny+',0,0'))
                for out in [root/'new',old]:
                    with self.assertRaises(ValueError):e.run(out,samples=changed)
                self.assertFalse((root/'new').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})

    def test_exact_json_tokens_and_guard(self):
        config=(e.ROOT/'data/config.json').read_text()
        slots=[('"threshold_logit": 0','"threshold_logit": {}'),
               ('[-2, -1, 0, 1, 2]','[-2, -1, {}, 1, 2]'),
               ('[0, 1, 2, 4, 8]','[{}, 1, 2, 4, 8]')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';e.run(old)
            before={p.name:p.read_bytes() for p in old.iterdir()}
            bad=[]
            for source,template in slots:
                self.assertIn(source,config)
                for token in ['1e-400','-1e-400']: bad.append(config.replace(source,template.format(token)))
                zero=root/'zero.json';zero.write_text(config.replace(source,template.format('0e-400')))
                self.assertEqual(e.load_inputs(config=zero)[2],e.DEFAULT_CONFIG)
            # Rounding close to a nonzero integer must not erase a changed value either.
            bad.append(config.replace('"schema": 1','"schema": 1.000000000000000000000000001'))
            original_loader=e.load_inputs
            for i,txt in enumerate(bad):
                changed=root/'changed.json';changed.write_text(txt)
                for out in [root/f'new{i}',old]:
                    with self.assertRaises(ValueError):e.run(out,config=changed)
                self.assertFalse((root/f'new{i}').exists())
                self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
                with patch.object(e,'load_inputs',side_effect=lambda *args, **kwargs:original_loader(config=changed)):
                    with self.assertRaises(ValueError):e.verify_teaching_artifacts()

    def test_repeated_payload_identical(self):
        a=e.prepare()[2];b=e.prepare()[2];self.assertEqual(a,b)
        for name in e.OUTPUT_NAMES:self.assertEqual(a[name],(e.ROOT/'outputs'/name).read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
