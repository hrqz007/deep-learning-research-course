"""Unit-level scientific/contract tests. Uses unittest, never Python assert."""
import unittest, math, tempfile, gzip, json, hashlib, subprocess, sys, shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import torch
import experiment as e

class Tests(unittest.TestCase):
    def test_all_hand_gradients_exact_reference(self):
        r=e.hand_reference();loss,g=e.hand_torch()
        self.assertAlmostEqual(loss,float(r['loss']),places=15)
        np.testing.assert_allclose(g,[float(v) for v in r['gradient']],atol=2e-16,rtol=0)
        newloss,_=e.hand_torch([float(v) for v in r['updated']])
        self.assertAlmostEqual(newloss,float(r['new_loss']),places=15)
        self.assertLess(newloss,loss)
    def test_central_difference_all_13(self):
        t=np.array([float(v) for v in e.hand_reference()['theta']]);_,g=e.hand_torch(t)
        fd=[]
        for i in range(13):
            a=t.copy();b=t.copy();a[i]+=1e-6;b[i]-=1e-6
            fd.append((e.hand_torch(a)[0]-e.hand_torch(b)[0])/2e-6)
        np.testing.assert_allclose(g,fd,atol=4e-11,rtol=0)
    def test_second_moment_identity(self):
        m=e.moments([-2,0,1,5]);self.assertAlmostEqual(m['second'],m['variance']+m['mean']**2,places=14)
        m=e.moments([2,2,2]);self.assertEqual(m['variance'],0);self.assertEqual(m['second'],4)
    def test_forward_ensemble_exact_enumeration(self):
        # nonzero input mean and correlated coordinates; independent zero-mean fresh weights
        vals=[]
        for x in (np.array([1.,1.]),np.array([3.,3.])):
            for w1 in (-1.,1.):
                for w2 in (-1.,1.):vals.append(w1*x[0]+w2*x[1])
        self.assertEqual(np.mean(vals),0)
        self.assertEqual(np.mean(np.array(vals)**2),10)
        self.assertEqual(2*np.mean(np.array([1.,3.])**2),10)
    def test_correlated_fixed_weight_counterexample(self):
        d=e.extra_mechanisms()['correlated_input'];self.assertAlmostEqual(d['actual_variance'],2)
        self.assertAlmostEqual(d['full_covariance'],2);self.assertAlmostEqual(d['diagonal_only'],1)
    def test_relu_symmetry_and_boundary(self):
        r=e.extra_mechanisms();a=r['relu_moments'];self.assertAlmostEqual(a['relu']['second'],a['z']['second']/2,places=14)
        self.assertGreater(a['relu']['mean'],0);self.assertEqual(r['gate_dependence']['relu_at_zero'],0)
        self.assertEqual(r['gate_dependence']['central_difference_at_zero'],.5)
    def test_layout_matches_official_initializer(self):
        w=torch.empty(2000,16,dtype=e.DTYPE);g=torch.Generator().manual_seed(4)
        torch.nn.init.kaiming_normal_(w,mode='fan_in',nonlinearity='relu',generator=g)
        self.assertAlmostEqual(float(w.square().mean()),2/16,delta=.004)
        # Same generator, same operation must match our fan_in formula exactly.
        g2=torch.Generator().manual_seed(4);expected=torch.randn(2000,16,dtype=e.DTYPE,generator=g2)*math.sqrt(2/16)
        torch.testing.assert_close(w,expected,atol=2e-16,rtol=2e-16)
    def test_paired_initialization(self):
        a=e.Network(width=8,depth=3,alpha=.5,seed=7);b=e.Network(width=8,depth=3,alpha=1.,seed=7)
        for wa,wb in zip(a.params[:-2:2],b.params[:-2:2]):torch.testing.assert_close(wa*2,wb)
        torch.testing.assert_close(a.params[-2],b.params[-2])
    def test_snapshot_variance_decomposition(self):
        x,y=e.load_data();x=torch.tensor(x[:16]);y=torch.tensor(y[:16]);net=e.Network(width=8,depth=3)
        _,rows,_=e.snapshot(net,x,y,torch.ones(16,8,dtype=e.DTYPE)/8)
        for r in rows:
            self.assertAlmostEqual(r['h_variance'],r['mean_unit_data_variance']+r['variance_unit_data_means'],places=13)
    def test_data_rejections(self):
        for x,y in [([[1],[2]],[1,2]),([[True],[2]],[[1],[2]]),([[1],[float('nan')]],[[1],[2]]),([['1'],[2]],[[1],[2]])]:
            with self.assertRaises(ValueError):e.validate_batch(x,y)
    def test_configuration_rejections(self):
        for kwargs in [{'width':True},{'depth':0},{'alpha':float('nan')},{'seed':1.2},{'act':'gelu'},{'alpha':0}]:
            with self.assertRaises(ValueError):e.Network(**kwargs)
    def test_invalid_input_before_write(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'results.json').write_text('DO NOT ALTER');before=(p/'results.json').read_bytes()
            with self.assertRaises(ValueError):e.run(p,steps=0)
            self.assertEqual(before,(p/'results.json').read_bytes())
            with self.assertRaises(FileNotFoundError):e.run(p,data=p/'absent.csv')
            self.assertEqual(before,(p/'results.json').read_bytes())
            new=p/'must_not_exist'
            with self.assertRaises(ValueError):e.run(new,steps=True)
            self.assertFalse(new.exists())
    def test_invalid_cli_before_write(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'results.json').write_text('SENTINEL')
            for flags in (['--steps','0'],['--data',str(p/'absent.csv')]):
                proc=subprocess.run([sys.executable,'-O',str(e.BASE/'experiment.py'),'--output',str(p),*flags],capture_output=True,text=True)
                self.assertNotEqual(proc.returncode,0)
                self.assertEqual((p/'results.json').read_text(),'SENTINEL')
    def test_invalid_figure_source_before_write(self):
        # All files exist, but a late-consumed layer statistic is invalid.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'source';source.mkdir();dest=p/'figures';dest.mkdir()
            for original in (e.BASE/'outputs').iterdir():shutil.copyfile(original,source/original.name)
            target=source/'layer_statistics.csv';text=target.read_text();text=text.replace('linear,0.5,35,1,','linear,nan,35,1,',1);target.write_text(text)
            sentinel=dest/'01_computation_path.png';sentinel.write_bytes(b'SENTINEL')
            proc=subprocess.run([sys.executable,'-O',str(e.BASE/'make_figures.py'),'--source',str(source),'--destination',str(dest)],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertEqual(sentinel.read_bytes(),b'SENTINEL')
            self.assertEqual(len(list(dest.iterdir())),1)
    def test_distribution_lossless_deterministic_gzip(self):
        selected={'values':[0.,-0.,1.2345678901234567,1e-200,-1e200]}
        expected=(json.dumps(selected,allow_nan=False)+'\n').encode('utf-8')
        first=e.distribution_gzip_bytes(selected)
        self.assertEqual(first,e.distribution_gzip_bytes(selected))
        self.assertEqual(first[:10],bytes.fromhex('1f8b08000000000002ff'))
        self.assertEqual(gzip.decompress(first),expected)
        self.assertEqual(json.loads(gzip.decompress(first)),selected)
        with self.assertRaises(ValueError):e.distribution_gzip_bytes({'x':float('nan')})
    def test_full_distribution_reference_bytes(self):
        packed=(e.BASE/'outputs/selected_distributions.json.gz').read_bytes()
        raw=gzip.decompress(packed)
        self.assertEqual(len(raw),23870931)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'14aa21584fc6fd57f1b38b84e11aba100c67c0f22b30c7fd7625460c9ac6cd10')
        groups=json.loads(raw)
        self.assertEqual(len(groups),12)
        self.assertEqual(sum(len(v) for group in groups.values() for v in group.values()),1179648)
        self.assertLess(4*((len(packed)+2)//3),16*1024*1024)
    def test_corrupt_or_missing_gzip_before_figure_write(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'source';source.mkdir()
            for original in (e.BASE/'outputs').iterdir():shutil.copyfile(original,source/original.name)
            target=source/'selected_distributions.json.gz'
            for mode in ('corrupt','missing'):
                if mode=='corrupt':target.write_bytes(b'not a gzip file')
                else:target.unlink()
                dest=p/mode;dest.mkdir();sentinel=dest/'01_computation_path.png';sentinel.write_bytes(b'SENTINEL')
                proc=subprocess.run([sys.executable,'-O',str(e.BASE/'make_figures.py'),'--source',str(source),'--destination',str(dest)],capture_output=True,text=True)
                self.assertNotEqual(proc.returncode,0)
                self.assertEqual(sentinel.read_bytes(),b'SENTINEL')
                self.assertEqual(len(list(dest.iterdir())),1)
    def test_small_case_determinism(self):
        torch.set_num_threads(1);x,y=e.load_data();x=torch.tensor(x[:8]);y=torch.tensor(y[:8])
        a=e.run_case(x,y,'tanh',1.,35,steps=2,width=8,depth=3)
        b=e.run_case(x,y,'tanh',1.,35,steps=2,width=8,depth=3)
        self.assertEqual(a,b)

if __name__=='__main__': unittest.main(verbosity=2)
