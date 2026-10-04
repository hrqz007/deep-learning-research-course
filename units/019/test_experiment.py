"""Independent Decimal/Fraction oracles, boundaries, and fresh-process checks."""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import csv
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
import experiment as ex


def dec(x):
    return Decimal(x.numerator) / Decimal(x.denominator)


def oracle(p, q=None):
    """Exact rational inputs; Decimal natural log at 80 digits, not math.log."""
    with localcontext() as ctx:
        ctx.prec=80
        pd=[dec(x) for x in p]
        qd=[dec(x) for x in (p if q is None else q)]
        h=-sum((x*x.ln() for x in pd if x),Decimal(0))
        if any(x>0 and y==0 for x,y in zip(pd,qd)):
            return float(h),math.inf,math.inf
        ce=-sum((x*y.ln() for x,y in zip(pd,qd) if x),Decimal(0))
        d=sum((x*(x.ln()-y.ln()) for x,y in zip(pd,qd) if x),Decimal(0))
        return float(h),float(ce),float(d)


def rational_grid(n=6):
    return [tuple(Fraction(v,n) for v in (a,b,n-a-b)) for a in range(n+1) for b in range(n-a+1)]


class TestFiniteInformation(unittest.TestCase):
    def close(self,a,b,tol=2e-12):
        if math.isinf(b): self.assertEqual(a,b)
        else: self.assertAlmostEqual(a,b,delta=tol)

    def test_01_exact_hand_bits(self):
        p,q=[.5,.25,.25],[.25,.25,.5]
        self.assertEqual(ex.entropy(p,2),1.5)
        self.assertEqual(ex.cross_entropy(p,q,2),1.75)
        self.assertEqual(ex.kl(p,q,2),.25)
        self.assertEqual(ex.kl_terms(p,q,2),(.5,0.,-.25))
        self.assertEqual(ex.entropy([1,0,0]),0)
        self.assertEqual(ex.kl([1,0],[1,0]),0)

    def test_02_decimal_exhaustive(self):
        grid=rational_grid()
        self.assertEqual(len(grid),28)
        for p,q in itertools.product(grid,repeat=2):
            h,ce,d=oracle(p,q)
            self.close(ex.entropy(p),h)
            self.close(ex.cross_entropy(p,q),ce)
            self.close(ex.kl(p,q),d)
            self.assertGreaterEqual(ex.kl(p,q),-2e-12)
            if math.isfinite(ce): self.close(ex.entropy(p)+ex.kl(p,q),ce)
        self.assertEqual(len(grid)**2,784)

    def test_03_support_directions(self):
        self.assertEqual(ex.kl([.5,.5,0],[1,0,0]),math.inf)
        self.close(ex.kl([1,0,0],[.5,.5,0]),math.log(2))
        self.assertEqual(ex.cross_entropy([0,1],[0,1]),0)
        self.assertEqual(ex.cross_entropy([1,0],[0,1]),math.inf)
        self.assertEqual(ex.kl_terms([0,1],[1,0]),(0.,math.inf))
        self.assertNotEqual(ex.kl([.75,.25],[.5,.5]),ex.kl([.5,.5],[.75,.25]))
        self.assertGreater(ex.kl([.1,.9],[.9,.1]),ex.kl([.1,.9],[.5,.5])+ex.kl([.5,.5],[.9,.1]))

    def test_04_label_enumeration(self):
        q=[Fraction(1,4),Fraction(1,4),Fraction(1,2)]
        count=0
        for n in range(1,6):
            for ys in itertools.product(range(3),repeat=n):
                # Exact code lengths: A and B use 2 bits, C uses 1 bit.
                bits=Fraction(sum(2 if y<2 else 1 for y in ys),n)
                self.close(ex.mean_nll(ys,q),float(bits)*math.log(2))
                self.close(ex.mean_nll(ys,q),ex.cross_entropy(ex.empirical_distribution(ys,3),q))
                count+=1
        self.assertEqual(count,363)

    def test_05_extreme_logs(self):
        for tiny in [1e-20,1e-100,1e-300,5e-324]:
            self.assertTrue(math.isfinite(ex.cross_entropy([0,1],[1,tiny])))
            self.close(ex.cross_entropy([0,1],[1,tiny]),-math.log(tiny))
        self.assertTrue(math.isfinite(ex.kl([1e-300,1],[1,1e-300])))
        self.assertEqual(ex.kl([.2,.3,.5],[.2,.3,.5]),0)
        self.assertTrue(math.isfinite(ex.kl([.5,.5],[.5+1e-12,.5-1e-12])))

    def test_06_rejections(self):
        bad=[[],[0,0],[.2,.2],[-.1,1.1],[math.nan,1],[math.inf,0],[True,0],['.5',.5],None,(),[1]+[0]*64,[Fraction(1,10**400),1]]
        for value in bad:
            with self.assertRaises(ValueError): ex.entropy(value)
        for base in [0,1,.5,-2,math.inf,math.nan,True,'2',10**400,Fraction(10**100+1,10**100)]:
            with self.assertRaises(ValueError): ex.entropy([.5,.5],base)
        with self.assertRaises(ValueError): ex.kl([1,0],[1])
        for labels,states in [([],3),([0],0),([0],65),([0],True),([True],3),([-1],3),([3],3),([.5],3),('01',3)]:
            with self.assertRaises(ValueError): ex.empirical_distribution(labels,states)
        for alpha in [-.1,1.1,math.nan,math.inf,True,'0.1',10**400,Fraction(1,10**400)]:
            with self.assertRaises(ValueError): ex.mix_with_uniform([.5,.5],alpha)
        self.assertEqual(ex.distribution([.5,.5+1e-13])[0]+ex.distribution([.5,.5+1e-13])[1],1.0)
        with self.assertRaises(ValueError): ex.distribution([.5,.5+1e-10])

    def test_07_data_validation(self):
        malformed=[('name,A,B,C\np,.5,.5,0\np,1,0,0\n',ex.load_distributions),('bad,A,B,C\np,.5,.5,0\n',ex.load_distributions),('name,A,B,C\np,-1,2,0\n',ex.load_distributions),('position,label\n2,0\n',ex.load_labels),('position,label\n1,3\n',ex.load_labels),('position,label\n',ex.load_labels)]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'bad.csv'
            for text,reader in malformed:
                path.write_text(text)
                with self.assertRaises(ValueError): reader(path)
        self.assertEqual(len(ex.load_distributions()),8)
        self.assertEqual(ex.load_labels(),[0,0,1,2])

    def test_08_direction_and_smoothing(self):
        ds=ex.load_distributions();p=ds['target']
        self.assertLess(ex.kl(p,ds['broad']),ex.kl(p,ds['left']))
        self.assertLess(ex.kl(ds['left'],p),ex.kl(ds['broad'],p))
        self.close(ex.kl(ds['left'],p),ex.kl(ds['right'],p))
        self.assertEqual(ex.kl(p,p),0)
        self.assertEqual(ex.mix_with_uniform([1,0,0],0),(1,0,0))
        self.close(ex.entropy(ex.mix_with_uniform([1,0,0],1)),math.log(3))
        s=ex.mix_with_uniform([1,0,0],.02)
        self.assertTrue(all(x>0 for x in s))
        self.close(ex.cross_entropy([.5,.5,0],s),-.5*(math.log(74/75)+math.log(1/150)))

    def test_09_fresh_process_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'one',Path(td)/'two'
            for out in [a,b]:
                subprocess.run([sys.executable,str(ex.ROOT/'experiment.py'),'--output-dir',str(out)],cwd=td,capture_output=True,text=True,check=True)
            for name in ['summary.json','directional.csv','epsilon.csv']:
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
                self.assertEqual((a/name).read_bytes(),(ex.ROOT/'outputs'/name).read_bytes())
            report=json.loads((a/'summary.json').read_text())
            self.assertEqual(report['zeros']['forward'],'Infinity')
            self.assertEqual(report['best_candidates'],{'forward':['broad'],'reverse':['left','right']})

    def test_10_subnormal_smoothing_preserves_support_or_rejects(self):
        for q in ([1,0],[1,0,0],[1]+[0]*63):
            with self.assertRaises(ValueError):ex.mix_with_uniform(q,5e-324)
        for alpha in (1e-300,1e-100,.02,1):
            self.assertTrue(all(x>0 for x in ex.mix_with_uniform([1,0,0],alpha)))
        self.assertEqual(ex.mix_with_uniform([1],5e-324),(1,))

    def test_11_bad_csv_preserves_old_outputs(self):
        import shutil
        source=(ex.ROOT/'data/distributions.csv').read_text()
        label_source=(ex.ROOT/'data/labels.csv').read_text()
        malformed=[
            ('distributions.csv',source.replace('name,A,B,C','name,A,A,B,C')),
            ('distributions.csv',source.replace('name,A,B,C','name,name,A,B,C')),
            ('distributions.csv',source.replace('main_p,0.5,0.25,0.25','main_p,0.5,0.5')),
            ('distributions.csv',source.replace('main_p,0.5,0.25,0.25','main_p,0.5,0.25,0.25,extra')),
            ('distributions.csv','\n'.join(line for line in source.splitlines() if not line.startswith('zero_q,'))+'\n'),
            ('labels.csv','position,label,label\n1,0,1\n'),
            ('labels.csv','position,position,label\n9,1,0\n'),
            ('labels.csv','position,label\n1\n'),
            ('labels.csv','position,label\n1,0,extra\n'),
        ]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data').mkdir();shutil.copyfile(ex.ROOT/'experiment.py',root/'experiment.py')
            old=root/'old';old.mkdir();names=['summary.json','directional.csv','epsilon.csv']
            for name in names:shutil.copyfile(ex.ROOT/'outputs'/name,old/name)
            before={name:(old/name).read_bytes() for name in names}
            for target_name,bad in malformed:
                (root/'data/distributions.csv').write_text(source)
                (root/'data/labels.csv').write_text(label_source)
                (root/'data'/target_name).write_text(bad)
                for output in (old,root/'new'):
                    result=subprocess.run([sys.executable,str(root/'experiment.py'),'--output-dir',str(output)],capture_output=True,text=True)
                    self.assertNotEqual(result.returncode,0)
                self.assertFalse((root/'new').exists())
                self.assertEqual(before,{name:(old/name).read_bytes() for name in names})

if __name__=='__main__': unittest.main(verbosity=2)
