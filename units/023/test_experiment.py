"""Independent exact-rational oracles and failure-before-write regression tests."""
from fractions import Fraction as F
from pathlib import Path
import csv, hashlib, importlib.util, io, json, random, tempfile, unittest
from unittest.mock import patch
import numpy as np
import experiment as e


def oracle(X,y,theta):
    n=len(y);p=len(theta);A=[list(map(F,row))+[F(1)] for row in X]
    yy=list(map(F,y));tt=list(map(F,theta))
    r=[sum(a*t for a,t in zip(row,tt))-v for row,v in zip(A,yy)]
    L=sum(v*v for v in r)/(2*n)
    g=[sum(A[i][j]*r[i] for i in range(n))/n for j in range(p)]
    return L,g


def exact_fit(X,y):
    # Exact Fraction elimination is an independent mathematical oracle, not a
    # recommendation to form normal equations in a floating-point solver.
    A=[list(map(F,row))+[F(1)] for row in X];n=len(y);p=len(A[0])
    aug=[[sum(A[k][i]*A[k][j] for k in range(n)) for j in range(p)]+
         [sum(A[k][i]*F(y[k]) for k in range(n))] for i in range(p)]
    for j in range(p):
        pivot=next(i for i in range(j,p) if aug[i][j]);aug[j],aug[pivot]=aug[pivot],aug[j]
        d=aug[j][j];aug[j]=[v/d for v in aug[j]]
        for i in range(p):
            if i!=j:
                v=aug[i][j];aug[i]=[a-v*b for a,b in zip(aug[i],aug[j])]
    return [row[-1] for row in aug]


def snapshot(p):
    return {str(f.relative_to(p)):f.read_bytes() for f in p.rglob('*') if f.is_file()}


class TestExperiment(unittest.TestCase):
    def test_01_exact_loss_gradient_and_updates(self):
        rng=random.Random(2301)
        for _ in range(180):
            n=rng.randrange(2,9);d=rng.randrange(1,5)
            X=[[rng.randrange(-5,6) for j in range(d)] for i in range(n)]
            y=[rng.randrange(-7,8) for i in range(n)]
            t=[rng.randrange(-4,5) for j in range(d+1)]
            L,g=oracle(X,y,t);actual,grad=e.loss_gradient(X,y,t)
            self.assertAlmostEqual(actual,float(L),places=10)
            np.testing.assert_allclose(grad,list(map(float,g)),rtol=2e-14,atol=1e-13)
            eta=F(1,128);end,trace=e.train(X,y,float(eta),4,t)
            exact=list(map(F,t))
            for k,row in enumerate(trace):
                LL,gg=oracle(X,y,exact)
                self.assertAlmostEqual(row['loss'],float(LL),places=9)
                np.testing.assert_allclose(row['theta'],list(map(float,exact)),rtol=1e-13,atol=1e-13)
                exact=[a-eta*b for a,b in zip(exact,gg)]

    def test_02_exact_solutions(self):
        rng=random.Random(2302);count=0
        while count<100:
            X=[[rng.randrange(-4,5),rng.randrange(-4,5)] for _ in range(7)]
            y=[rng.randrange(-9,10) for _ in X]
            try: expected=exact_fit(X,y)
            except StopIteration: continue
            got,ref=e.least_squares(X,y)
            np.testing.assert_allclose(got,list(map(float,expected)),rtol=1e-11,atol=1e-11)
            self.assertEqual(ref['rank'],3);count+=1

    def test_03_directional_quadratic_identity(self):
        rng=np.random.default_rng(2303)
        for _ in range(120):
            X=rng.normal(size=(9,3));y=rng.normal(size=9);t=rng.normal(size=4);v=rng.normal(size=4)
            L,g=e.loss_gradient(X,y,t);L2,_=e.loss_gradient(X,y,t+v)
            expected=L+g@v+np.sum((e.design(X)@v)**2)/(2*len(y))
            self.assertAlmostEqual(L2,expected,places=10)
            check=e.gradient_check(X,y,t,1e-5)
            self.assertLess(max(r['scaled_error'] for r in check),3e-9)

    def test_04_hand_example_and_duplicate_invariance(self):
        X=[[-1],[0],[1]];y=[-1,1,3]
        t,tr=e.train(X,y,.5,1,[0,0])
        self.assertEqual(tr[0]['loss'],11/6)
        np.testing.assert_allclose(t,[2/3,1/2]);self.assertAlmostEqual(tr[1]['loss'],155/216)
        LL,gg=e.loss_gradient(X,y,[1,2]);L2,g2=e.loss_gradient(X*5,y*5,[1,2])
        self.assertEqual(LL,L2);np.testing.assert_array_equal(gg,g2)
        # Broadcasting failure: predictions exactly right, malformed y creates n-by-n loss.
        pred=np.array([-1.,1.,3.]);self.assertEqual(float(np.mean((pred[:,None]-pred)**2)),16/3)

    def test_05_main_exact_reference_and_convergence(self):
        rows=e.load_samples(e.BASE/'data/samples.csv');cfg=e.load_config(e.BASE/'data/config.json')
        summary,trace,preds,checks=e.compute(rows,cfg)
        X,y=e.split_arrays(rows,'train')
        exact=exact_fit([[F(str(v)) for v in z] for z in X],[F(str(v)) for v in y])
        self.assertEqual(exact,[F(2),F(-3,4),F(3,2)])
        self.assertAlmostEqual(summary['baseline'],2.5)
        targets={'train':F(7,60),'validation':F(9,320),'test':F(2,25)}
        for split,val in targets.items():self.assertAlmostEqual(summary['mse'][split]['trained'],float(val))
        self.assertLess(summary['final_gradient_norm'],1e-10)
        self.assertLess(summary['max_training_prediction_difference'],2e-12)
        self.assertLess(summary['max_scaled_gradient_error'],1e-8)
        self.assertTrue(all(b['loss']<=a['loss']+2e-14 for a,b in zip(trace,trace[1:])))
        self.assertAlmostEqual(summary['curvature_upper_bound'],31/6)
        self.assertEqual(len(preds),30);self.assertEqual(len(trace),301)

    def test_06_fixed_split_information_boundary(self):
        rows=e.load_samples(e.BASE/'data/samples.csv');cfg=e.DEFAULT_CONFIG.copy()
        first=e.compute(rows,cfg)
        altered=[dict(r, y=r['y']+1000, x1=r['x1']-10) if r['split']!='train' else dict(r) for r in rows]
        second=e.compute(altered,cfg)
        for key in ('theta','reference_theta','baseline','final_loss','max_scaled_gradient_error'):
            self.assertEqual(first[0][key],second[0][key])
        self.assertEqual(first[1],second[1]);self.assertEqual(first[3],second[3])

    def test_07_rank_deficiency_and_instability(self):
        X=[[-1,-1],[0,0],[1,1]];y=[-1,1,3]
        ref,info=e.least_squares(X,y);self.assertEqual(info['rank'],2)
        np.testing.assert_allclose(ref,[1,1,1],atol=1e-13)
        end,_=e.train(X,y,.2,300,[4,-2,0])
        self.assertAlmostEqual(end[0]-end[1],6);np.testing.assert_allclose(e.predict(X,end),y,atol=1e-12)
        # No-intercept one-dimensional recurrence: theta_{t+1}= (1-eta) theta_t+eta.
        _,bad=e.train([[0]],[1],2.1,30,[0,0]);self.assertGreater(bad[-1]['loss'],bad[0]['loss'])
        _,edge=e.train([[0]],[1],2,10,[0,0]);self.assertTrue(all(r['loss']==.5 for r in edge))
        near=np.column_stack(([-1.,0.,1.],np.array([-1.,0.,1.])+1e-8*np.array([1.,-2.,1.])))
        target=e.design(near)@np.array([2.,-1.,.5]);got,info=e.least_squares(near,target)
        self.assertEqual(info['rank'],3);np.testing.assert_allclose(e.predict(near,got),target,atol=1e-14)

    def test_08_invalid_math_inputs(self):
        cases=[lambda:e.loss_gradient([[1]],[[1]],[0,0]),lambda:e.loss_gradient([[1]],[1],[0]),
               lambda:e.loss_gradient([[True]],[1],[0,0]),lambda:e.loss_gradient([[True,2]],[1],[0,0,0]),
               lambda:e.loss_gradient([[1]],[False],[0,0]),lambda:e.loss_gradient([[1]],[1],[0,True]),
               lambda:e.loss_gradient([[complex(1,1)]],[1],[0,0]),lambda:e.loss_gradient([['1']],[1],[0,0]),
               lambda:e.loss_gradient([[float('nan')]],[1],[0,0]),lambda:e.loss_gradient([[float('inf')]],[1],[0,0]),
               lambda:e.loss_gradient([[]],[1],[0]),lambda:e.loss_gradient([[1]],[],[0,0]),
               lambda:e.loss_gradient([[1],[2]],[1],[0,0]),lambda:e.loss_gradient([[1e101]],[1],[0,0]),
               lambda:e.loss_gradient([[0]],[1e-200],[0,0]),lambda:e.train([[1]],[1],0),
               lambda:e.train([[1]],[1],-1),lambda:e.train([[1]],[1],True),lambda:e.train([[1]],[1],1e308),
               lambda:e.train([[1]],[1],.1,True),lambda:e.train([[1]],[1],.1,-1),lambda:e.train([[1]],[1],.1,10001),
               lambda:e.train([[1]],[1],.1,1.0),lambda:e.gradient_check([[1]],[1],[1,1],0),
               lambda:e.gradient_check([[1]],[1],[1,1],1e-100),lambda:e.gradient_check([[1]],[1],[1,1],True),
               lambda:e.train(np.ones((10000,2)),np.ones(10000),.1,10000),
               lambda:e.loss_gradient(np.ones((2,65)),[1,2],np.zeros(66))]
        for call in cases:
            with self.subTest(call=call),self.assertRaises((ValueError,FloatingPointError,OverflowError)):call()
        self.assertEqual(len(cases),28)

    def test_09_malformed_csv_and_config_preserve_outputs(self):
        samples=(e.BASE/'data/samples.csv').read_text();cfg=json.loads((e.BASE/'data/config.json').read_text())
        bad_csv=[samples.replace('x1,x2','x1,x1'),samples.replace('x1,x2','x2,x1'),samples.replace(',y\n',',y,extra\n'),
                 samples.replace('train-00','train-01',1),samples.replace('train-00,train','train-00,other',1),
                 samples.replace('train-00,train,-2','train-00,train,nan',1),samples.replace('train-00,train,-2','train-00,train,',1),
                 samples.replace('train-00,train,-2','train-00,train, -2',1),samples.replace('train-00,train,-2,-1','train-00,train,-2',1),
                 '\n'.join(z for z in samples.splitlines() if ',test,' not in z)+'\n',
                 'sample_id,split,x1,x2,y\n', samples.replace('train-00,train,-2','train-00,train,inf',1)]
        bad_cfg=[dict(cfg,steps=True),dict(cfg,steps=10001),dict(cfg,initial=[0,0]),dict(cfg,initial=[True,0,0]),
                 dict(cfg,learning_rate=1e308),dict(cfg,fd_step=1e-100),dict(cfg,gradient_point=[1,1,1]),
                 dict(cfg,extra=2)]
        # gradient_point=[1,1,1] is valid; use an invalid type instead.
        bad_cfg[6]=dict(cfg,gradient_point=[0,0,'0'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'samples.csv';config=root/'config.json';out=root/'old';out.mkdir()
            for name in e.OUTPUT_NAMES:(out/name).write_text('keep '+name)
            (out/'unrelated.txt').write_text('keep');before=snapshot(out)
            for i,text in enumerate(bad_csv):
                data.write_text(text);config.write_text(json.dumps(cfg))
                for dest in (out,root/'new'):
                    with self.assertRaises((ValueError,FloatingPointError,OverflowError,csv.Error)):e.run(config,data,dest)
                    self.assertEqual(snapshot(out),before);self.assertFalse((root/'new').exists())
            data.write_text(samples)
            for value in bad_cfg:
                config.write_text(json.dumps(value))
                for dest in (out,root/'new'):
                    with self.assertRaises((ValueError,FloatingPointError,OverflowError)):e.run(config,data,dest)
                    self.assertEqual(snapshot(out),before);self.assertFalse((root/'new').exists())
            # Syntactically valid run diverges during computation: still no output mutation.
            config.write_text(json.dumps(dict(cfg,learning_rate=100,steps=300)))
            with self.assertRaises((ValueError,FloatingPointError)):e.run(config,data,out)
            self.assertEqual(snapshot(out),before)
            config.write_text(json.dumps(cfg)[:-1]+',"steps":3}')
            with self.assertRaises(ValueError):e.run(config,data,out)
            self.assertEqual(snapshot(out),before)

    def test_10_serialization_and_path_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'old';out.mkdir();(out/'summary.json').write_text('sentinel');before=snapshot(out)
            real=e.compute(e.load_samples(e.BASE/'data/samples.csv'),e.DEFAULT_CONFIG.copy())
            bad=list(real);bad[3]=[dict(real[3][0],analytic=float('nan'))]
            with patch.object(e,'compute',return_value=tuple(bad)):
                for dest in (out,root/'new'):
                    with self.assertRaises(ValueError):e.run(output=dest)
            self.assertEqual(snapshot(out),before);self.assertFalse((root/'new').exists())
            target=root/'target';target.write_text('safe');(out/'trajectory.csv').symlink_to(target)
            with self.assertRaises(ValueError):e.run(output=out)
            self.assertEqual(target.read_text(),'safe');self.assertEqual((out/'summary.json').read_text(),'sentinel')
            (root/'alias').symlink_to(out,target_is_directory=True)
            with self.assertRaises(ValueError):e.run(output=root/'alias')
            with self.assertRaises(ValueError):e.write_outputs({'target':'bad'},root,[target])
            with self.assertRaises(ValueError):e.write_outputs({'../bad':'bad'},out)
            self.assertEqual(target.read_text(),'safe')

    def test_11_fixed_artifact_guards(self):
        e.teaching_payload()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg=p/'config.json';data=p/'samples.csv'
            cfg.write_text(json.dumps(dict(e.DEFAULT_CONFIG,steps=2)))
            with self.assertRaises(ValueError):e.teaching_payload(config=cfg)
            data.write_text((e.BASE/'data/samples.csv').read_text().replace('train-00','changed-id',1))
            with self.assertRaises(ValueError):e.teaching_payload(samples=data)
            for name in e.OUTPUT_NAMES:(p/name).write_bytes((e.BASE/'outputs'/name).read_bytes())
            text=json.loads((p/'summary.json').read_text());text['theta'][0]+=1
            (p/'summary.json').write_text(json.dumps(text))
            with self.assertRaises(ValueError):e.teaching_payload(results=p)

    def test_12_input_conversion_underflow(self):
        tiny=np.longdouble('1e-400')
        if tiny != 0:
            with self.assertRaises(ValueError):e.array([tiny],'tiny',1)
            with self.assertRaises(ValueError):e.scalar(tiny,'tiny')
            with self.assertRaises(ValueError):e.loss_gradient([[0]],[tiny],[0,0])
        np.testing.assert_array_equal(e.array([np.longdouble('0')],'zero',1),[0.])
        self.assertEqual(e.checked_float_text('0e-400'),0.)
        for text in ('1e-400','-1e-400'):
            with self.assertRaises(ValueError):e.checked_float_text(text)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';e.run(output=old);before=snapshot(old)
            raw=(e.BASE/'data/samples.csv').read_text();lines=raw.splitlines();parts=lines[1].split(',');parts[-1]='1e-400';lines[1]=','.join(parts);data=root/'tiny.csv';data.write_text('\n'.join(lines)+'\n')
            for out in (old,root/'newcsv'):
                with self.assertRaises(ValueError):e.run(samples=data,output=out)
                self.assertEqual(snapshot(old),before)
            self.assertFalse((root/'newcsv').exists())
            cfg=root/'tiny.json';cfg.write_text(json.dumps(e.DEFAULT_CONFIG).replace('0.0','1e-400',1))
            for out in (old,root/'newjson'):
                with self.assertRaises(ValueError):e.run(config=cfg,output=out)
                self.assertEqual(snapshot(old),before)
            self.assertFalse((root/'newjson').exists())

if __name__=='__main__':unittest.main(verbosity=2)
