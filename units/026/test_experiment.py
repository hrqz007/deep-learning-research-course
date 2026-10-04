"""Independent exact-forward sensitivities, Decimal activations and failure probes."""
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from decimal import Decimal as D, localcontext
from pathlib import Path
from unittest.mock import patch
import json, math, random, tempfile, unittest
import experiment as e

class Test026(unittest.TestCase):
    def test_complete_hand_graph(self):
        n=e.example();g=e.backward(n['L'])
        exact={'x':(F(2),F(39,64)),'y':(F(1,4),F(-13,16)),
               'w':(F(1,2),F(13,4)),'b':(F(-1,4),F(39,32)),
               'm':(F(1),F(39,32)),'a':(F(3,4),F(39,32)),
               'h':(F(9,16),F(13,16)),'s':(F(17,16),F(13,16)),
               'e':(F(13,16),F(13,16)),'q':(F(169,256),F(1,2)),
               'L':(F(169,512),F(1))}
        for key,(value,gradient) in exact.items():
            self.assertEqual(n[key].value,float(value));self.assertEqual(g[n[key]],float(gradient))
        order=e.topological(n['L']);self.assertEqual(len(order),13)
        self.assertEqual(sum(len(x.parents) for x in order),15)
        pos={x:i for i,x in enumerate(order)}
        for node in order:
            for parent,_ in node.parents:self.assertLess(pos[parent],pos[node])

    def test_240_fraction_forward_mode_graphs(self):
        # Independent forward-mode rational oracle: gradient vectors travel in the
        # FORWARD construction, not by the student's reverse graph algorithm.
        rng=random.Random(2601)
        for _ in range(240):
            leaves=[e.Scalar(rng.randrange(-5,6)/8) for _ in range(4)]
            states=[(node,F(node.value),[F(int(i==j)) for j in range(4)]) for i,node in enumerate(leaves)]
            for k in range(24):
                an,a,da=rng.choice(states);bn,b,db=rng.choice(states)
                op=k%4
                if op==0: node=an+bn;value=a+b;d=[u+v for u,v in zip(da,db)]
                elif op==1: node=an*bn;value=a*b;d=[u*b+a*v for u,v in zip(da,db)]
                elif op==2: node=an.relu();value=max(a,0);d=[u if a>0 else F(0) for u in da]
                else: node=-an;value=-a;d=[-u for u in da]
                states.append((node,value,d))
            root,value,d=states[-1];g=e.backward(root)
            self.assertAlmostEqual(root.value,float(value),places=12)
            for leaf,expected in zip(leaves,d):
                self.assertAlmostEqual(g.get(leaf,0),float(expected),places=11)

    def test_225_expanded_polynomial_oracles(self):
        for iw in range(-7,8):
            for ib in range(-7,8):
                x,y,w,b=F(3,2),F(-1,4),F(iw,8),F(ib,8)
                a=w*x+b;err=a*a+w-y
                n=e.example(*map(float,(x,y,w,b)));g=e.backward(n['L'])
                self.assertEqual(n['L'].value,float(err*err/2))
                for key,value in {'w':err*(2*a*x+1),'b':2*a*err,'x':2*a*w*err,'y':-err}.items():
                    self.assertEqual(g[n[key]],float(value))

    def test_shared_paths_repeated_edges_and_identity(self):
        w=e.Scalar(2,name='same');independent=e.Scalar(2,name='same')
        a=w*w;root=a+a+w;g=e.backward(root)
        self.assertEqual(g[w],9);self.assertEqual(len(a.parents),2)
        self.assertEqual(len(e.topological(root)),4)
        self.assertNotIn(independent,g)
        # Disconnecting is an actual model change even with equal forward value.
        broken=e.Scalar(a.value)+w
        self.assertEqual(e.backward(broken)[w],1)
        self.assertEqual(e.backward(a+w)[w],5)

    def test_repeated_calls_seeds_and_immutable_values(self):
        n=e.example();a=e.backward(n['L']);b=e.backward(n['L'])
        self.assertEqual(a,b);self.assertIsNot(a,b)
        c=e.backward(n['L'],3)
        for node in a:self.assertEqual(c[node],3*a[node])
        self.assertTrue(all(v==0 for v in e.backward(n['L'],0).values()))
        with self.assertRaises(FrozenInstanceError):n['w'].value=7
        n2=e.example(w=.4)
        self.assertNotEqual(n2['L'].value,n['L'].value)
        self.assertEqual(e.backward(n['L'])[n['w']],3.25)

    def test_81_decimal_tanh_references(self):
        with localcontext() as c:
            c.prec=100
            for i in range(-40,41):
                x=e.Scalar(float(i));out=x.tanh();actual=e.backward(out)[x]
                z=D(i);q=(-2*abs(z)).exp();expected=4*q/(1+q)**2
                value=(1-q)/(1+q) if i>=0 else -(1-q)/(1+q)
                self.assertLessEqual(abs(out.value-float(value)),2e-16)
                self.assertTrue(math.isclose(actual,float(expected),rel_tol=4e-16,abs_tol=0))
        x=e.Scalar(20.0);self.assertEqual(1-math.tanh(20)**2,0)
        self.assertGreater(e.backward(x.tanh())[x],1e-17)

    def test_relu_kink_and_finite_difference(self):
        for value,target in [(-1.,0.),(0.,0.),(1.,1.)]:
            x=e.Scalar(value);self.assertEqual(e.backward(x.relu())[x],target)
        step=1e-6
        self.assertEqual((max(step,0)-max(-step,0))/(2*step),.5)
        n=e.example();g=e.backward(n['L'])
        for key in ('w','b'):
            plus=dict(x=2.,y=.25,w=.5,b=-.25);minus=plus.copy();plus[key]+=step;minus[key]-=step
            numeric=(e.direct_loss(**plus)-e.direct_loss(**minus))/(2*step)
            self.assertLess(abs(numeric-g[n[key]]),1e-9)

    def test_iterative_3000_node_chain(self):
        x=e.Scalar(.25);v=x
        for _ in range(1500):v=v+1.
        self.assertEqual(v.value,1500.25);self.assertEqual(e.backward(v)[x],1)
        self.assertEqual(len(e.topological(v)),3001)

    def test_numerical_type_and_range_rejections(self):
        bad=[True,False,'1',None,1+0j,[],{},float('nan'),float('inf'),-float('inf'),1e101,10**500]
        for value in bad:
            with self.subTest(value=str(value)[:20]),self.assertRaises(ValueError):e.Scalar(value)
        for a,b in [(True,2),(2,False),(0,1e101),(1e101,0),('2',3),(None,0)]:
            with self.assertRaises(ValueError):e.product(a,b)
        self.assertEqual(e.product(0,2),0)
        with self.assertRaises(ValueError):e.Scalar(1e100)*2
        with self.assertRaises(ValueError):e.Scalar(1e-300)*1e-300
        with self.assertRaises(ValueError):e.backward(e.Scalar(1)*1e-300,1e-300)
        with self.assertRaises(ValueError):e.Scalar(101).tanh()
        with self.assertRaises(ValueError):e.backward(2)
        with self.assertRaises(ValueError):e.backward(e.Scalar(1),True)
        with self.assertRaises(ValueError):e.Scalar(1,parents=[])
        self.assertEqual(e.Scalar(0.0).value,0)
        self.assertEqual(e.Scalar(math.ulp(0)).value,math.ulp(0))
        self.assertEqual((e.Scalar(2)*0).value,0)

    def test_bad_configs_preserve_old_and_new_outputs(self):
        text=(e.HERE/'data/config.json').read_text()
        bad=['[]','{',text.replace('"schema": 1','"schema": 1, "schema": 1'),
             text.replace('"schema": 1','"schema": true'),text.replace('"schema": 1','"extra": 1, "schema": 1')]
        for key,value in [('x','NaN'),('x','Infinity'),('x','1e-400'),('x','-1e-400'),('x','11'),('x','"2"'),
                          ('iterations','true'),('iterations','-1'),('iterations','501'),
                          ('learning_rate','0'),('learning_rate','0.2'),('b','null')]:
            import re
            bad.append(re.sub(r'"'+key+r'"\s*:\s*[^,\n}]+','"'+key+'": '+value,text))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';e.run(output=old);before={p.name:p.read_bytes() for p in old.iterdir()}
            for i,config in enumerate(bad):
                cp=root/'bad.json';cp.write_text(config)
                for out in [old,root/f'new{i}']:
                    with self.assertRaises((ValueError,json.JSONDecodeError)):e.run(cp,out)
                self.assertFalse((root/f'new{i}').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
            for token in ['0e-400','-0e-400']:
                cp=root/'zero.json';cp.write_text(text.replace('"x": 2.0','"x": '+token));self.assertEqual(e.read_config(cp)['x'],0)
            # Legal settings can diverge. No partial files or old/new mixture.
            cfg=e.DEFAULT|{'x':10.,'w':10.,'b':10.,'learning_rate':.1}
            cp=root/'diverge.json';cp.write_text(json.dumps(cfg))
            for out in [old,root/'divergent']:
                with self.assertRaises(ValueError):e.run(cp,out)
            self.assertFalse((root/'divergent').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})

    def test_late_serialization_and_destination_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';e.run(output=old);before={p.name:p.read_bytes() for p in old.iterdir()}
            for out in [old,root/'new']:
                with patch.object(e,'csv_text',side_effect=ValueError('late CSV failure')):
                    with self.assertRaises(ValueError):e.run(output=out)
            self.assertFalse((root/'new').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
            (root/'linked').symlink_to(old,target_is_directory=True)
            with self.assertRaises(ValueError):e.run(output=root/'linked')
            target=old/'graph.csv';target.unlink();target.symlink_to(e.HERE/'data/config.json')
            with self.assertRaises(ValueError):e.run(output=old)
            target.unlink();target.write_bytes(before['graph.csv'])
            collision=root/'collision';collision.mkdir();cp=collision/'summary.json';cp.write_text((e.HERE/'data/config.json').read_text())
            with self.assertRaises(ValueError):e.run(cp,collision)
            self.assertEqual(cp.read_bytes(),(e.HERE/'data/config.json').read_bytes())

    def test_fixed_teaching_digests(self):
        e.require_teaching_inputs()
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            copy=Path(tmp)
            for name in e.TEACHING_HASHES:
                dest=copy/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(e.HERE/name,dest)
            with patch.object(e,'HERE',copy):
                e.require_teaching_inputs()
                for name in e.TEACHING_HASHES:
                    target=copy/name;original=target.read_bytes();target.write_bytes(original+b' ')
                    with self.assertRaises(ValueError):e.require_teaching_inputs()
                    target.write_bytes(original)
                e.require_teaching_inputs()

    def test_41_training_states_independent(self):
        # Formula derived after algebraic substitution, independent of the engine.
        payload=e.serialize_run(e.DEFAULT.copy());import csv,io
        rows=list(csv.DictReader(io.StringIO(payload['training.csv'])))
        w,b=.5,-.25
        for row in rows:
            a=2*w+b;err=a*a+w-.25;gw=err*(4*a+1);gb=2*a*err
            self.assertAlmostEqual(float(row['loss']),.5*err*err,places=14)
            self.assertAlmostEqual(float(row['w']),w,places=14);self.assertAlmostEqual(float(row['b']),b,places=14)
            self.assertAlmostEqual(float(row['gradient_w']),gw,places=14)
            w,b=w-.02*gw,b-.02*gb
        self.assertEqual(len(rows),41)
        self.assertAlmostEqual(float(rows[1]['w']),.435,places=15)
        self.assertAlmostEqual(float(rows[1]['b']),-.274375,places=15)

if __name__=='__main__':unittest.main(verbosity=2)
