"""Independent numerical and adversarial checks for DL030; no network access."""
import copy, csv, io, json, math, random, tempfile, unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
import experiment as e

class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.x,cls.y,cls.split,cls.cfg=e.load_inputs()

    def test_01_data_partition_and_copy_ownership(self):
        e.validate_data(self.x,self.y,self.split)
        rows=e.Rows(self.x,self.y,self.split['train']);original=rows[0][0]
        probe=rows[0][0];probe.add_(100)
        self.assertTrue(torch.equal(rows[0][0],original))
        for kind in ('overlap','missing','boolean','empty'):
            s=copy.deepcopy(self.split)
            if kind=='overlap':s['test'][0]=s['train'][0]
            elif kind=='missing':s['test'].pop()
            elif kind=='boolean':s['train'][0]=True
            else:s['test']=[]
            with self.assertRaises(ValueError):e.validate_data(self.x,self.y,s)
        for x,y in [(self.x.float(),self.y),(self.x,self.y[:,0]),(self.x*float('nan'),self.y),(self.x,self.y*1000)]:
            with self.assertRaises(ValueError):e.validate_data(x,y,self.split)

    def test_02_normalization_fraction_and_registration(self):
        r=e.make_run(self.x,self.y,self.split,self.cfg);m=r['model']
        for j in range(2):
            vals=[Fraction(float(self.x[i,j])) for i in self.split['train']]
            mean=sum(vals)/len(vals);var=sum((v-mean)**2 for v in vals)/len(vals)
            self.assertAlmostEqual(m.mean[j].item(),float(mean),places=15)
            self.assertAlmostEqual(m.scale[j].item(),math.sqrt(float(var)),places=15)
        self.assertEqual(sum(p.numel() for p in m.parameters()),17)
        self.assertEqual(set(dict(m.named_parameters())),{'hidden.weight','hidden.bias','output.weight','output.bias'})
        self.assertEqual(set(dict(m.named_buffers())),{'mean','scale'})
        self.assertFalse(m.mean.requires_grad)

    def test_03_full_reference_forward_and_mse(self):
        rng=np.random.default_rng(303)
        for _ in range(60):
            r=e.make_run(self.x,self.y,self.split,self.cfg);m=r['model'];m.eval()
            with torch.no_grad():
                for p in m.parameters():p.copy_(torch.tensor(rng.uniform(-1,1,p.shape)))
            xx=rng.uniform(-3,3,(rng.integers(1,17),2));yy=rng.uniform(-2,2,(len(xx),1))
            st={k:v.detach().numpy() for k,v in m.state_dict().items()}
            normalized=(xx-st['mean'])/st['scale']
            pred=np.tanh(normalized@st['hidden.weight'].T+st['hidden.bias'])@st['output.weight'].T+st['output.bias']
            actual=m(torch.tensor(xx)).detach().numpy()
            np.testing.assert_allclose(actual,pred,atol=9e-16,rtol=1e-14)
            self.assertAlmostEqual(e.batch_mse(torch.tensor(actual),torch.tensor(yy)).item(),math.fsum((float(a)-float(b))**2 for a,b in zip(pred[:,0],yy[:,0]))/len(xx),places=13)
        with self.assertRaises(ValueError):e.batch_mse(torch.ones(3,1),torch.ones(3))

    def test_04_evaluate_is_observational_and_weighted(self):
        r=e.make_run(self.x,self.y,self.split,self.cfg);e.train_one_epoch(r)
        m=r['model'];m.dropout.eval()  # Deliberately mixed modes.
        modes=[q.training for q in m.modules()];before=e.clone_state(m)
        opt=copy.deepcopy(r['optimizer'].state_dict());rng=torch.get_rng_state().clone();loader=r['generator'].get_state().clone()
        a=e.evaluate(m,r['loaders']['validation'])
        self.assertEqual(modes,[q.training for q in m.modules()])
        self.assertTrue(e.tensor_dict_equal(before,e.clone_state(m)))
        self.assertTrue(e.nested_equal(opt,r['optimizer'].state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(loader,r['generator'].get_state()))
        exact=math.fsum((z['prediction']-z['target'])**2 for z in a['predictions'])/len(a['predictions'])
        self.assertAlmostEqual(a['mse'],exact,places=15)
        self.assertNotAlmostEqual(a['mse'],a['wrong_equal_batch_mean'],places=8)

    def test_05_resume_all_states_at_many_boundaries(self):
        for seed in (11,300,913):
            for cut in (1,4,7):
                cfg=dict(self.cfg,seed=seed,epochs=8)
                full=e.make_run(self.x,self.y,self.split,cfg);e.train_until(full,cut)
                state=e.load_checkpoint_bytes(e.checkpoint_bytes(e.checkpoint(full)))
                e.train_until(full,8);expected=e.checkpoint(full)
                resumed=e.restore(state,self.x,self.y,self.split,cfg);e.train_until(resumed,8)
                self.assertTrue(e.nested_equal(expected,e.checkpoint(resumed)),(seed,cut))
                self.assertTrue(all(sorted(q)==self.split['train'] for q in resumed['orders']))

    def test_06_missing_state_changes_future(self):
        r=e.make_run(self.x,self.y,self.split,self.cfg);e.train_until(r,4);state=e.checkpoint(r)
        e.train_until(r,12);expected=e.clone_state(r['model'])
        for omit in ('optimizer','torch_rng','loader_rng'):
            t=e.restore(state,self.x,self.y,self.split,self.cfg,omit);e.train_until(t,12)
            self.assertGreater(e.maximum_parameter_gap(expected,e.clone_state(t['model'])),1e-5)
            self.assertEqual(t['orders']==r['orders'],omit!='loader_rng')

    def test_07_heldout_intervention(self):
        a=e.make_run(self.x,self.y,self.split,self.cfg);e.train_until(a,12);ea=e.checkpoint(a)
        xx=self.x.clone();yy=self.y.clone();xx[self.split['test']]+=10;yy[self.split['test']]=-50
        b=e.make_run(xx,yy,self.split,self.cfg);e.train_until(b,12);eb=e.checkpoint(b)
        self.assertTrue(e.nested_equal(ea,eb))
        self.assertNotEqual(e.evaluate(a['model'],a['loaders']['test'])['mse'],e.evaluate(b['model'],b['loaders']['test'])['mse'])
        yy=self.y.clone();yy[self.split['validation']]=20
        c=e.make_run(self.x,yy,self.split,self.cfg);e.train_until(c,12)
        self.assertTrue(e.tensor_dict_equal(ea['model'],e.clone_state(c['model'])))
        self.assertNotEqual(a['history'],c['history'])

    def test_08_momentum_fraction_recurrence(self):
        rng=random.Random(34)
        for _ in range(100):
            start=Fraction(rng.randint(-10,10),8);lr=Fraction(1,16);mu=Fraction(3,4)
            p=torch.nn.Parameter(torch.tensor(float(start),dtype=torch.float64));o=torch.optim.SGD([p],lr=float(lr),momentum=float(mu),foreach=False)
            w=start;v=Fraction(0)
            for step in range(6):
                g=Fraction(rng.randint(-10,10),8);v=mu*v+g;w-=lr*v
                p.grad=torch.tensor(float(g),dtype=torch.float64);o.step()
                self.assertEqual(p.item(),float(w));self.assertEqual(o.state[p]['momentum_buffer'].item(),float(v))

    def test_09_snapshot_independence_and_semantics(self):
        r=e.make_run(self.x,self.y,self.split,self.cfg);e.train_until(r,2);s=e.checkpoint(r);blob=e.checkpoint_bytes(s)
        e.train_until(r,3);self.assertEqual(blob,e.checkpoint_bytes(s))
        p=e.semantic_probes()
        self.assertEqual(p['unregistered_parameter_names'],[]);self.assertTrue(p['unregistered_w_has_gradient'])
        for k in ('shallow_state_changed','clone_state_unchanged','eval_does_not_prevent_update','no_grad_train_dropout_changes_output'):self.assertTrue(p[k])
        self.assertEqual(p['momentum_hand'],[{'parameter':.8,'buffer':2.},{'parameter':.4,'buffer':4.}])

    def test_10_checkpoint_corruption_rejected(self):
        r=e.make_run(self.x,self.y,self.split,self.cfg);e.train_until(r,4);s=e.checkpoint(r)
        edits=[lambda s:s.update(format=7),lambda s:s.update(epoch=True),lambda s:s['config'].update(learning_rate=.4),
            lambda s:s['split']['test'].pop(),lambda s:s['input_sha256'].update(a='b'),lambda s:s['runtime'].update(torch='other'),
            lambda s:s['history'].pop(),lambda s:s['orders'][0].pop(),lambda s:s.update(best_epoch=0),
            lambda s:s.update(best_loss=float('nan')),lambda s:s['model']['mean'].add_(1),
            lambda s:s['best']['output.bias'].fill_(float('inf')),lambda s:s['model'].pop('scale'),
            lambda s:s['optimizer']['state'].pop(0),lambda s:s['optimizer']['state'][0]['momentum_buffer'].fill_(float('nan')),
            lambda s:s.update(loader_rng=torch.ones(2)),lambda s:s['optimizer']['param_groups'][0].update(lr=.5)]
        for edit in edits:
            q=copy.deepcopy(s);edit(q)
            with self.assertRaises((ValueError,KeyError,RuntimeError)):e.restore(q,self.x,self.y,self.split,self.cfg)

    def test_11_bad_input_no_new_or_existing_output_change(self):
        for name in e.INPUT_HASHES:
            for exists in (False,True):
                with tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);data=root/'data';data.mkdir()
                    for n in e.INPUT_HASHES:(data/n).write_bytes((e.ROOT/'data'/n).read_bytes())
                    (data/name).write_bytes((data/name).read_bytes()+b' ')
                    output=root/'result'
                    if exists:output.mkdir();(output/'sentinel').write_text('keep')
                    with self.assertRaises(ValueError):e.write_report(output,data)
                    self.assertEqual(output.exists(),exists)
                    if exists:self.assertEqual([p.name for p in output.iterdir()],['sentinel'])

    def test_12_complete_first_step_trace(self):
        q=e.first_step_trace(self.x,self.y,self.split,self.cfg)
        v={k:np.array(a) for k,a in q['values'].items()}
        st={k:np.array(a) for k,a in q['before'].items()}
        post={k:np.array(a) for k,a in q['after'].items()}
        z=(v['x']-st['mean'])/st['scale'];h=np.tanh(z@st['hidden.weight'].T+st['hidden.bias'])
        drop=h*v['mask_scale'];p=drop@st['output.weight'].T+st['output.bias']
        dp=2*(p-v['target'])/len(p);dd=dp@st['output.weight'];dh=dd*v['mask_scale'];da=dh*(1-h*h)
        for name,expected in [('normalized',z),('hidden',h),('dropped_hidden',drop),('prediction',p),('d_prediction',dp),('d_dropped_hidden',dd),('d_hidden',dh),('d_affine',da),('d_normalized',da@st['hidden.weight']),('d_input',(da@st['hidden.weight'])/st['scale'])]:
            np.testing.assert_allclose(v[name],expected,rtol=1e-13,atol=2e-15)
        grads={'hidden.weight':da.T@z,'hidden.bias':da.sum(0),'output.weight':dp.T@drop,'output.bias':dp.sum(0)}
        for name,g in grads.items():
            np.testing.assert_allclose(q['gradients'][name],g,rtol=1e-13,atol=2e-15)
            np.testing.assert_allclose(post[name],st[name]-self.cfg['learning_rate']*g,rtol=1e-13,atol=2e-15)
            self.assertEqual(q['momentum_after'][name],q['gradients'][name])
            for index in np.ndindex(g.shape):
                def scalar_loss(theta):
                    hh=np.tanh(z@theta['hidden.weight'].T+theta['hidden.bias'])
                    pp=(hh*v['mask_scale'])@theta['output.weight'].T+theta['output.bias']
                    return float(np.mean((pp-v['target'])**2))
                plus={k:a.copy() for k,a in st.items()};minus={k:a.copy() for k,a in st.items()}
                plus[name][index]+=1e-5;minus[name][index]-=1e-5
                fd=(scalar_loss(plus)-scalar_loss(minus))/2e-5
                self.assertAlmostEqual(fd,g[index],places=8)
        a2=z@post['hidden.weight'].T+post['hidden.bias'];h2=np.tanh(a2);p2=h2@post['output.weight'].T+post['output.bias']
        np.testing.assert_allclose(q['next_forward_same_batch_eval']['prediction'],p2,rtol=1e-13,atol=2e-15)
        self.assertEqual(q['ids'],[56,10,26,43,14,11,2])

if __name__=='__main__':unittest.main()
