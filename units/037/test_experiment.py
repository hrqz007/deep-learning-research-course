"""Independent numerical/contract tests; unittest guards remain active with -O."""
from fractions import Fraction as Q
from pathlib import Path
import csv, io, hashlib, inspect, itertools, json, math, shutil, subprocess, sys, tempfile, unittest
import numpy as np
import torch
import experiment as e

def fraction_reference():
    x=[[Q(1),Q(2)],[Q(-1),Q(1)]];y=[Q(7,10),Q(-1,5)]
    t=list(map(Q,['.5','-.2','.3','.4','.1','.2','.6','-.5','.1']))
    mask=[[1,0],[1,1]];grad=[Q(0) for _ in t];pred=[];paths=[]
    for a,target,m in zip(x,y,mask):
        z=[a[0]*t[j]+a[1]*t[2+j]+t[4+j] for j in range(2)]
        h=[max(Q(0),v) for v in z];d=[h[j]*m[j]*2 for j in range(2)];p=sum(d[j]*t[6+j] for j in range(2))+t[8]
        pred.append(p);dp=(p-target)/2;dz=[dp*t[6+j]*m[j]*2*int(z[j]>0) for j in range(2)]
        g=[a[0]*dz[0],a[0]*dz[1],a[1]*dz[0],a[1]*dz[1],*dz,dp*d[0],dp*d[1],dp];paths.append(g)
        grad=[u+v for u,v in zip(grad,g)]
    penidx=(0,1,2,3,6,7);pen=sum(t[j]**2 for j in penidx)/20
    for j in penidx:grad[j]+=t[j]/10
    loss=sum((p-v)**2 for p,v in zip(pred,y))/4+pen
    return t,loss,grad,paths,[a-b/20 for a,b in zip(t,grad)]

def np_forward(p,x,scale=None):
    w,b,v,d,u,c=p;a=np.tanh(x@w+b);h=a if scale is None else a*scale;b2=np.tanh(h@v+d);z=(b2@u+c)[:,0]
    return z,(a,h,b2)

def np_gradient(p,x,y,scale=None):
    z,(a,h,b2)=np_forward(p,x,scale);dz=(1/(1+np.exp(-z))-y)[:,None]/len(x)
    du=b2.T@dz;dc=dz.sum(0);db2=dz@p[4].T;dt=db2*(1-b2*b2);dv=h.T@dt;dd=dt.sum(0)
    dh=dt@p[2].T;da=dh if scale is None else dh*scale;dz1=da*(1-a*a)
    return [x.T@dz1,dz1.sum(0),dv,dd,du,dc]

def np_metrics(p,x,y):
    z,_=np_forward(p,x);pr=1/(1+np.exp(-z));return float(np.mean(np.logaddexp(0,z)-y*z)),pr

def independent_replay(train,validation,config,seed,epochs=160):
    """No production network, loss, augmentation, or optimizer is called here."""
    init=np.random.default_rng(seed+20000);shapes=((2,24),(24,),(24,24),(24,),(24,1),(1,))
    p=[init.normal(0,math.sqrt(2/sum(s)),s) if len(s)==2 else np.zeros(s) for s in shapes]
    m=[np.zeros_like(a) for a in p];v=[np.zeros_like(a) for a in p]
    order=np.random.default_rng(seed+10000);drop=np.random.default_rng(seed+30000);aug=np.random.default_rng(seed+40000)
    x,y=train['x'],train['y'];vx,vy=validation['x'],validation['y'];hist=[];best=float('inf');wait=0;best_p=None;t=0;best_epoch=0
    hist.append((np_metrics(p,x,y)[0],np_metrics(p,vx,vy)[0]));counts=[0,0]
    for epoch in range(1,epochs+1):
        ix=order.permutation(len(x))
        for start in range(0,len(x),16):
            a=x[ix[start:start+16]].copy();target=y[ix[start:start+16]].copy()
            if config.startswith('augment_'):
                chosen=aug.random(len(a))<.5;counts[0]+=int(chosen.sum())
                coord=1 if config=='augment_valid' else 0;a[chosen,coord]*=-1
                if coord==0:counts[1]+=int(chosen.sum())
            scale=(drop.random((len(a),24))>=.3)/.7 if config=='dropout' else None
            if config=='label_smooth':target=.9*target+.05
            grad=np_gradient(p,a,target,scale);t+=1
            for j,g in enumerate(grad):
                m[j]=.9*m[j]+.1*g;v[j]=.99*v[j]+.01*g*g
                rate=.1 if config=='decay' and j in (0,2,4) else 0
                p[j]=p[j]*(1-.01*rate)-.01*(m[j]/(1-.9**t))/(np.sqrt(v[j]/(1-.99**t))+1e-8)
        tr=np_metrics(p,x,y)[0];va=np_metrics(p,vx,vy)[0];hist.append((tr,va))
        if config=='early_stop':
            if va<best:best=va;best_p=[a.copy() for a in p];best_epoch=epoch;wait=0
            else:wait+=1
            if wait>=20:break
    return best_p if config=='early_stop' else p,hist,best_epoch if config=='early_stop' else epoch,t,counts

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.data=e.read_data()
        cls.saved=json.loads((e.BASE/'outputs/run-details.json').read_text())
    def test_fraction_all_parameters_paths_update(self):
        t,l,g,paths,new=fraction_reference();r=e.hand_trace()
        self.assertEqual(l,Q(741,2500));np.testing.assert_allclose(r['initial']['sample_parameter_paths'],[[float(a) for a in b] for b in paths],atol=1e-15,rtol=0)
        np.testing.assert_allclose(r['initial']['gradient'],list(map(float,g)),atol=1e-15,rtol=0)
        np.testing.assert_allclose(r['updated_theta'],list(map(float,new)),atol=1e-15,rtol=0)
        self.assertAlmostEqual(r['same_mask_next']['objective'],.16755551015269082,places=14)
    def test_autograd_all_masks_all_parameters(self):
        for theta in (e.HAND_INITIAL,np.array(e.hand_trace()['updated_theta'])):
            for bits in itertools.product((0.,1.),repeat=4):
                mask=np.array(bits).reshape(2,2);r=e.hand_forward(theta,mask);loss,g,gx=e.hand_torch(theta,mask)
                self.assertAlmostEqual(loss,r['objective'],places=14)
                np.testing.assert_allclose(g,r['gradient'],atol=8e-16,rtol=0);np.testing.assert_allclose(gx,r['input_gradient'],atol=5e-16,rtol=0)
                fd=[]
                for j in range(9):
                    a=theta.copy();b=theta.copy();a[j]+=1e-6;b[j]-=1e-6
                    fd.append((e.hand_forward(a,mask)['objective']-e.hand_forward(b,mask)['objective'])/2e-6)
                np.testing.assert_allclose(g,fd,atol=2e-10,rtol=0)
    def test_exact_dropout_expectation_not_nonlinearity(self):
        r=e.enumerate_masks();self.assertEqual(len(r['rows']),16)
        self.assertAlmostEqual(sum(row['probability'] for row in r['rows']),1)
        np.testing.assert_allclose(r['mean_prediction'],[.42,-.3],atol=2e-16)
        self.assertAlmostEqual(r['mean_data_loss'],.2317,places=14);self.assertAlmostEqual(r['no_dropout']['data_loss'],.0221,places=14)
        self.assertAlmostEqual(r['mean_data_loss'],r['no_dropout']['data_loss']+r['variance_term'],places=14)
        self.assertGreater(abs(r['mean_sigmoid_prediction'][0]-1/(1+math.exp(-.42))),.01)
        # Mean conditional-mask gradient is derivative of a finite expected objective.
        fd=[]
        for j in range(9):
            a=e.HAND_INITIAL.copy();b=a.copy();a[j]+=1e-6;b[j]-=1e-6
            plus=sum(e.hand_forward(a,np.array(bits).reshape(2,2))['objective'] for bits in itertools.product((0.,1.),repeat=4))/16
            minus=sum(e.hand_forward(b,np.array(bits).reshape(2,2))['objective'] for bits in itertools.product((0.,1.),repeat=4))/16
            fd.append((plus-minus)/2e-6)
        np.testing.assert_allclose(fd,r['mean_gradient'],atol=2e-10,rtol=0)
    def test_inverted_moments_exact(self):
        for q in (.2,.5,.8,1.):
            h=np.array([2.,-3.]);mean=(1-q)*0+q*h/q;second=q*(h/q)**2
            np.testing.assert_allclose(mean,h);np.testing.assert_allclose(second-h*h,(1-q)/q*h*h,atol=1e-14)
    def test_numpy_network_all_697_parameter_gradients(self):
        net=e.Network(372,p=.3);p=[a.detach().numpy().copy() for a in net.params]
        x=self.data['train']['x'][:5];y=self.data['train']['y'][:5];mask=(np.random.default_rng(1).random((5,24))>.3).astype(float)
        z=net(torch.tensor(x),torch.tensor(mask));torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor(.9*y+.05)).backward()
        g=np_gradient(p,x,.9*y+.05,mask/.7)
        self.assertEqual(sum(a.size for a in p),697)
        for a,b in zip(g,net.params):np.testing.assert_allclose(a,b.grad.numpy(),atol=1e-16,rtol=1e-13)
    def test_all_21_full_trajectories_independently_rebuilt(self):
        with (e.BASE/'outputs/training-history.csv').open() as f:rows=list(csv.DictReader(f))
        for d in self.saved['runs']:
            p,h,selected,steps,counts=independent_replay(self.data['train'],self.data['validation'],d['config'],d['seed'])
            self.assertEqual(selected,d['selected_epoch']);self.assertEqual(steps,d['updates']);self.assertEqual(counts,[d['selected_augmentation_count'],d['semantic_change_count']])
            for a,b in zip(p,d['parameters']):np.testing.assert_allclose(a,b,atol=2e-10,rtol=2e-10)
            actual=[(float(r['train_ce']),float(r['validation_ce'])) for r in rows if r['config']==d['config'] and int(r['seed'])==d['seed']]
            np.testing.assert_allclose(h,actual,atol=2e-11,rtol=0)
            te,pr=np_metrics(p,self.data['test']['x'],self.data['test']['y']);self.assertAlmostEqual(te,d['test']['ce'],places=10)
    def test_early_stop_is_baseline_prefix_and_restore(self):
        rows=list(csv.DictReader(io.StringIO((e.BASE/'outputs/training-history.csv').read_text())))
        for seed in e.SEEDS:
            a=[r for r in rows if r['config']=='baseline' and int(r['seed'])==seed];b=[r for r in rows if r['config']=='early_stop' and int(r['seed'])==seed]
            for ra,rb in zip(a,b):
                for key in ('train_ce','validation_ce','stochastic_training_objective'):self.assertEqual(ra[key],rb[key])
            d=next(r for r in self.saved['runs'] if r['config']=='early_stop' and r['seed']==seed)
            self.assertEqual(d['epochs_run']-d['selected_epoch'],20)
            self.assertEqual(float(b[d['selected_epoch']]['validation_ce']),d['validation']['ce'])
    def test_test_predictions_recompute_all_metrics(self):
        rows=list(csv.DictReader(io.StringIO((e.BASE/'outputs/test-predictions.csv').read_text())))
        self.assertEqual(len(rows),8064)
        for d in self.saved['runs']:
            sub=[r for r in rows if r['config']==d['config'] and int(r['seed'])==d['seed']];p=np.array([float(r['probability']) for r in sub]);y=np.array([int(r['y']) for r in sub])
            self.assertEqual([int(r['id']) for r in sub],self.data['test']['id'].tolist())
            self.assertAlmostEqual(float(np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p)))),d['test']['ce'],places=13)
            self.assertEqual(float(np.mean((p>=.5)!=y)),d['test']['error']);self.assertAlmostEqual(float(np.mean((p-y)**2)),d['test']['brier'],places=14)
    def test_augmentation_semantics(self):
        x=np.array([[.2,.7],[-.4,-.9],[.3,-.6]])
        valid,chosen=e.augment(x,'valid',np.random.default_rng(7));bad,chosen2=e.augment(x,'invalid',np.random.default_rng(7))
        np.testing.assert_array_equal(chosen,chosen2);np.testing.assert_array_equal(valid[:,0],x[:,0]);np.testing.assert_array_equal((bad[:,0]>0)!=(x[:,0]>0),chosen)
        np.testing.assert_array_equal(x,[[.2,.7],[-.4,-.9],[.3,-.6]])
    def test_mode_and_label_smoothing(self):
        r=e.mode_probe();a,b,c,d=r['dropout_modes']
        self.assertNotEqual(a['first'],a['second']);self.assertEqual(a['first'],a['input_gradient']);self.assertFalse(b['requires_grad'])
        self.assertEqual(c['first'],c['second']);self.assertTrue(c['requires_grad']);self.assertEqual(c['input_gradient'],np.ones((2,6)).tolist())
        # Identity may return the original requires_grad tensor even in no_grad.
        self.assertTrue(d['requires_grad']);self.assertFalse(any(row['has_running_buffers'] for row in r['dropout_modes']))
        self.assertAlmostEqual(r['label_smoothing_framework']['manual_ce'],r['label_smoothing_framework']['torch_ce'],places=15)
    def test_no_test_argument_in_training_and_eval_restores_mode(self):
        self.assertEqual(list(inspect.signature(e.train_one).parameters),['train','validation','config','seed','epochs','patience'])
        a=e.Network();a.train();before=[v.detach().clone() for v in a.params];e.eval_metrics(a,self.data['train']['x'],self.data['train']['y']);self.assertTrue(a.training)
        for x,y in zip(before,a.params):torch.testing.assert_close(x,y,atol=0,rtol=0)
    def test_reject_invalid_input_values(self):
        for x,y in [([[True,0]],[1]),([['1',0]],[1]),([[float('nan'),0]],[1]),([[1,2]],[[1]]),([[1,2]],[.2]),([],[])]:
            with self.assertRaises(ValueError):e.validate_batch(x,y)
        for kw in ({'seed':True},{'p':1},{'p':float('nan')},{'width':0}):
            with self.assertRaises(ValueError):e.Network(**kw)
        for kw in ({'q':0},{'lam':-.1},{'mask':np.ones((1,2))},{'mask':np.ones((2,2))*.5}):
            with self.assertRaises(ValueError):e.hand_forward(**kw)
    def test_reject_before_output_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'output';out.mkdir();sentinel=out/'calculations.json';sentinel.write_text('KEEP')
            for epochs in (0,True,1.2):
                with self.assertRaises(ValueError):e.run(out,epochs=epochs)
                self.assertEqual(sentinel.read_text(),'KEEP')
            src=root/'data';shutil.copytree(e.BASE/'data',src);f=src/'test.csv';f.write_text(f.read_text().replace('0.','0.9',1))
            with self.assertRaises(ValueError):e.run(out,src)
            self.assertEqual(sentinel.read_text(),'KEEP');self.assertEqual(len(list(out.iterdir())),1)
            new=root/'absent'
            with self.assertRaises(FileNotFoundError):e.run(new,root/'missing')
            self.assertFalse(new.exists())
    def test_cli_invalid_in_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'
            r=subprocess.run([sys.executable,'-O',str(e.BASE/'experiment.py'),'--output',str(out),'--epochs','0'],cwd=temp,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse(out.exists())
    def test_reject_corrupt_figures_before_write(self):
        if not (e.BASE/'make_figures.py').exists():self.fail('figure script missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'outputs';shutil.copytree(e.BASE/'outputs',src);(src/'decision-grid.json').write_text('{"bad":true}')
            dest=root/'figs';dest.mkdir();sentinel=dest/'01_weight_penalty.png';sentinel.write_bytes(b'KEEP')
            r=subprocess.run([sys.executable,'-O',str(e.BASE/'make_figures.py'),'--source',str(src),'--destination',str(dest)],cwd=temp,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertEqual(sentinel.read_bytes(),b'KEEP');self.assertEqual(len(list(dest.iterdir())),1)

if __name__=='__main__':unittest.main(verbosity=2)
