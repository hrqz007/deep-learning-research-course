"""Independent reference calculations and failure checks for unit 033."""
import copy, itertools, tempfile, unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
import experiment as e


def fraction_trace(hand):
    X=[[F(str(v)) for v in row] for row in hand['X']];y=[F(str(row[0])) for row in hand['y']]
    t=[F(str(v)) for v in hand['theta']];mu=F(str(hand['momentum']));buffer=[F(0)]*9;out=[]
    def one(t):
        Z=[[sum(X[i][a]*t[a*2+b] for a in range(2))+t[4+b] for b in range(2)] for i in range(len(X))]
        H=[[max(F(0),v) for v in row] for row in Z];P=[sum(H[i][b]*t[6+b] for b in range(2))+t[8] for i in range(len(X))]
        R=[P[i]-y[i] for i in range(len(X))];dP=[r/len(X) for r in R];dH=[[d*t[6+j] for j in range(2)] for d in dP];dZ=[[dH[i][j] if Z[i][j]>0 else F(0) for j in range(2)] for i in range(len(X))]
        C=[[X[i][a]*dZ[i][b] for a in range(2) for b in range(2)]+dZ[i]+[dP[i]*v for v in H[i]]+[dP[i]] for i in range(len(X))]
        G=[sum(row[j] for row in C) for j in range(9)];loss=sum(r*r for r in R)/(2*len(X))
        return Z,H,P,R,dP,dH,dZ,C,G,loss
    for lr in hand['learning_rates']:
        vals=one(t);buffer=[mu*b+g for b,g in zip(buffer,vals[8])];t=[v-F(str(lr))*b for v,b in zip(t,buffer)]
        out.append((vals,buffer.copy(),t.copy(),one(t)))
    return out


class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e.configure_cpu();cls.cfg,cls.hand,cls.X,cls.y=e.read_inputs()

    def test_01_complete_fraction_trace(self):
        for (ref,buf,t,nxt),got in zip(fraction_trace(self.hand),e.trace_steps(self.hand)['steps']):
            for name,index in [('Z',0),('H',1),('P',2),('residual',3),('per_sample_parameter_contribution',7),('gradient',8)]:
                arr=np.array(ref[index],float)
                if name in ('P','residual'):arr=arr[:,None]
                np.testing.assert_allclose(got[name],arr,rtol=0,atol=3e-16)
            for name,index in [('dL_dP',4),('dL_dH',5),('dL_dZ',6)]:
                arr=np.array(ref[index],float)
                if name=='dL_dP':arr=arr[:,None]
                np.testing.assert_allclose(got['chain'][name],arr,rtol=0,atol=3e-16)
            np.testing.assert_allclose(got['buffer_after'],np.array(buf,float),atol=3e-16,rtol=0)
            np.testing.assert_allclose(got['theta_after'],np.array(t,float),atol=3e-16,rtol=0)
            self.assertAlmostEqual(got['loss'],float(ref[9]),places=15);self.assertAlmostEqual(got['next_forward']['loss'],float(nxt[9]),places=15)

    def test_02_random_fraction_networks(self):
        rng=np.random.default_rng(9033)
        for n in range(1,7):
            for _ in range(10):
                h={'X':(rng.integers(-5,6,(n,2))/5).tolist(),'y':(rng.integers(-5,6,(n,1))/5).tolist(),'theta':(rng.integers(-5,6,9)/5).tolist(),'momentum':.8,'learning_rates':[.01]}
                ref=fraction_trace(h)[0][0]
                if np.min(np.abs(np.array(ref[0],float)))<1e-10:continue
                got=e.trace_steps(h)['steps'][0]
                np.testing.assert_allclose(got['autograd_gradient'],np.array(ref[8],float),rtol=1e-12,atol=1e-14)

    def test_03_all_parameters_finite_difference(self):
        for step in e.trace_steps(self.hand)['steps']:
            theta=np.array(step['theta']);g=[]
            for j in range(9):
                a=theta.copy();b=theta.copy();a[j]+=1e-6;b[j]-=1e-6
                g.append((e.manual_backward(a,self.hand['X'],self.hand['y'])['loss']-e.manual_backward(b,self.hand['X'],self.hand['y'])['loss'])/2e-6)
            np.testing.assert_allclose(g,step['gradient'],rtol=0,atol=4e-11)

    def test_04_fraction_momentum_sequences(self):
        rng=np.random.default_rng(33)
        for _ in range(100):
            mu=F(int(rng.integers(0,10)),10);a=F(1,7);b=F(0)
            p=torch.nn.Parameter(torch.tensor([float(a)],dtype=torch.float64));o=torch.optim.SGD([p],lr=.1,momentum=float(mu))
            for _ in range(10):
                g=F(int(rng.integers(-9,10)),10);lr=F(int(rng.integers(1,10)),100)
                b=mu*b+g;a-=lr*b;p.grad=torch.tensor([float(g)],dtype=torch.float64);o.param_groups[0]['lr']=float(lr);o.step()
                self.assertAlmostEqual(float(p.detach()[0]),float(a),places=14)

    def test_05_ema_weight_and_variance(self):
        mu=.8;n=10;values=[]
        for signs in itertools.product([-1.,1.],repeat=n):
            m=0
            for g in signs:m=mu*m+(1-mu)*g
            values.append(m);self.assertAlmostEqual(m,(1-mu)*sum(mu**(n-1-j)*signs[j] for j in range(n)),places=14)
        self.assertAlmostEqual(np.var(values),(1-mu)**2*(1-mu**(2*n))/(1-mu**2),places=14)

    def test_06_quadratic_independent_recurrence(self):
        for mu in [0,.8]:
            path=e.quadratic_run(momentum=mu);previous=np.array([3.,1.]);current=previous.copy()
            for row in path[1:]:
                nxt=(1+mu-.045*np.array([1.,40.]))*current-mu*previous
                np.testing.assert_allclose([row['x'],row['y']],nxt,rtol=1e-11,atol=2e-14);previous,current=current,nxt
        self.assertGreater(e.quadratic_run(steps=50,lr=.06,momentum=0)[-1]['loss'],24.5)

    def test_07_schedule_endpoints_units(self):
        for total in [2,3,20,72]:
            for w in range(total):
                vals=[e.schedule_lr(k,total,.06,.006,w) for k in range(total+3)]
                self.assertEqual(vals[total-1],.006);self.assertEqual(vals[-1],.006)
                if w:self.assertAlmostEqual(vals[w-1],.06);self.assertAlmostEqual(vals[0],.06/w)
                else:self.assertAlmostEqual(vals[0],.06)
                self.assertTrue(all(a<=b+1e-15 for a,b in zip(vals[:w],vals[1:w])))
                self.assertTrue(all(a>=b-1e-15 for a,b in zip(vals[max(w-1,0):],vals[max(w,1):])))
        self.assertEqual(e.schedule_lr(72,72,.06,.006,12),.006);self.assertGreater(e.schedule_lr(36,72,.06,.006,12),.006)

    def test_08_actual_scheduler_and_initialization(self):
        p=torch.nn.Parameter(torch.zeros(9,dtype=torch.float64));o,s=e.make_optimizer(p,self.cfg,'momentum_warmup_cosine',72)
        self.assertEqual(s.last_epoch,0);self.assertAlmostEqual(o.param_groups[0]['lr'],.005)
        for k in range(72):
            self.assertAlmostEqual(o.param_groups[0]['lr'],e.schedule_lr(k,72,.06,.006,12),places=16)
            p.grad=torch.zeros_like(p);o.step();s.step()
        self.assertEqual(s.last_epoch,72)

    def test_09_full_numpy_training(self):
        for mode in e.MODES:
            run=e.train_network(self.cfg,self.X,self.y,mode)['payload'];t=np.array(self.cfg['initial_theta']);buf=np.zeros(9);k=0
            for order in run['orders']:
                for start in range(0,48,8):
                    ids=order[start:start+8];grad=np.array(e.manual_backward(t,self.X[ids],self.y[ids])['gradient'])
                    buf=(0 if mode=='sgd_fixed' else .8)*buf+grad;t-=e.rate_for_mode(k,self.cfg,mode,72)*buf;k+=1
                    self.assertAlmostEqual(e.manual_backward(t,self.X,self.y)['loss'],run['history'][k]['loss'],places=14)
            np.testing.assert_allclose(t,run['theta'],rtol=0,atol=2e-15)

    def test_10_accumulation_is_same_gradient(self):
        for start in range(0,48,8):
            p=torch.tensor(self.cfg['initial_theta'],dtype=torch.float64,requires_grad=True)
            X=torch.tensor(self.X[start:start+8],dtype=torch.float64);y=torch.tensor(self.y[start:start+8],dtype=torch.float64)
            ((e.forward(p,X)[2]-y)**2).sum().div(16).backward();g=p.grad.clone();p.grad=None
            for i in [0,4]:((e.forward(p,X[i:i+4])[2]-y[i:i+4])**2).sum().div(16).backward()
            torch.testing.assert_close(g,p.grad,rtol=0,atol=1e-16)

    def test_11_resume_across_seeds_cuts(self):
        for seed in [0,33,918]:
            cfg=copy.deepcopy(self.cfg);cfg['seed']=seed
            for mode in e.MODES:
                full=e.train_network(cfg,self.X,self.y,mode)
                for cut in [1,4,11]:
                    c=e.train_network(cfg,self.X,self.y,mode,stop_epoch=cut);r=e.train_network(cfg,self.X,self.y,mode,checkpoint=c)
                    self.assertEqual(e.canonical(full),e.canonical(r))

    def test_12_checkpoint_rejects_bad_states(self):
        ck=e.train_network(self.cfg,self.X,self.y,stop_epoch=4)
        changes=[lambda p:p.pop('scheduler'),lambda p:p.update(completed_epochs=12),lambda p:p.update(completed_epochs=True),lambda p:p.update(mode='sgd_fixed'),lambda p:p.update(torch_version='0'),lambda p:p.update(theta=[1.]*8),lambda p:p.update(momentum_buffer=[True]*9),lambda p:p['scheduler'].update(last_epoch=25),lambda p:p['scheduler'].update(_step_count=24),lambda p:p['scheduler'].update(base_lrs=[.6]),lambda p:p['optimizer_group'].update(lr=.06),lambda p:p['optimizer_group'].update(momentum=.9),lambda p:p['orders'][0].__setitem__(0,p['orders'][0][1]),lambda p:p.update(shuffle_rng=[0]*3),lambda p:p.update(data_sha256='bad'),lambda p:p['config'].update(accumulation=1),lambda p:p['history'].pop(),lambda p:p['history'][-1].update(lr_next=.07)]
        for change in changes:
            p=copy.deepcopy(ck['payload']);change(p)
            with self.assertRaises(ValueError):e.validate_checkpoint(e.seal(p),self.cfg,self.X,self.y,'momentum_warmup_cosine')
        bad=copy.deepcopy(ck);bad['payload']['theta'][0]+=.01
        with self.assertRaises(ValueError):e.validate_checkpoint(bad,self.cfg,self.X,self.y,'momentum_warmup_cosine')

    def test_13_public_api_bad_inputs(self):
        for X,y in [([[True,False]],[[1.]]),([[1,2]],[1.]),([[1,2]],[[float('nan')]]),([],[]),([[1,2,3]],[[1]]),([[1,200]],[[1]])]:
            with self.assertRaises(ValueError):e.validate_batch(X,y)
        for args in [(True,72,.06,.006,12),(0,1,.06,.0,0),(0,72,0,.0,12),(0,72,.06,.07,12),(0,72,.06,.006,72),(-1,72,.06,.006,12)]:
            with self.assertRaises(ValueError):e.schedule_lr(*args)
        for key,value in [('epochs',True),('accumulation',5),('microbatch_size',0),('momentum',1.),('peak_lr',float('nan')),('seed',-1)]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(ValueError):e.train_network(cfg,self.X,self.y)

    def test_14_resume_negative_controls(self):
        full=e.train_network(self.cfg,self.X,self.y);ck=e.train_network(self.cfg,self.X,self.y,stop_epoch=4)
        for omit in ['optimizer','scheduler','rng']:
            bad=e.train_network(self.cfg,self.X,self.y,checkpoint=ck,_omit=omit)
            self.assertNotEqual(bad,full);self.assertGreater(np.max(np.abs(np.array(bad['payload']['theta'])-full['payload']['theta'])),1e-5)
        with self.assertRaises(ValueError):e.train_network(self.cfg,self.X,self.y,checkpoint=full)
        with self.assertRaises(ValueError):e.train_network(self.cfg,self.X,self.y,stop_epoch=4,checkpoint=ck)

    def test_15_ownership_and_serialization(self):
        cfg=copy.deepcopy(self.cfg);X=self.X.copy();y=self.y.copy();ck=e.train_network(cfg,X,y,stop_epoch=4);old=copy.deepcopy(ck)
        e.train_network(cfg,X,y,checkpoint=ck);self.assertEqual(ck,old);self.assertEqual(cfg,self.cfg);np.testing.assert_array_equal(X,self.X);np.testing.assert_array_equal(y,self.y)
        with self.assertRaises(ValueError):e.canonical({'x':float('nan')})
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);target=d/'untouched';target.write_bytes(b'old');out=d/'out';out.mkdir();(out/'a.json').symlink_to(target)
            with self.assertRaises(ValueError):e.write_outputs({'a.json':b'new'},out)
            self.assertEqual(target.read_bytes(),b'old')
            with self.assertRaises(ValueError):e.write_outputs({'a.json':b'x','../oops':b'z'},d/'absent')
            self.assertFalse((d/'absent').exists())

if __name__=='__main__':unittest.main()
