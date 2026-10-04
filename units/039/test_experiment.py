"""Independent numeric checks and failure contracts for unit 039."""
import csv,hashlib,json,math,tempfile,unittest
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
import torch
import experiment as e

ROOT=Path(__file__).resolve().parent
ERRORS={}

def scalar_trace(theta):
    """Decimal, scalar loops; no call to production forward/backward."""
    with localcontext() as ctx:
        ctx.prec=55;d=lambda x:Decimal(str(x));t=list(map(d,theta));xs=[[d(1),d(2)],[d(-1),d(1)]];ys=[d(1),d(0)];weights=[d(3),d(1)];paths=[];risk=d(0)
        for x,y,a in zip(xs,ys,weights):
            z=[sum(x[k]*t[2*k+j] for k in range(2))+t[4+j] for j in range(2)];h=[max(q,d(0)) for q in z];s=sum(h[j]*t[6+j] for j in range(2))+t[8];p=1/(1+(-s).exp());loss=(1+((-s) if y else s).exp()).ln();risk+=a*loss/4;r=a*(p-y)/4
            dz=[r*t[6+j] if z[j]>0 else d(0) for j in range(2)]
            paths.append([x[k]*dz[j] for k in range(2) for j in range(2)]+dz+[r*q for q in h]+[r])
        return float(risk),np.array([[float(q) for q in row] for row in paths])

def numpy_train(x,y,method,seed,epochs):
    rng=np.random.default_rng(seed);w=rng.normal(0,.45,(2,6));b=np.zeros(6);u=rng.normal(0,.35,6);c=np.array(0.);rng=np.random.default_rng(seed+39000);a=np.where(y==1,.5/y.mean(),.5/(1-y.mean()));history=[]
    for _ in range(epochs):
        ix=rng.choice(len(y),len(y),replace=True,p=a/a.sum()) if method=='resampled' else np.arange(len(y));xx=x[ix];yy=y[ix];h=np.tanh(xx@w+b);s=h@u+c;p=1/(1+np.exp(-s));weights=a[ix] if method=='weighted' else np.ones(len(y))
        history.append(float(np.mean(weights*np.logaddexp(0,np.where(yy==1,-s,s)))))
        ds=weights*(p-yy)/len(y);dh=ds[:,None]*u;dz=dh*(1-h*h);dw=xx.T@dz;db=dz.sum(axis=0);du=h.T@ds;dc=ds.sum()
        w=w-.08*dw;b=b-.08*db;u=u-.08*du;c=c-.08*dc
    return [w,b,u,c],history

class Test039(unittest.TestCase):
    def test_hand_all_paths_decimal_autograd_fd(self):
        first=e.hand_trace();allmax=[]
        for theta in [first['theta'],first['theta_next']]:
            a=e.hand_trace(theta);loss,paths=scalar_trace(theta);np.testing.assert_allclose(a['paths'],paths,atol=3e-16,rtol=1e-14);self.assertAlmostEqual(a['risk'],loss,14)
            t=torch.tensor(theta,dtype=torch.float64,requires_grad=True);x=torch.tensor(a['x'],dtype=torch.float64);s=torch.relu(x@t[:4].reshape(2,2)+t[4:6])@t[6:8]+t[8];ell=torch.nn.functional.binary_cross_entropy_with_logits(s,torch.tensor(a['y'],dtype=torch.float64),reduction='none');j=(ell*torch.tensor([3.,1.],dtype=torch.float64)).sum()/4;j.backward();np.testing.assert_allclose(t.grad.numpy(),a['gradient'],atol=3e-16,rtol=1e-13)
            fd=[]
            for k in range(9):
                plus=np.array(theta);minus=np.array(theta);plus[k]+=1e-6;minus[k]-=1e-6;fd.append((scalar_trace(plus)[0]-scalar_trace(minus)[0])/2e-6)
            allmax.append(float(np.max(np.abs(np.array(fd)-a['gradient']))));np.testing.assert_allclose(fd,a['gradient'],atol=1e-9,rtol=1e-8)
            np.testing.assert_allclose(a['theta_next'],np.array(theta)-.1*np.array(a['gradient']),atol=1e-16)
        ERRORS['hand_finite_difference_max']=max(allmax)
    def test_stability_and_reductions(self):
        np.testing.assert_allclose(e.bce([1000.,-1000.,1000.,-1000.],[1,0,0,1]),[0,0,1000,1000])
        z=[.42,-.3];y=[1,0];w=[3,1];self.assertAlmostEqual(e.weighted_risk(z,y,w,'count'),2*e.weighted_risk(z,y,w,'weights'))
        zz=torch.tensor(z,dtype=torch.float64);yy=torch.tensor(y,dtype=torch.float64)
        a=torch.nn.functional.binary_cross_entropy_with_logits(zz,yy,pos_weight=torch.tensor(3.),reduction='mean')
        b=torch.nn.functional.cross_entropy(torch.stack([torch.zeros(2),zz],1),torch.tensor(y),weight=torch.tensor([1.,3.],dtype=torch.float64))
        self.assertAlmostEqual(float(a),2*float(b),13)
    def test_extreme_weights_stable_or_explicitly_rejected(self):
        self.assertAlmostEqual(e.weighted_risk([0.,0.],[0,1],[1e308,1e308],'weights'),math.log(2),15)
        self.assertAlmostEqual(e.weighted_risk([0.,0.],[0,1],[1e-308,1e-308],'weights'),math.log(2),15)
        got=e.weighted_risk([2.,-1.],[0,1],[1e308,5e307],'weights')
        expected=(2*math.log1p(math.exp(2))+math.log1p(math.exp(1)))/3
        self.assertAlmostEqual(got,expected,14)
        self.assertTrue(math.isfinite(e.weighted_risk([0.,0.],[0,1],[1e308,1e308],'count')))
        with self.assertRaises(ValueError):e.weighted_risk([1000.,1000.],[0,0],[1e308,1e308],'count')
    def test_float64_conversion_checked_after_cast(self):
        with np.errstate(over='ignore',invalid='ignore'):
            extended=np.array([np.longdouble('1e400')])
        with self.assertRaises(ValueError):e.array(extended,'weights')
        self.assertEqual(e.array(np.array([np.longdouble('1e200')]),'x').dtype,np.float64)
    def test_hand_local_probability_range_and_serialization(self):
        original=e.hand_trace()['theta']
        for bias in (1000.,-1000.):
            t=list(original);t[-1]=bias
            with self.assertRaises(ValueError):e.hand_trace(t)
        for bias in (29.,-29.):
            t=list(original);t[-1]=bias;out=e.hand_trace(t)
            json.dumps(out,allow_nan=False)
            self.assertTrue(np.isfinite(out['local_dloss_dp']).all())
        t=list(original);t[0]=1e7
        with self.assertRaises(ValueError):e.hand_trace(t)
    def test_temperature_saturated_mirror_and_flat_status(self):
        for logit,label in [(1000.,1),(-1000.,0),(1e6,1),(-1e6,0)]:
            r=e.fit_temperature([logit],[label]);self.assertEqual(r['beta'],20.);self.assertEqual(r['status'],'upper_bound')
        for logit,label in [(1000.,0),(-1000.,1),(1e6,0),(-1e6,1)]:
            r=e.fit_temperature([logit],[label]);self.assertEqual(r['beta'],.05);self.assertEqual(r['status'],'lower_bound')
        tiny=np.nextafter(0.,1.)
        r=e.fit_temperature([tiny,tiny],[1,0]);self.assertEqual(r['status'],'numerically_flat');self.assertEqual(r['beta'],1.)
        r=e.fit_temperature([40.,-1.],[1,0]);self.assertEqual(r['status'],'upper_bound')
    def test_weighted_population_and_double_weighting(self):
        eta=.2;a0=1.;a1=4.;q=a1*eta/(a1*eta+a0*(1-eta));self.assertEqual(q,.5)
        self.assertAlmostEqual(e.sigmoid(np.array(math.log(q/(1-q))-math.log(a1/a0))).item(),eta)
        losses=np.array([.1,.6,1.3]);a=np.array([1.,2.,3.]);q=a/a.sum();self.assertAlmostEqual(float(q@losses),float((a*losses).sum()/a.sum()))
        # Enumerate all two-draw ordered batches and their probabilities.
        fixed=selfnorm=0.
        for i in range(3):
            for j in range(3):
                fixed+=(a[i]*losses[i]+a[j]*losses[j])/(2*a.mean())/9
                selfnorm+=(a[i]*losses[i]+a[j]*losses[j])/(a[i]+a[j])/9
        self.assertAlmostEqual(fixed,float(q@losses));self.assertGreater(abs(selfnorm-fixed),.01)
    def test_all_2160_updates_independent_numpy(self):
        data=e.load_split(ROOT/'data','train');frozen=json.loads((ROOT/'outputs/frozen-plan.json').read_text());rows=list(csv.DictReader((ROOT/'outputs/training-history.csv').read_text().splitlines()));maxp=0.;maxloss=0.
        for r in frozen['runs']:
            ps,loss=numpy_train(data['x'],data['y'],r['method'],r['seed'],240)
            for a,b in zip(ps,r['params']): maxp=max(maxp,float(np.max(np.abs(a-b))))
            expected=[float(h['objective_before']) for h in rows if h['method']==r['method'] and int(h['seed'])==r['seed']]
            maxloss=max(maxloss,float(np.max(np.abs(np.array(loss)-expected))))
        self.assertLess(maxp,2e-12);self.assertLess(maxloss,2e-12);ERRORS.update({'training_final_parameter_max':maxp,'training_loss_max':maxloss})
    def test_main_all_parameter_gradients(self):
        data=e.load_split(ROOT/'data','train');x=data['x'][:7];y=data['y'][:7];ps=e.initial(11);w,b,u,c=ps;h=np.tanh(x@w+b);s=h@u+c;r=(1/(1+np.exp(-s))-y)/len(y);dz=(r[:,None]*u)*(1-h*h);gs=[x.T@dz,dz.sum(0),h.T@r,np.array(r.sum())]
        ts=[torch.tensor(a,dtype=torch.float64,requires_grad=True) for a in ps];loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.tanh(torch.tensor(x)@ts[0]+ts[1])@ts[2]+ts[3],torch.tensor(y));loss.backward()
        for a,b in zip(gs,ts):np.testing.assert_allclose(a,b.grad.numpy(),atol=2e-15)
    def test_temperature_independent_minimum(self):
        data=e.load_split(ROOT/'data','validation');m=data['role']=='calibration';frozen=json.loads((ROOT/'outputs/frozen-plan.json').read_text());errors=[]
        for r in frozen['runs']:
            z=e.predict(r['params'],data['x'])[m]-r['prior_logit_subtract'];y=data['y'][m]
            # Independent golden-section minimizer, objective via scalar math.
            def f(beta):return math.fsum(math.log1p(math.exp(-abs(beta*s)))+max(beta*s,0)-yi*beta*s for s,yi in zip(z,y))/len(y)
            lo,hi=.05,20.;gold=(math.sqrt(5)-1)/2
            for _ in range(120):
                l=hi-gold*(hi-lo);rr=lo+gold*(hi-lo)
                if f(l)<f(rr):hi=rr
                else:lo=l
            beta=(lo+hi)/2;stored=r['temperature']['beta'];errors.append(abs(beta-stored));self.assertLess(abs(f(beta)-f(stored)),1e-13);self.assertLessEqual(r['temperature']['nll_after'],r['temperature']['nll_before']+1e-14)
            self.assertTrue(np.array_equal(z>=0,stored*z>=0));self.assertTrue(np.array_equal(np.argsort(z),np.argsort(stored*z)))
        ERRORS['temperature_golden_beta_max']=max(errors)
        self.assertEqual(e.fit_temperature([0.,0.],[0,1])['status'],'flat')
        self.assertTrue(math.isfinite(e.fit_temperature([1e6,-1e6],[1,0])['nll_after']))
        self.assertEqual(e.fit_temperature([-1.,1.],[0,1])['status'],'upper_bound')
        self.assertEqual(e.fit_temperature([-1.,1.],[1,0])['status'],'lower_bound')
    def test_all_metrics_independent_records(self):
        with (ROOT/'outputs/predictions.csv').open() as f:rows=list(csv.DictReader(f))
        groups={}
        for r in rows:
            k=(r['method'],r['seed'],r['view']);groups.setdefault(k,[]).append(r)
        with (ROOT/'outputs/metrics.csv').open() as f:refs=list(csv.DictReader(f))
        for ref in refs:
            rr=[r for r in groups[(ref['method'],ref['seed'],ref['view'])] if ref['group']=='all' or r['group']==ref['group']];n=len(rr);yy=[int(r['y']) for r in rr];pp=[float(r['probability']) for r in rr];pred=[int(r['predicted']) for r in rr];ss=[float(r['logit']) for r in rr]
            fp=sum(a==1 and b==0 for a,b in zip(pred,yy));fn=sum(a==0 and b==1 for a,b in zip(pred,yy));self.assertEqual(n,int(ref['n']));self.assertEqual(fp,int(ref['fp']));self.assertEqual(fn,int(ref['fn']))
            self.assertAlmostEqual((fp+4*fn)/n,float(ref['cost']),14);self.assertAlmostEqual(sum(a==b for a,b in zip(pred,yy))/n,float(ref['accuracy']),14)
            self.assertAlmostEqual(math.fsum((p-y)**2 for p,y in zip(pp,yy))/n,float(ref['brier']),14)
            nll=math.fsum(math.log1p(math.exp(-abs(s)))+max(s,0)-y*s for s,y in zip(ss,yy))/n;self.assertAlmostEqual(nll,float(ref['nll']),14)
            for bins,col in [(5,'ece5'),(10,'ece'),(20,'ece20')]:
                members=[[] for _ in range(bins)]
                for p,y in zip(pp,yy):members[min(int(p*bins),bins-1)].append((p,y))
                error=math.fsum(abs(math.fsum(p-y for p,y in b)) for b in members)/n;self.assertAlmostEqual(error,float(ref[col]),14)
    def test_threshold_grid_and_ties(self):
        t,rows=e.choose_threshold([0.,0.],[0,0]);self.assertEqual(t,1.);self.assertEqual(len(rows),21)
        r=e.metrics([0.],[1.],.5);self.assertEqual(r['tp'],1)
        self.assertIsNone(e.metrics([-2.,-1.],[0,0])['recall']);self.assertIsNone(e.metrics([-2.,-1.],[0,0])['precision'])
    def test_invalid_inputs_explicit_exceptions(self):
        for z,y in [([],[]),([float('nan')],[0]),([0],[2]),([[0]],[0]),([True],[0]),([0,True],[0,1]),([0,1],[0,True]),(['1'],[0]),([float('inf')],[0]),([1e8],[1]),([1,2],[0])]:
            with self.assertRaises(ValueError):e.bce(z,y)
        for v in [0,-1,True,1.2,'3',float('nan')]:
            with self.assertRaises(ValueError):e.integer(v,'number')
        for t in [-1,2,True,float('nan'),'0.5']:
            with self.assertRaises(ValueError):e.metrics([0],[1],t)
        for a,d in [([0,1],'count'),([-1,1],'weights'),([1],'weights'),([1,1],'bad')]:
            with self.assertRaises(ValueError):e.weighted_risk([0,1],[0,1],a,d)
        with self.assertRaises(ValueError):e.train([[1,2]],[0],'uniform',11)
    def test_failed_input_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            for name in e.DATA_HASHES:(data/name).write_bytes((ROOT/'data'/name).read_bytes())
            out=root/'out';out.mkdir();(out/'sentinel').write_text('keep')
            (data/'validation.csv').write_text('broken')
            with self.assertRaises(ValueError):e.run(data,out)
            self.assertEqual([p.name for p in out.iterdir()],['sentinel'])
    def test_runtime_selection_before_test_labels(self):
        from unittest.mock import patch
        events=[];original_load=e.load_split;original_train=e.train;original_temp=e.fit_temperature;original_choose=e.choose_threshold
        def tracked_load(directory,name):
            if name=='test':
                self.assertEqual(events.count('train'),9);self.assertEqual(events.count('temperature'),18);self.assertEqual(events.count('threshold'),9)
            events.append('load_'+name);return original_load(directory,name)
        def tracked_train(*args,**kwargs):
            result=original_train(*args,**kwargs);events.append('train');return result
        def tracked_temp(*args,**kwargs):
            result=original_temp(*args,**kwargs);events.append('temperature');return result
        def tracked_choose(*args,**kwargs):
            result=original_choose(*args,**kwargs);events.append('threshold');return result
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(e,'load_split',tracked_load),patch.object(e,'train',tracked_train),patch.object(e,'fit_temperature',tracked_temp),patch.object(e,'choose_threshold',tracked_choose):
                e.run(output=directory,epochs=1)
        self.assertEqual(events.count('load_test'),1)

    def test_test_label_isolation(self):
        import inspect
        text=inspect.getsource(e.run);self.assertLess(text.index("frozen="),text.index("test=load_split"));self.assertNotIn("test[",text[:text.index("test=load_split")])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Test039);result=unittest.TextTestRunner(verbosity=2).run(suite);print(json.dumps({'independent_numeric_errors':ERRORS},indent=2));raise SystemExit(0 if result.wasSuccessful() else 1)
