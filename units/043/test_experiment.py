"""Meaningful numerical contracts, runnable under both Python and Python -O."""
from pathlib import Path
import hashlib,json,tempfile,unittest
import numpy as np
import torch
from experiment import conv_forward as forward,conv_backward as backward,hand_ledger,run,array
import generate_data
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)

def close(a,b,tol=2e-11):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def numeric(v):
    if isinstance(v,dict) and 'float' in v:return v['float']
    return [numeric(x) for x in v]

class Contracts(unittest.TestCase):
    def test_forward_backward_framework_configurations(self):
        rng=np.random.default_rng(4331)
        cases=[((2,2,5,6),(3,2,2,3),1,0,1),((1,2,6,7),(2,2,3,2),(2,1),(1,2),(1,2)),((1,1,3,4),(1,1,1,1),2,0,1),((1,1,4,4),(1,1,2,2),1,0,2)]
        for xs,ks,s,p,d in cases:
            x=rng.normal(size=xs);k=rng.normal(size=ks);b=rng.normal(size=ks[0]);kw=dict(stride=s,padding=p,dilation=d)
            y=forward(x,k,b,**kw);u=rng.normal(size=y.shape)
            xt=torch.tensor(x,requires_grad=True);kt=torch.tensor(k,requires_grad=True);bt=torch.tensor(b,requires_grad=True)
            z=torch.nn.functional.conv2d(xt,kt,bt,**kw);(z*torch.tensor(u)).sum().backward()
            close(y,z.detach().numpy())
            for actual,expected in zip(backward(x,k,b,u,**kw),(xt.grad,kt.grad,bt.grad)):close(actual,expected.numpy())
    def test_finite_difference_all_coordinates(self):
        rng=np.random.default_rng(4332);x=rng.normal(size=(1,1,3,4));k=rng.normal(size=(2,1,2,2));b=rng.normal(size=2)
        kw={'stride':(2,1),'padding':(1,0),'dilation':(1,2)};u=rng.normal(size=forward(x,k,b,**kw).shape)
        grads=backward(x,k,b,u,**kw)
        for a,g in zip((x,k,b),grads):
            for idx in np.ndindex(a.shape):
                old=a[idx];a[idx]=old+1e-6;plus=np.sum(forward(x,k,b,**kw)*u);a[idx]=old-1e-6;minus=np.sum(forward(x,k,b,**kw)*u);a[idx]=old
                self.assertAlmostEqual(float(g[idx]),float((plus-minus)/2e-6),delta=2e-8)
    def test_explicit_linear_operator_adjoint(self):
        # Each basis response is independently computed via torch, not the tested forward.
        rng=np.random.default_rng(4333);shape=(1,1,4,5);k=rng.normal(size=(2,1,2,2));b=np.zeros(2);kw={'stride':(2,1),'padding':1,'dilation':2}
        cols=[]
        for j in range(np.prod(shape)):
            basis=np.zeros(shape);basis.flat[j]=1
            cols.append(torch.nn.functional.conv2d(torch.tensor(basis),torch.tensor(k),torch.tensor(b),**kw).numpy().ravel())
        A=np.array(cols).T;x=rng.normal(size=shape);y=forward(x,k,b,**kw);u=rng.normal(size=y.shape);dx,_,_=backward(x,k,b,u,**kw)
        close(y.ravel(),A@x.ravel());close(dx.ravel(),A.T@u.ravel());self.assertAlmostEqual(float(y.ravel()@u.ravel()),float(x.ravel()@dx.ravel()),places=10)
    def test_exact_hand_every_state_and_gradient(self):
        x=np.array([[[[1,0,2],[0,1,0],[2,0,1]]],[[[0,1,0],[1,0,1],[0,1,0]]]],float);targets=np.array([.5,1.])
        for state in hand_ledger():
            theta=np.array(numeric(state['theta']));xt=torch.tensor(x,requires_grad=True);kt=torch.tensor(theta[:4].reshape(1,1,2,2),requires_grad=True);bt=torch.tensor(theta[4:],requires_grad=True)
            z=torch.nn.functional.conv2d(xt,kt,bt);h=torch.relu(z);pred=h.mean((1,2,3));loss=((pred-torch.tensor(targets))**2).mean()/2;loss.backward()
            close(z.detach().numpy().reshape(2,4),[numeric(s['z']) for s in state['samples']]);close(pred.detach().numpy(),[s['pred']['float'] for s in state['samples']]);self.assertAlmostEqual(float(loss.detach()),state['loss']['float'],places=14)
            close(np.r_[kt.grad.numpy().ravel(),bt.grad.numpy()],numeric(state['gradient']));close(xt.grad.numpy()[:,0],[numeric(s['dx']) for s in state['samples']])
    def test_shared_gradient_accumulates_samples(self):
        rng=np.random.default_rng(4334);x=rng.normal(size=(3,2,4,4));k=rng.normal(size=(2,2,2,2));b=np.ones(2);y=forward(x,k,b);u=rng.normal(size=y.shape)
        dx,dk,db=backward(x,k,b,u);g=[backward(x[i:i+1],k,b,u[i:i+1]) for i in range(3)]
        close(dk,sum(a[1] for a in g));close(db,sum(a[2] for a in g));close(dx,np.concatenate([a[0] for a in g]))
    def test_noncontiguous_no_input_mutation(self):
        x=np.arange(48.).reshape(1,2,4,6)[:,:,:,::2];k=np.ones((1,2,2,2));b=np.zeros(1);old=x.copy();x.flags.writeable=False
        y=forward(x,k,b);backward(x,k,b,np.ones_like(y));close(x,old)
    def test_rejections(self):
        x=np.ones((1,1,3,3));k=np.ones((1,1,2,2));b=np.zeros(1)
        for kw in [{'stride':0},{'stride':True},{'padding':-1},{'dilation':1.5},{'padding':10**9},{'stride':(1,)},{'dilation':(1,False)}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):forward(x,k,b,**kw)
        for bad in [np.ones((3,3)),np.zeros((0,1,3,3)),np.full_like(x,np.nan),x.astype(complex),x.astype(bool)]:
            with self.subTest(bad=bad.shape),self.assertRaises(ValueError):forward(bad,k,b)
        with self.assertRaises(ValueError):forward(x,np.ones((1,2,2,2)),b)
        with self.assertRaises(ValueError):forward(x,k,np.zeros(2))
        with self.assertRaises(ValueError):forward(x,np.ones((1,1,4,4)),b)
        with self.assertRaises(ValueError):backward(x,k,b,np.ones((1,1,2,1)))
        with self.assertRaises(FloatingPointError):forward(x*1e308,k*1e308,b)
        with self.assertRaises(TypeError):forward(x,k,b,groups=2)
    def test_fixed_data_regeneration(self):
        generated=generate_data.generate();stored=json.loads((ROOT/'data/experiment_data.json').read_text());self.assertEqual(generated,stored)
    def test_full_experiment_and_retained_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=run(tmp);self.assertTrue((Path(tmp)/'results.json').is_file());self.assertEqual(json.loads((Path(tmp)/'results.json').read_text()),r)
        self.assertLess(r['learning']['torch_max_abs_gap'],1e-12);self.assertEqual(r['learning']['least_squares_rank'],10)
        self.assertGreater(r['learning']['train_half_mse'],r['learning']['least_squares_train_half_mse'])
        rows={v['case']:v for v in r['symmetry']['rows']}
        for name in ['zero_stride1_interior','circular_stride1','circular_stride2_shift2','circular_relu']:self.assertLess(rows[name]['max_abs'],1e-12)
        self.assertGreater(rows['zero_stride1_full']['max_abs'],1.)
        self.assertGreater(r['hand'][1]['samples'][0]['loss']['float'],r['hand'][0]['samples'][0]['loss']['float'])

if __name__=='__main__':unittest.main(verbosity=2)
