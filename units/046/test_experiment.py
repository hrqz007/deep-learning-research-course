"""Tests remain active under python -O; all comparisons use explicit checks."""
from pathlib import Path
import json,unittest,tempfile
from unittest.mock import patch
import numpy as np
import torch
from reference import *
from experiment import init,forward,load_data
ROOT=Path(__file__).resolve().parent
ERRORS={}

def close(a,b,tol=1e-8):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.allclose(a,b,atol=tol,rtol=tol):raise AssertionError(f'mismatch {a} != {b}')

class Tests(unittest.TestCase):
    def test_iou_geometry(self):
        close(iou([[0,0,2,2]],[[1,0,3,2]]),[[1/3]])
        close(iou([[0,0,2,2]],[[2,0,4,2]]),[[0]])
        close(iou([[0,0,2,2]],[[0,0,2,2]]),[[1]])
        close(iou(np.empty((0,4)),[[0,0,1,1]]),np.empty((0,1)))
        # Independent integer-lattice cell-set area, no min/max formula.
        rng=np.random.default_rng(4699)
        for _ in range(50):
            l,t=rng.integers(-3,3,2);w,h=rng.integers(1,5,2);a=[l,t,l+w,t+h];l,t=rng.integers(-3,3,2);w,h=rng.integers(1,5,2);b=[l,t,l+w,t+h]
            sa={(x,y) for x in range(a[0],a[2]) for y in range(a[1],a[3])};sb={(x,y) for x in range(b[0],b[2]) for y in range(b[1],b[3])};close(iou([a],[b]),[[len(sa&sb)/len(sa|sb)]])
        h=1e-6;s=.7;f=lambda z:iou([[z,0,z+2,2]],[[0,0,2,2]])[0,0];close((f(s+h)-f(s-h))/(2*h),-4/(2+s)**2)
    def test_box_encoding(self):
        anchor=[[0,0,2,2]];target=[[1,0,5,2]];t=encode_boxes(anchor,target);close(t,[[1,0,np.log(2),0]]);close(decode_boxes(anchor,t),target)
        with self.assertRaises(ValueError):decode_boxes(anchor,[[0,0,100,0]])
        with self.assertRaises(ValueError):encode_boxes(anchor,[[0,0,0,1]])
    def test_nms(self):
        b=[[0,0,2,2],[0,0,2,2],[2,0,4,2]]
        self.assertEqual(nms(b,[.9,.9,.8],.5),[0,2]);self.assertEqual(nms(b,[.9,.9,.8],1),[0,1,2]);self.assertEqual(nms(np.empty((0,4)),[],.5),[])
    def test_ap(self):
        gt=[{'image_id':'a','box':[0,0,2,2]},{'image_id':'b','box':[0,0,2,2]}]
        pred=[{'image_id':'a','box':[0,0,2,2],'score':.9},{'image_id':'a','box':[0,0,2,2],'score':.8},{'image_id':'b','box':[0,0,2,2],'score':.7}]
        r=detection_ap(gt,pred);self.assertEqual(r['tp'],[1,0,1]);close(r['ap_all_point'],5/6);close(r['ap_101_point'],(51+50*2/3)/101)
        self.assertIsNone(detection_ap([],pred)['ap_all_point']);close(detection_ap(gt,[])['ap_all_point'],0)
        pred.insert(0,{'image_id':'empty','box':[0,0,2,2],'score':1});close(detection_ap(gt,pred)['ap_all_point'],.5)
        # Same coordinates in another image cannot match; threshold equality counts.
        self.assertEqual(detection_ap(gt[:1],[{'image_id':'b','box':[0,0,2,2],'score':1}])['tp'],[0])
        self.assertEqual(detection_ap(gt[:1],[{'image_id':'a','box':[0,0,1,2],'score':1}],.5)['tp'],[1])
    def test_losses(self):
        r=np.array([-2.,-.5,0,.5,2.]);l,g=smooth_l1(r);close(l,[1.5,.125,0,.125,1.5]);close(g,[-1,-.5,0,.5,1]);close(smooth_l1([1.],1e-300)[0],[1.])
        z=np.array([-1000.,-1.,0.,1.,1000.]);y=np.array([1.,0.,1.,0.,1.])
        for q in [1.,3.]:
            loss,g=bce(z,y,q);zt=torch.tensor(z,requires_grad=True);lt=torch.nn.functional.binary_cross_entropy_with_logits(zt,torch.tensor(y),pos_weight=torch.tensor(q,dtype=zt.dtype));lt.backward();close(loss,lt.detach().numpy());close(g,zt.grad.numpy())
    def test_hand_all_paths(self):
        x=torch.tensor([[0.,1.],[1.,2.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
        for r in hand_ledger():
            w,b=[torch.tensor(t,dtype=torch.float64,requires_grad=True) for t in r['theta']];z=w*x+b;loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y);loss.backward();close(loss.detach().numpy(),r['loss']);close([w.grad.item(),b.grad.item()],r['gradient']);close(x.grad.numpy(),[[p['dL_dx'] for p in s['pixels']] for s in r['samples']]);x.grad=None
            for s in r['samples']:
                for p in s['pixels']:close(p['dL_dp']*p['dp_dz'],p['dL_dz'])
        close(hand_ledger()[0]['gradient'],[7/30,19/120])
    def test_cnn_all_elements(self):
        torch.set_num_threads(1);rng=np.random.default_rng(463);x=rng.normal(.4,.2,(2,1,3,4));y=rng.integers(0,2,x.shape).astype(float);p=init('cnn_bce',11);raw=[v.detach().numpy().copy() for v in p]
        loss,gs,dx,out=cnn_reference(raw,x,y,2.7);xt=torch.tensor(x,requires_grad=True);z=forward(p,xt);lt=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor(y),pos_weight=torch.tensor(2.7,dtype=torch.float64));lt.backward();close(out,z.detach().numpy());close(loss,lt.detach().numpy());close(dx,xt.grad.numpy())
        for a,b in zip(gs,p):close(a,b.grad.numpy())
        eps=1e-6;errs=[]
        for k,a in enumerate(raw):
            for ix in np.ndindex(a.shape):
                old=a[ix];a[ix]=old+eps;lp=cnn_reference(raw,x,y,2.7)[0];a[ix]=old-eps;lm=cnn_reference(raw,x,y,2.7)[0];a[ix]=old;errs.append(abs((lp-lm)/(2*eps)-gs[k][ix]))
        for ix in np.ndindex(x.shape):
            old=x[ix];x[ix]=old+eps;lp=cnn_reference(raw,x,y,2.7)[0];x[ix]=old-eps;lm=cnn_reference(raw,x,y,2.7)[0];x[ix]=old;errs.append(abs((lp-lm)/(2*eps)-dx[ix]))
        ERRORS['cnn_all_parameters_and_inputs_fd_max']=max(errs);self.assertLess(max(errs),2e-8)
    def test_mask_empty_overlap(self):
        p=np.array([[[0,0],[0,0]],[[1,1],[0,0]]]);y=np.array([[[0,0],[0,0]],[[1,0],[1,0]]]);r=mask_metrics(p,y);close(r['per_image_iou'],[1,1/3]);close(r['foreground_iou_micro'],1/3);close(r['foreground_iou_macro_empty1'],2/3);close(r['pixel_accuracy'],.75)
        close(mask_metrics(np.ones((1,2,2)),np.ones((1,2,2)))['mean_class_iou_micro'],1)
        close(mask_metrics(np.zeros((1,2,2)),np.zeros((1,2,2)))['foreground_iou_micro'],1)
        data=load_data();meta=json.loads((ROOT/'data/metadata.json').read_text());ids=[]
        for s,(x,y) in data.items():
            self.assertEqual(x.shape,y.shape);self.assertEqual(x.shape[1:],(1,16,16))
            for i,r in enumerate(meta['splits'][s]['records']):
                expected=np.zeros((16,16),np.uint8)
                for l,t,rr,b in r['boxes']:expected[t:b,l:rr]=1
                close(y[i,0],expected);ids.append(r['id'])
                if r['kind']=='overlap':self.assertLess(expected.sum(),sum((b[2]-b[0])*(b[3]-b[1]) for b in r['boxes']))
        self.assertEqual(len(ids),len(set(ids)))
    def test_actual_run_positive_weight_dtype_and_first_step(self):
        import experiment
        original=torch.nn.functional.binary_cross_entropy_with_logits
        captured=[]
        def capture(logits,targets,*args,**kwargs):
            weight=kwargs.get('pos_weight')
            if weight is not None:captured.append((weight.detach().clone(),logits.dtype,logits.device))
            return original(logits,targets,*args,**kwargs)
        old_dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            with tempfile.TemporaryDirectory(prefix='dl046-entry-regression-') as output:
                with patch.object(experiment,'SEEDS',[11]),patch.object(experiment,'ARMS',['cnn_weighted']),patch.object(experiment,'LRS',[1.]),patch.object(experiment,'STEPS',1),patch('torch.nn.functional.binary_cross_entropy_with_logits',side_effect=capture):
                    result=experiment.run(output)
                self.assertEqual(len(captured),1)
                weight,dtype,device=captured[0];q=result['protocol']['weighted_positive_weight_train_only']
                self.assertEqual(weight.dtype,torch.float64);self.assertEqual(weight.dtype,dtype);self.assertEqual(weight.device,device);self.assertEqual(weight.item(),q)
                x,y=load_data()['train'];initial=[v.detach().numpy().copy() for v in init('cnn_weighted',11)];expected_loss,grad,_,_=cnn_reference(initial,x,y,q)
                actual=np.load(Path(output)/'arrays.npz')['cnn_weighted_lr1_s11_parameters']
                expected=np.concatenate([(v-g).ravel() for v,g in zip(initial,grad)])
                close(actual[1],expected,tol=5e-13);close(result['candidates'][0]['history'][0]['train_objective_before_update'],expected_loss,tol=5e-13)
                ERRORS['actual_entry_weight_value_error']=abs(weight.item()-q);ERRORS['actual_entry_first_step_max_error']=float(np.max(np.abs(actual[1]-expected)))
        finally:torch.set_default_dtype(old_dtype)

    def test_domain_rejections(self):
        for bad in [[[0,0,0,1]],[[2,0,1,1]],[[0,0,np.inf,1]],[[0,0,1]],[[0,0,1e9,1]],[[0,0,1e-300,1]]]:
            with self.assertRaises(ValueError):boxes(bad)
        for fn in [lambda:nms([[0,0,1,1]],[],.5),lambda:nms([[0,0,1,1]],[1],1.1),lambda:bce([0],[.5]),lambda:bce([0],[1],0),lambda:bce([0],[1],np.inf),lambda:bce([],[]),lambda:bce([1j],[0]),lambda:smooth_l1([1],0),lambda:mask_metrics([[0]],[[0]]),lambda:mask_metrics(np.zeros((1,1,1)),np.ones((1,2,1))),lambda:detection_ap([],[],np.nan)]:
            with self.assertRaises(ValueError):fn()

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite);print(json.dumps(ERRORS,indent=2));raise SystemExit(not r.wasSuccessful())
