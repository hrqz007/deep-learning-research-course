"""Tests survive -O: geometry, independent interpolation/gradients and replay."""
from pathlib import Path
import json,unittest
import numpy as np
import torch
from experiment import normalize,resize_bilinear,resize_mask,crop_flip,hand_ledger,replay,model_values,numpy_gradient
import generate_data
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)
def close(x,y):np.testing.assert_allclose(x,y,rtol=2e-11,atol=2e-11)
def fs(v):return np.array([x['float'] for x in v])
class Contracts(unittest.TestCase):
    def test_half_pixel_resize_independent_framework(self):
        rng=np.random.default_rng(4560)
        for shape,size in [((3,4,5),(7,3)),((1,1,5),(3,1)),((2,3,1),(1,7)),((1,2,2),(3,3))]:
            x=rng.normal(size=shape);a=resize_bilinear(x,size);b=torch.nn.functional.interpolate(torch.tensor(x)[None],size=size,mode='bilinear',align_corners=False,antialias=False)[0].numpy();close(a,b)
        close(resize_bilinear(np.array([[[0.,1.],[2.,3.]]]),(3,3))[0],[[0,.5,1],[1,1.5,2],[2,2.5,3]])
    def test_categorical_mask_nearest_exact(self):
        mask=np.array([[0,2],[2,0]],int);a=resize_mask(mask,(3,5));b=torch.nn.functional.interpolate(torch.tensor(mask,dtype=torch.float64)[None,None],size=(3,5),mode='nearest-exact')[0,0].numpy();close(a,b);self.assertEqual(set(np.unique(a)),{0,2})
        self.assertIn(1.,resize_bilinear(mask[None],(3,3)))
    def test_normalization_and_local_derivative(self):
        x=np.array([[[.1,.5]],[[.2,.9]]]);m=np.array([.2,.3]);s=np.array([.5,.2]);close(normalize(x,m,s),[[[-.2,.6]],[[-.5,3.]]])
        for c in range(2):
            shifted=x.copy();shifted[c,0,0]+=1e-6;self.assertAlmostEqual((normalize(shifted,m,s)[c,0,0]-normalize(x,m,s)[c,0,0])/1e-6,1/s[c],places=7)
    def test_crop_flip_box_mask_alignment(self):
        x=np.arange(24.).reshape(1,4,6);mask=np.zeros((4,6),int);mask[1:3,1:5]=2;boxes=np.array([[1,1,5,3],[0,0,1,1]],float)
        g=crop_flip(x,mask,boxes,(1,2,3,4),True);close(g['image'],x[:,1:4,2:6][:,:,::-1]);close(g['boxes'],[[1,0,4,2]]);close(g['box_area_retention'],[.75,0]);self.assertEqual(g['keep_indices'].tolist(),[0])
        rr,cc=np.where(g['mask']==2);close(g['boxes'][0],[cc.min(),rr.min(),cc.max()+1,rr.max()+1])
        e=crop_flip(x,mask,np.empty((0,4)),(0,0,4,6));self.assertEqual(e['boxes'].shape,(0,4))
    def test_hand_exact_states_autograd(self):
        x=np.array([[[[0,1],[.5,.25]]],[[[1,.5],[.75,.25]]]],float);target=np.array([7/16,5/8])
        for state in hand_ledger():
            xx=torch.tensor(x,requires_grad=True);theta=fs(state['theta']);w=torch.tensor(theta[:4],requires_grad=True);b=torch.tensor(theta[4],requires_grad=True)
            u=(torch.flip(xx,[-1])-.5)/.5;p=u.flatten(1)@w+b;L=((p-torch.tensor(target))**2).mean()/2;L.backward()
            close(p.detach().numpy(),[s['prediction']['float'] for s in state['samples']]);close(np.r_[w.grad.numpy(),b.grad.numpy()],fs(state['gradient']));close(xx.grad.numpy().reshape(2,4),[fs(s['original_input_gradient']) for s in state['samples']]);self.assertAlmostEqual(float(L.detach()),state['loss']['float'],places=14)
    def test_main_network_independent_gradient_and_finite_difference(self):
        rng=np.random.default_rng(4561);ps=[rng.normal(0,.1,(4,1,3,3)),np.ones(4)*.3,rng.normal(0,.02,256),np.array(.1)];x=rng.normal(0,.2,(2,1,8,8));y=np.array([0,1]);p=[torch.tensor(v,requires_grad=True) for v in ps];loss=torch.nn.functional.binary_cross_entropy_with_logits(model_values(p,torch.tensor(x))[0],torch.tensor(y,dtype=torch.float64));loss.backward();nl,ng=numpy_gradient(ps,x,y);self.assertAlmostEqual(float(loss.detach()),nl,places=14)
        for a,g,q in zip(ps,ng,p):
            close(g,q.grad.numpy())
            for idx in np.ndindex(a.shape):
                old=a[idx];a[idx]=old+1e-6;plus=numpy_gradient(ps,x,y)[0];a[idx]=old-1e-6;minus=numpy_gradient(ps,x,y)[0];a[idx]=old;self.assertAlmostEqual(float(g[idx]),(plus-minus)/2e-6,delta=3e-8)
    def test_fixed_data_replay_and_semantic_changes(self):
        data=json.loads((ROOT/'data/images.json').read_text());plans=json.loads((ROOT/'data/augmentation_plans.json').read_text());self.assertEqual(data,generate_data.generate());self.assertEqual(plans,generate_data.plans());s=data['train'];p=plans['4511'];x=np.array(s['x']);mask=np.array(s['mask'],bool)
        a,y=replay(s,p,0,'background');close(a[mask],x[mask]);close(y,s['y']);close(a,replay(s,p,0,'background')[0])
        wrong,wy=replay(s,p,0,'background_rotate_stale');good,gy=replay(s,p,0,'background_rotate_correct');close(wrong,good);close(wy,s['y']);flags=np.array(p['rotate90'][0],bool);self.assertEqual(int(np.sum(wy!=gy)),int(flags.sum()))
    def test_failures(self):
        x=np.ones((1,2,2));m=np.zeros((2,2),int)
        for size in [(0,2),(True,2),(1.5,2),(10**9,2)]:
            with self.assertRaises(ValueError):resize_bilinear(x,size)
        with self.assertRaises(ValueError):normalize(x,[0],[0])
        with self.assertRaises(ValueError):normalize(x,[0,1],[1,1])
        with self.assertRaises(ValueError):resize_bilinear(x*np.nan,(3,3))
        with self.assertRaises(ValueError):resize_mask(m.astype(float),(3,3))
        with self.assertRaises(ValueError):crop_flip(x,m,[[0,0,3,2]],(0,0,2,2))
        with self.assertRaises(ValueError):crop_flip(x,m,[],(1,0,2,2))
        with self.assertRaises(ValueError):crop_flip(x,m,[],(0,0,2,2),flip=1)
        with self.assertRaises(FloatingPointError):normalize(x*1e308,[-1e308],[1])
if __name__=='__main__':unittest.main(verbosity=2)
