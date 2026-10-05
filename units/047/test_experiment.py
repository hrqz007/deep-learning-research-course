"""Independent patch/attention/whole-model/gradient contracts, survive Python -O."""
from pathlib import Path
import math,json,unittest,gzip
import numpy as np
import torch
from experiment import patchify,unpatchify,hand_ledger,new_model,permutation_audit,saved_tensor_audit,TinyAttention
import generate_data
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)
def close(a,b,tol=3e-11):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def npconv(x,k,b):
    xp=np.pad(x,((0,0),(0,0),(1,1),(1,1)));patches=np.lib.stride_tricks.sliding_window_view(xp,(3,3),axis=(2,3));return np.einsum('ncrsab,ocab->nors',patches,k)+b[None,:,None,None]
def numpy_model(s,x,kind):
    linear=lambda x,name:x@s[name+'.weight'].T+s[name+'.bias']
    if kind=='cnn':
        a=np.maximum(npconv(x,s['conv.0.weight'],s['conv.0.bias']),0);a=np.maximum(npconv(a,s['conv.2.weight'],s['conv.2.bias']),0);return linear(a.mean((2,3)),'head')
    if kind=='hybrid':x=np.maximum(npconv(x,s['stem.0.weight'],s['stem.0.bias']),0)
    # Explicit slices independently of experiment.patchify and torch reshape.
    tokens=np.stack([x[:,:,r:r+2,c:c+2].reshape(len(x),-1) for r in range(0,8,2) for c in range(0,8,2)],axis=1)
    z=linear(tokens,'patch')+s['pos']
    def norm(x,name):
        center=x-x.mean(-1,keepdims=True);return center/np.sqrt((center*center).mean(-1,keepdims=True)+1e-5)*s[name+'.weight']+s[name+'.bias']
    zz=norm(z,'norm1');qkv=linear(zz,'attn.qkv');Q,K,V=np.split(qkv,3,axis=-1);B,N,D=Q.shape
    Q=Q.reshape(B,N,2,4).transpose(0,2,1,3);K=K.reshape(B,N,2,4).transpose(0,2,1,3);V=V.reshape(B,N,2,4).transpose(0,2,1,3);scores=Q@K.transpose(0,1,3,2)/2;A=np.exp(scores-scores.max(-1,keepdims=True));A/=A.sum(-1,keepdims=True)
    z=z+linear((A@V).transpose(0,2,1,3).reshape(B,N,D),'attn.proj');f=linear(norm(z,'norm2'),'ff.0');gelu=.5*f*(1+np.vectorize(math.erf)(f/math.sqrt(2)));z=z+linear(gelu,'ff.2');return linear(norm(z,'final_norm').mean(1),'head')
def nll(logits,y):
    m=logits.max(-1);return np.mean(m+np.log(np.exp(logits-m[:,None]).sum(-1))-logits[np.arange(len(y)),y])
class Contracts(unittest.TestCase):
    def test_patch_order_roundtrip_and_conv_embedding(self):
        x=np.arange(2*3*4*6.).reshape(2,3,4,6);p=patchify(x,2);manual=np.stack([x[:,:,r:r+2,c:c+2].reshape(2,-1) for r in range(0,4,2) for c in range(0,6,2)],axis=1);close(p,manual);close(unpatchify(p,2,x.shape),x)
        rng=np.random.default_rng(4771);W=rng.normal(size=(12,5));b=rng.normal(size=5);dense=p@W+b;conv=torch.nn.functional.conv2d(torch.tensor(x),torch.tensor(W.T.reshape(5,3,2,2)),torch.tensor(b),stride=2).flatten(2).transpose(1,2).numpy();close(dense,conv)
    def test_attention_official_module(self):
        with torch.random.fork_rng(devices=[]):torch.manual_seed(4772);a=TinyAttention().double();official=torch.nn.MultiheadAttention(8,2,batch_first=True,dropout=0,dtype=torch.float64)
        with torch.no_grad():official.in_proj_weight.copy_(a.qkv.weight);official.in_proj_bias.copy_(a.qkv.bias);official.out_proj.weight.copy_(a.proj.weight);official.out_proj.bias.copy_(a.proj.bias)
        x=torch.arange(2*5*8,dtype=torch.float64).reshape(2,5,8)/50;close(a(x).detach().numpy(),official(x,x,x,need_weights=True)[0].detach().numpy())
    def test_full_models_independent_numpy_forward(self):
        rng=np.random.default_rng(4773);x=rng.normal(size=(2,1,8,8))
        for kind,count in [('cnn',682),('vit',802),('hybrid',938)]:
            m=new_model(kind,4711);s={k:v.detach().numpy().copy() for k,v in m.state_dict().items()};close(m(torch.tensor(x)).detach().numpy(),numpy_model(s,x,kind));self.assertEqual(sum(p.numel() for p in m.parameters()),count)
    def test_full_vit_all_parameters_numpy_difference(self):
        rng=np.random.default_rng(4774);x=rng.normal(0,.3,(2,1,8,8));y=np.array([0,1]);m=new_model('vit',4712);loss=torch.nn.functional.cross_entropy(m(torch.tensor(x)),torch.tensor(y));loss.backward();s={k:v.detach().numpy().copy() for k,v in m.state_dict().items()};self.assertAlmostEqual(float(loss.detach()),nll(numpy_model(s,x,'vit'),y),places=12)
        for name,p in m.named_parameters():
            array=s[name];g=p.grad.numpy()
            for idx in np.ndindex(array.shape):
                old=array[idx];array[idx]=old+1e-6;plus=nll(numpy_model(s,x,'vit'),y);array[idx]=old-1e-6;minus=nll(numpy_model(s,x,'vit'),y);array[idx]=old
                self.assertAlmostEqual(float(g[idx]),float((plus-minus)/2e-6),delta=2e-7,msg=f'{name}{idx}')
    def test_hand_all_gradients_and_states(self):
        x=np.array([[[1.,0.],[0.,1.]],[[0.,1.],[1.,1.]]]);y=torch.tensor([1.,0.],dtype=torch.float64)
        for state in hand_ledger():
            xx=torch.tensor(x,requires_grad=True);t=torch.tensor(state['theta'],dtype=torch.float64,requires_grad=True);e=xx@t[:2]+t[2:4];q=e*t[4];k=e*t[5];v=e*t[6];A=torch.softmax(q[:,:,None]*k[:,None,:],dim=-1);z=A@v[:,:,None];p=t[7]*z.mean((1,2))+t[8];L=((p-y)**2).mean()/2;L.backward()
            close(A.detach().numpy(),[s['attention'] for s in state['samples']]);close(p.detach().numpy(),[s['prediction'] for s in state['samples']]);close(t.grad.numpy(),state['gradient']);close(xx.grad.numpy(),[s['input_gradient'] for s in state['samples']]);self.assertAlmostEqual(float(L.detach()),state['loss'],places=14)
    def test_softmax_local_jacobian_and_permutation(self):
        scores=np.array([.2,-.4,1.]);a=np.exp(scores-scores.max());a/=a.sum();up=np.array([.5,-.7,.2]);analytic=a*(up-np.dot(a,up))
        for j in range(3):
            plus=scores.copy();minus=scores.copy();plus[j]+=1e-6;minus[j]-=1e-6;soft=lambda s:np.exp(s-s.max())/np.exp(s-s.max()).sum();self.assertAlmostEqual(analytic[j],((soft(plus)-soft(minus))@up)/2e-6,places=9)
        p=permutation_audit();self.assertLess(p['no_position_equivariance_max_abs'],1e-12);self.assertLess(p['joint_permutation_mean_invariance_max_abs'],1e-12);self.assertGreater(p['fixed_position_mean_change'],1e-3)
    def test_saved_tensor_measurement_semantics(self):
        for kind,count in [('cnn',682),('vit',802),('hybrid',938)]:
            r=saved_tensor_audit(kind,2);self.assertEqual(r['parameter_bytes'],count*8);self.assertGreater(r['saved_tensor_logical_bytes'],0);self.assertGreater(r['unique_saved_storage_bytes'],0);self.assertEqual(r['attention_score_bytes'],0 if kind=='cnn' else 2*2*16*16*8)
    def test_final_models_and_posthoc_rule(self):
        results=json.loads((ROOT/'outputs/results.json').read_text());states=json.loads(gzip.decompress((ROOT/'outputs/final_states.json.gz').read_bytes()));data=json.loads((ROOT/'data/images.json').read_text());x=np.array(data['test']['x']);y=np.array(data['test']['y'])
        self.assertEqual(len(states),18)
        for item,run in zip(states,results['runs']):
            self.assertEqual((item['kind'],item['train_size'],item['seed']),(run['kind'],run['train_size'],run['seed']))
            state={k:np.array(v,float) for k,v in item['state_dict'].items()};logits=numpy_model(state,x,item['kind']);close(logits,run['test']['logits'],tol=2e-9);self.assertAlmostEqual(nll(logits,y),run['test']['nll'],places=10)
        baseline=results['orientation_reference'];self.assertIn('post-hoc',baseline['status']);score=[]
        for image in x[:,0]:
            horizontal=sum(image[r,c]*image[r,c+1] for r in range(8) for c in range(7));vertical=sum(image[r,c]*image[r+1,c] for r in range(7) for c in range(8));score.append(horizontal-vertical)
        close(score,baseline['splits']['test']['score']);self.assertEqual(baseline['splits']['test']['accuracy'],float(np.mean((np.array(score)>=0)==y)))

    def test_fixed_data_and_rejections(self):
        self.assertEqual(generate_data.generate(),json.loads((ROOT/'data/images.json').read_text()))
        for x,p in [(np.ones((1,1,3,4)),2),(np.ones((1,1,4,4)),0),(np.ones((1,1,4,4)),True),(np.ones((1,1,4,4))*np.nan,2),(np.zeros((0,1,4,4)),2),(np.ones((1,1,4,4),complex),2)]:
            with self.assertRaises(ValueError):patchify(x,p)
        with self.assertRaises(ValueError):unpatchify(np.ones((1,4,4)),2,(1,1,3,4))
        with self.assertRaises(ValueError):new_model('vit',1)(torch.zeros((1,1,9,8),dtype=torch.float64))
        with self.assertRaises(ValueError):new_model('unknown',1)
if __name__=='__main__':unittest.main(verbosity=2)
