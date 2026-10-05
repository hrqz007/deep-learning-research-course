"""DL053: transparent CPU float64 Transformer block and structural experiments.

No network, training corpus or dropout. Boolean `allow` means permitted key.
Every query must have a nonempty support. This is NOT PyTorch's bool mask meaning.
"""
from pathlib import Path
import argparse, copy, json, math, platform
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parent

def setup(seed=5301):
    """Fix local numeric conditions; benchmark claims are deliberately absent."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

class Norm(nn.Module):
    """Layer normalization over D, with population variance and learned affine map."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(d, dtype=torch.float64))
        self.eps = eps
    def forward(self, x):
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + self.eps) * self.weight + self.bias

class Block(nn.Module):
    """One self-attention+ReLU FFN block. No final model norm or position module."""
    def __init__(self, d=4, heads=2, width=7, pre=True, eps=1e-5):
        super().__init__()
        if d < 1 or heads < 1 or d % heads or width < 1 or eps <= 0:
            raise ValueError('positive dimensions, D divisible by H, and eps>0 required')
        self.d, self.heads, self.width, self.pre = d, heads, width, pre
        self.qkv = nn.Linear(d, 3*d, dtype=torch.float64)
        self.proj = nn.Linear(d, d, dtype=torch.float64)
        self.fc1 = nn.Linear(d, width, dtype=torch.float64)
        self.fc2 = nn.Linear(width, d, dtype=torch.float64)
        self.norm1, self.norm2 = Norm(d, eps), Norm(d, eps)
    def attention(self, x, allow):
        b, length, d = x.shape
        # The projection output is [all Q features | all K | all V].
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # Reshape splits features, transpose puts the head axis before positions.
        q, k, v = [a.reshape(b, length, self.heads, d//self.heads)
                    .transpose(1, 2) for a in (q, k, v)]
        scores = q @ k.transpose(-1, -2) / math.sqrt(d//self.heads)
        weights = torch.softmax(scores.masked_fill(~allow[:, None], -torch.inf), dim=-1)
        values = weights @ v
        # transpose is required; reshape alone would scramble token/head order.
        merged = values.transpose(1, 2).contiguous().reshape(b, length, d)
        return self.proj(merged)
    def forward(self, x, allow=None):
        if x.ndim != 3 or x.shape[-1] != self.d or min(x.shape) == 0:
            raise ValueError('x must have nonempty shape B,L,D')
        if x.device.type != 'cpu' or x.dtype != torch.float64 or not torch.isfinite(x).all():
            raise ValueError('finite CPU float64 input required')
        b, length, _ = x.shape
        if allow is None:
            allow = torch.ones(b, length, length, dtype=torch.bool)
        if allow.dtype != torch.bool or allow.device.type != 'cpu':
            raise ValueError('allow must be a CPU bool tensor')
        if allow.shape == (length, length):
            allow = allow.expand(b, -1, -1)
        if allow.shape != (b, length, length) or not allow.any(dim=-1).all():
            raise ValueError('allow shape mismatch or empty query support')
        # Residual additions use exact B,L,D shapes; no implicit feature projection.
        if self.pre:
            x = x + self.attention(self.norm1(x), allow)
            return x + self.fc2(F.relu(self.fc1(self.norm2(x))))
        x = self.norm1(x + self.attention(x, allow))
        return self.norm2(x + self.fc2(F.relu(self.fc1(x))))

def reference(block):
    """Copy every parameter, including both norm eps values, into official 2.7 API."""
    ref = nn.TransformerEncoderLayer(block.d, block.heads, block.width,
            dropout=0., activation='relu', layer_norm_eps=block.norm1.eps,
            batch_first=True, norm_first=block.pre, dtype=torch.float64)
    mapping = {'qkv': 'self_attn.in_proj', 'proj': 'self_attn.out_proj',
               'fc1': 'linear1', 'fc2': 'linear2', 'norm1': 'norm1', 'norm2': 'norm2'}
    state = ref.state_dict()
    for name, parameter in block.named_parameters():
        module, suffix = name.split('.')
        key = mapping[module] + ('_' if module == 'qkv' else '.') + suffix
        state[key] = parameter.detach().clone()
    ref.load_state_dict(state)
    return ref, mapping

def sinusoidal(positions, d):
    """Pairs (sin,cos) at frequencies 10000^(-2r/D); accepts explicit position IDs."""
    if d < 2 or d % 2 or positions.ndim != 1 or not torch.isfinite(positions).all():
        raise ValueError('even D and finite 1D positions required')
    angle = positions.double()[:, None] / (10000. ** (torch.arange(0,d,2).double()/d))
    return torch.stack((angle.sin(), angle.cos()), dim=-1).reshape(len(positions), d)

def rotate_pairs(x, positions, base=10000.):
    """RoPE for shape L,D, adjacent coordinate pairs; rotates Q/K, not V here."""
    if x.ndim != 2 or x.shape[-1] % 2 or positions.shape != (x.shape[0],) or base <= 0:
        raise ValueError('x=L,even D and one position per row required')
    angles = positions.double()[:,None] / (base ** (torch.arange(0,x.shape[-1],2).double()/x.shape[-1]))
    even, odd = x[:,0::2], x[:,1::2]
    return torch.stack((even*angles.cos()-odd*angles.sin(),
                        even*angles.sin()+odd*angles.cos()), -1).reshape_as(x)

def manual_backward(x, target, block):
    """Explicit chain rule for the single-head hand example; no autograd derivatives."""
    x=x.detach();z=block.norm1(x).detach();q,k,v=[t.detach() for t in block.qkv(z).chunk(3,-1)]
    a=(q@k.transpose(-1,-2)/math.sqrt(2)).softmax(-1)
    mixed=a@v;h=(x+block.proj(mixed)).detach();z2=block.norm2(h).detach()
    pre=block.fc1(z2).detach();u=pre.relu();y=(h+block.fc2(u)).detach()
    gy=(y-target)/4
    grads={};trace={'GY':gy}
    def linear_back(name, inp, upstream):
        module=getattr(block,name)
        grads[name+'.weight']=upstream.reshape(-1,upstream.shape[-1]).T@inp.reshape(-1,inp.shape[-1])
        grads[name+'.bias']=upstream.sum((0,1))
        return upstream@module.weight.detach()
    def norm_back(name, inp, upstream):
        module=getattr(block,name);c=inp-inp.mean(-1,keepdim=True)
        scale=torch.sqrt((c*c).mean(-1,keepdim=True)+module.eps);hat=c/scale
        grads[name+'.weight']=(upstream*hat).sum((0,1));grads[name+'.bias']=upstream.sum((0,1))
        t=upstream*module.weight.detach()
        return (t-t.mean(-1,keepdim=True)-hat*(t*hat).mean(-1,keepdim=True))/scale
    gu=linear_back('fc2',u,gy);gpre=gu*(pre>0);gz2=linear_back('fc1',z2,gpre)
    gln2=norm_back('norm2',h,gz2);gh=gy+gln2
    gmixed=linear_back('proj',mixed,gh);gv=a.transpose(-1,-2)@gmixed
    ga=gmixed@v.transpose(-1,-2);gs=a*(ga-(a*ga).sum(-1,keepdim=True))
    gq=gs@k/math.sqrt(2);gk=gs.transpose(-1,-2)@q/math.sqrt(2)
    gqkv=torch.cat([gq,gk,gv],-1);gz1=linear_back('qkv',z,gqkv)
    gln1=norm_back('norm1',x,gz1);gx=gh+gln1
    trace.update(GU=gu,Gpre=gpre,GZ2=gz2,GLN2_to_H=gln2,GH=gh,Gmixed=gmixed,
                 GV=gv,GA=ga,GS=gs,GQ=gq,GK=gk,GZ1=gz1,GLN1_to_X=gln1,GX=gx)
    return {k:v.tolist() for k,v in trace.items()},{k:v.tolist() for k,v in grads.items()}

def hand_ledger():
    """Two-token, two-feature pre-norm worked example, including two SGD updates."""
    setup(5300)
    block=Block(2,1,2,pre=True)
    with torch.no_grad():
        block.qkv.weight.copy_(torch.eye(2,dtype=torch.float64).repeat(3,1))
        block.qkv.bias.zero_(); block.proj.weight.copy_(torch.eye(2,dtype=torch.float64));block.proj.bias.zero_()
        block.fc1.weight.copy_(torch.eye(2,dtype=torch.float64));block.fc1.bias.zero_()
        block.fc2.weight.copy_(0.2*torch.eye(2,dtype=torch.float64));block.fc2.bias.zero_()
    x=torch.tensor([[[1.,3.],[2.,0.]]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([[[0.,2.],[2.,0.]]],dtype=torch.float64)
    records=[]
    for step in range(3):
        block.zero_grad(set_to_none=True);x.grad=None
        z=block.norm1(x);q,k,v=block.qkv(z).chunk(3,-1)
        scores=q@k.transpose(-1,-2)/math.sqrt(2)
        a=scores.softmax(-1);att=block.proj(a@v);h=x+att
        z2=block.norm2(h);hidden=F.relu(block.fc1(z2));ff=block.fc2(hidden);out=h+ff
        loss=((out-target)**2).sum()/8  # 1/(2N), N=4 scalar outputs.
        loss.backward()
        record={name: value.detach().tolist() for name,value in
                [('x',x),('z',z),('q',q),('k',k),('v',v),('scores',scores),('a',a),
                 ('attention',att),('h',h),('z2',z2),('hidden',hidden),('ff',ff),('out',out)]}
        record.update(step=step,loss=loss.item(),output_gradient=((out-target)/4).detach().tolist(),
             parameters={n:p.detach().tolist() for n,p in block.named_parameters()},
             gradients={n:p.grad.tolist() for n,p in block.named_parameters()})
        trace,manual_grads=manual_backward(x,target,block)
        record['backward_trace']=trace;record['manual_gradients']=manual_grads
        record['input_gradient']=x.grad.tolist()
        record['manual_autograd_max']=max(
            [(torch.tensor(manual_grads[n],dtype=torch.float64)-p.grad).abs().max().item() for n,p in block.named_parameters()]
            +[(torch.tensor(trace['GX'],dtype=torch.float64)-x.grad).abs().max().item()])
        records.append(record)
        if step < 2:
            # All parameters use the SAME forward pass gradient, at learning rate .05.
            with torch.no_grad():
                for p in block.parameters(): p.add_(p.grad,alpha=-.05)
    return records

def run(output):
    setup()
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    errors=[]
    for seed in [5301,5302,5303]:
        for pre in [True,False]:
            setup(seed); block=Block(pre=pre);ref,mapping=reference(block)
            x=torch.randn(2,5,4,dtype=torch.float64,requires_grad=True)
            xr=x.detach().clone().requires_grad_(True)
            allow=torch.ones(5,5,dtype=torch.bool).tril()
            y=block(x,allow); yr=ref(xr,src_mask=~allow)
            probe=torch.arange(y.numel(),dtype=torch.float64).reshape_as(y)/y.numel()
            (y*probe).sum().backward();(yr*probe).sum().backward()
            rp=dict(ref.named_parameters());pe=[]
            for name,p in block.named_parameters():
                m,s=name.split('.');key=mapping[m]+('_' if m=='qkv' else '.')+s
                pe.append((p.grad-rp[key].grad).abs().max().item())
            errors.append({'seed':seed,'pre':pre,'output_max':(y-yr).abs().max().item(),
              'input_gradient_max':(x.grad-xr.grad).abs().max().item(),'parameter_gradient_max':max(pe)})
    setup();block=Block();x=torch.randn(1,5,4,dtype=torch.float64)
    perm=torch.tensor([2,0,4,1,3]);pe=sinusoidal(torch.arange(5),4)
    full=block(x);causal=torch.ones(5,5,dtype=torch.bool).tril()
    q=torch.randn(5,4,dtype=torch.float64);k=torch.randn(5,4,dtype=torch.float64);positions=torch.arange(5)
    rope=rotate_pairs(q,positions)@rotate_pairs(k,positions).T
    shifted=rotate_pairs(q,positions+11)@rotate_pairs(k,positions+11).T
    changed=x.clone();changed[:,3:]+=torch.tensor([2.,-3.,1.,4.],dtype=torch.float64)
    findings={
      'permutation_no_position_max':(block(x[:,perm])-full[:,perm]).abs().max().item(),
      'permutation_fixed_positions_max':(block(x[:,perm]+pe)-block(x+pe)[:,perm]).abs().max().item(),
      'permutation_tokens_and_positions_max':(block((x+pe)[:,perm])-block(x+pe)[:,perm]).abs().max().item(),
      'causal_future_change_max':(block(changed,causal)[:,:3]-block(x,causal)[:,:3]).abs().max().item(),
      'bidirectional_future_change_max':(block(changed)[:,:3]-full[:,:3]).abs().max().item(),
      'causal_prefix_extension_max':(block(x,causal)[:,:3]-block(x[:,:3],causal[:3,:3])).abs().max().item(),
      'bidirectional_prefix_extension_max':(full[:,:3]-block(x[:,:3])).abs().max().item(),
      'rope_joint_shift_max':(rope-shifted).abs().max().item(),
      'rope_norm_max':(rotate_pairs(q,positions).norm(dim=-1)-q.norm(dim=-1)).abs().max().item(),
      'parameter_count':sum(p.numel() for p in block.parameters())}
    # A fixed mask must also be relabelled if arbitrary permutations are compared.
    masked_perm=causal[perm][:,perm]
    findings['causal_permuted_mask_max']=(block(x[:,perm],masked_perm)-block(x,causal)[:,perm]).abs().max().item()
    findings['causal_fixed_mask_max']=(block(x[:,perm],causal)-block(x,causal)[:,perm]).abs().max().item()
    result={'alignment':errors,'structure':findings,'hand':hand_ledger(),
            'scope':'CPU float64; dropout=0; numerical equivalence, not learning superiority'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
       'numpy':np.__version__,'device':'CPU','dtype':'float64','threads':1},indent=2)+'\n')
    print(json.dumps({'output':str(out),'alignment_max':max(max(r[k] for k in ['output_max','input_gradient_max','parameter_gradient_max']) for r in errors),'structure':findings},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'outputs'));a=p.parse_args();run(a.output)
