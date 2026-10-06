"""An explicit, inference-only KV path for the DL055 decoder architecture.

Training uses forward(); caching is enabled only via cached() in eval/no-grad mode.
This module is self-contained and does not import earlier course directories.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
V, MAX_LEN, HEADS, LAYERS = 20, 7, 4, 2

class DecoderBlock(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d=d
        self.norm1=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
        self.norm2=nn.LayerNorm(d); self.fc1=nn.Linear(d,2*d); self.fc2=nn.Linear(2*d,d)
    def apply_attention(self,x,past=None):
        b,n,d=x.shape
        q,k,v=[a.reshape(b,n,HEADS,d//HEADS).transpose(1,2) for a in self.qkv(self.norm1(x)).chunk(3,-1)]
        previous=0 if past is None else past[0].shape[2]
        if past is not None:
            k=torch.cat([past[0],k],dim=2); v=torch.cat([past[1],v],dim=2)
        # Absolute query positions permit both multi-token prefill and chunked decode.
        forbidden=torch.arange(k.shape[2],device=x.device)[None,:] > (previous+torch.arange(n,device=x.device))[:,None]
        weights=(q@k.transpose(-1,-2)/math.sqrt(d//HEADS)).masked_fill(forbidden,-torch.inf).softmax(-1)
        h=(weights@v).transpose(1,2).contiguous().reshape(b,n,d)
        x=x+self.proj(h); x=x+self.fc2(F.relu(self.fc1(self.norm2(x))))
        return x,(k,v)
    def forward(self,x): return self.apply_attention(x)[0]

class TinyLM(nn.Module):
    def __init__(self,d=16):
        super().__init__()
        if not isinstance(d,int) or d<4 or d%HEADS: raise ValueError('d must be a positive multiple of4')
        self.d=d; self.embed=nn.Embedding(V,d); self.position=nn.Embedding(MAX_LEN,d)
        self.blocks=nn.ModuleList([DecoderBlock(d) for _ in range(LAYERS)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,V)
    def validate(self,x,offset=0):
        if x.ndim!=2 or x.dtype!=torch.long or min(x.shape)==0: raise ValueError('input must be nonempty B,L int64')
        if offset+x.shape[1]>MAX_LEN: raise ValueError('position capacity7 exceeded; no silent truncation')
        if (x<0).any() or (x>=V).any(): raise ValueError('token outside vocabulary')
    def forward(self,x):
        self.validate(x); h=self.embed(x)+self.position(torch.arange(x.shape[1],device=x.device))[None]
        for block in self.blocks: h=block(h)
        return self.head(self.norm(h))
    @torch.no_grad()
    def cached(self,x,cache=None):
        if self.training: raise ValueError('call eval() before inference caching')
        offset=0
        if cache is not None:
            if len(cache)!=LAYERS: raise ValueError('one K,V pair per layer required')
            offset=cache[0][0].shape[2]
            for k,v in cache:
                if k.shape!=v.shape or k.shape!=(x.shape[0],HEADS,offset,self.d//HEADS): raise ValueError('cache shape mismatch')
                if k.device!=x.device or k.dtype!=self.embed.weight.dtype: raise ValueError('cache device/dtype mismatch')
        self.validate(x,offset)
        h=self.embed(x)+self.position(torch.arange(offset,offset+x.shape[1],device=x.device))[None]
        updated=[]
        for i,block in enumerate(self.blocks):
            h,kv=block.apply_attention(h,None if cache is None else cache[i]); updated.append(kv)
        return self.head(self.norm(h)),updated

def load_model(path):
    model=TinyLM()
    with np.load(path,allow_pickle=False) as arrays:
        model.load_state_dict({k:torch.from_numpy(arrays[k].copy()) for k in arrays.files},strict=True)
    return model.eval()
