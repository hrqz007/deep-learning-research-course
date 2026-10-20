"""CPU post-training quantization study. Inference uses dequantized float32 weights."""
from pathlib import Path
import os,time,json,csv,platform,copy
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl-course-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/dl-course-cache')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';FIG=ROOT/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
SEEDS=(11,23,37)
def dataset(seed,n):
 r=np.random.default_rng(seed);g=r.random(n)<.2;x=r.normal(size=(n,2)).astype('float32');x[g,0]+=3.;y=(np.sin(2*x[:,0])+.6*x[:,1]>0).astype('int64');return x,y,g

def quantize(w,bits):
 qmax=2**(bits-1)-1;scale=float(np.max(np.abs(w)))/qmax
 if scale==0:scale=1.
 q=np.clip(np.rint(w/scale),-qmax,qmax).astype('int8');return q,np.float32(scale)

def pack(q,bits):
 qmax=2**(bits-1)-1;u=(q.astype('int16').ravel()+qmax).astype('uint8');binary=((u[:,None]>>np.arange(bits-1,-1,-1))&1).astype('uint8');return np.packbits(binary.ravel()).tobytes()

def unpack(raw,bits,shape):
 n=int(np.prod(shape));binary=np.unpackbits(np.frombuffer(raw,dtype='uint8'))[:n*bits].reshape(n,bits);u=(binary*(2**np.arange(bits-1,-1,-1))).sum(axis=1);return (u-(2**(bits-1)-1)).astype('int8').reshape(shape)

def forward(x,w1,b1,w2,b2,act_limit=None):
 h=torch.relu(x@w1.T+b1)
 if act_limit is not None:h=torch.round(torch.clamp(h,0,act_limit)/act_limit*15)/15*act_limit
 return h@w2.T+b2

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data={};params={};predictions={};records=[];times=[];calibration=[]
 for split,seed,n in [('train',404,1800),('calibration',505,400),('test',606,1200)]:
  for name,a in zip(('x','y','group'),dataset(seed,n)):data[split+'_'+name]=a
 np.savez(OUT/'data.npz',**data);tx=torch.from_numpy(data['train_x']);ty=torch.from_numpy(data['train_y']);cx=torch.from_numpy(data['calibration_x']);xx=torch.from_numpy(data['test_x'])
 for seed in SEEDS:
  torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(2,32),torch.nn.ReLU(),torch.nn.Linear(32,2));opt=torch.optim.Adam(model.parameters(),lr=.015)
  for _ in range(700):opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(tx),ty);loss.backward();opt.step()
  w1,b1,w2,b2=[p.detach().numpy().copy() for p in model.parameters()]
  with torch.no_grad():h=torch.relu(cx@torch.from_numpy(w1).T+torch.from_numpy(b1));limit=float(np.quantile(h.numpy(),.995))
  calibration.append(dict(seed=seed,hidden_clip_percentile=99.5,hidden_clip_limit=limit,calibration_examples=400))
  for method,bits,act in [('float32',None,False),('w8',8,False),('w4',4,False),('w2',2,False),('w4a4_fake',4,True)]:
   decoded=[];payload=bytearray();meta=[]
   for weight,bias in [(w1,b1),(w2,b2)]:
    if bits is None:ww=weight.copy();raw=weight.astype('float32').tobytes();scale=None
    else:
     q,scale=quantize(weight,bits);raw=pack(q,bits);ww=unpack(raw,bits,weight.shape).astype('float32')*scale
    payload.extend(raw)
    if scale is not None:payload.extend(scale.tobytes())
    payload.extend(bias.astype('float32').tobytes());decoded.extend([torch.from_numpy(ww),torch.from_numpy(bias.copy())]);meta.append(dict(shape=list(weight.shape),weight_payload_bytes=len(raw),scale=None if scale is None else float(scale),bias_count=len(bias)))
   name=f'{method}_s{seed}';(OUT/(name+'.bin')).write_bytes(payload)
   for key,a in zip(('w1','b1','w2','b2'),decoded):params[name+'_'+key]=a.numpy()
   with torch.no_grad():logits=forward(xx,*decoded,limit if act else None).numpy()
   predictions[name]=logits;pred=logits.argmax(1);correct=pred==data['test_y'];g=data['test_group']
   for group,mask in [('all',np.ones(len(g),bool)),('common',~g),('tail',g)]:records.append(dict(seed=seed,method=method,group=group,n=int(mask.sum()),accuracy=float(correct[mask].mean()),raw_parameter_payload_bytes=len(payload),inference_kernel='float32 matmul; optional fake activation quantization',activation_clip_limit=limit if act else None))
   for batch in (1,64):
    sample=xx[:batch]
    with torch.no_grad():
     for _ in range(5):forward(sample,*decoded,limit if act else None)
     for repeat in range(21):
      t=time.perf_counter();forward(sample,*decoded,limit if act else None);sec=time.perf_counter()-t;times.append(dict(seed=seed,method=method,batch=batch,repeat=repeat,seconds=sec))
 np.savez(OUT/'parameters.npz',**params);np.savez(OUT/'predictions.npz',**predictions)
 for fname,rs in [('quality.csv',records),('timings.csv',times),('calibration.csv',calibration)]:
  with (OUT/fname).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 metrics=dict(protocol='Train 3 float32 teachers; calibrate activation clipping on distinct split; compare PTQ weight payloads and fake activation quantization; all CPU inference uses float32 matmul. Packed payload excludes metadata/container overhead.',quality=records,timings=times,calibration=calibration,configs=['float32','w8','w4','w2','w4a4_fake'],training_steps=700,seeds=list(SEEDS),test_examples=1200)
 (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
 fig,ax=plt.subplots(1,2,figsize=(9,3.8));methods=metrics['configs'];xs=np.arange(len(methods))
 for group in ('all','common','tail'):
  means=[np.mean([r['accuracy'] for r in records if r['group']==group and r['method']==m]) for m in methods];ax[0].plot(xs,means,'o-',label=group)
 ax[0].set_xticks(xs,methods,rotation=20);ax[0].set(ylabel='Accuracy (3-seed mean)',ylim=(0,1.03));ax[0].legend()
 sizes=[next(r['raw_parameter_payload_bytes'] for r in records if r['method']==m) for m in methods];ax[1].bar(xs,sizes,color='#2874a6');ax[1].set_xticks(xs,methods,rotation=20);ax[1].set(ylabel='Actual packed parameter payload bytes');fig.tight_layout();fig.savefig(FIG/'01_quality_size.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.6))
 for batch in (1,64):ax.plot(xs,[np.median([r['seconds'] for r in times if r['method']==m and r['batch']==batch])*1e6 for m in methods],'o-',label=f'batch {batch}')
 ax.set_xticks(xs,methods);ax.set(ylabel='Median measured CPU microseconds',title='Dequantized float inference: storage is not kernel speed');ax.legend();fig.tight_layout();fig.savefig(FIG/'02_latency.png',dpi=160);plt.close(fig)
 (ROOT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='CPU',threads=1,dtype='float32',timing_scope='forward only; excludes training, quantization, decoding, and disk IO'),indent=2));print(json.dumps(dict(quality=records,calibration=calibration),indent=2))
if __name__=='__main__':main()
