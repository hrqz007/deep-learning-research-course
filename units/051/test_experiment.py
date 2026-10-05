from pathlib import Path
import json,unittest,gzip,tempfile
from unittest.mock import patch
import numpy as np
import torch
import experiment
from experiment import *
from hand_calculation import ledger,objective_gradient,SEQUENCES,TARGETS
import generate_data
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)
def close(a,b,tol=3e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
class Contracts(unittest.TestCase):
 def test_cell_matches_official_and_all_gradients(self):
  rng=np.random.default_rng(5171)
  for kind in ['rnn','gru']:
   p=initialize(kind,5172,D=2,H=3);tx=torch.tensor(rng.normal(size=(4,2)),dtype=torch.float64,requires_grad=True);th=torch.tensor(rng.normal(size=(4,3)),dtype=torch.float64,requires_grad=True);tp={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in p.items()};cell=(torch.nn.RNNCell(2,3) if kind=='rnn' else torch.nn.GRUCell(2,3)).double()
   with torch.no_grad():
    for name,key in [('weight_ih','encW'),('weight_hh','encU'),('bias_ih','encbi'),('bias_hh','encbh')]:getattr(cell,name).copy_(tp[key])
   upstream=torch.tensor(rng.normal(size=(4,3)),dtype=torch.float64);a=torch_cell(tx,th,tp,'enc',kind);b=cell(tx,th);close(a.detach(),b.detach());ga=torch.autograd.grad((a*upstream).sum(),[tx,th]+[tp[k] for k in ['encW','encU','encbi','encbh']]);gb=torch.autograd.grad((b*upstream).sum(),[tx,th]+list(cell.parameters()))
   for u,v in zip(ga,gb):close(u,v)
 def test_complete_numpy_gradients_and_right_padding(self):
  for kind in ['rnn','gru']:
   p=initialize(kind,5173,D=2,H=2);seq=[[4,4,5],[6],[]];tp={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in p.items()};loss,z,c=torch_forward(tp,kind,seq);loss.backward();nl,ng,nz,nc=numpy_reference(p,kind,seq);close(loss.detach(),nl);close(z.detach(),nz);close(c.detach(),nc)
   for k in p:close(tp[k].grad,ng[k])
   pl,pg,pz,pc=numpy_reference(p,kind,seq,pad_extra=5);close(pl,nl);close(pc,nc)
   for k in p:close(pg[k],ng[k])
 def test_all_free_small_model_coordinates_finite_difference(self):
  for kind in ['rnn','gru']:
   p=initialize(kind,5174,D=2,H=2);seq=[[4,5],[6]];l,g,*_=numpy_reference(p,kind,seq)
   for key,a in p.items():
    for ij in np.ndindex(a.shape):
     if key=='E' and ij[0]==0:continue
     old=a[ij];a[ij]=old+1e-6;plus=numpy_reference(p,kind,seq)[0];a[ij]=old-1e-6;minus=numpy_reference(p,kind,seq)[0];a[ij]=old;self.assertAlmostEqual(g[key][ij],(plus-minus)/2e-6,delta=2e-8)
 def test_boundaries_lengths_and_no_teacher_future_leakage(self):
  src,sm,din,y,dm=batch([[4,5],[6],[]],pad_extra=2);self.assertEqual(src[0,:4].tolist(),[1,4,5,2]);self.assertEqual(din[0,:3].tolist(),[1,5,4]);self.assertEqual(y[0,:3].tolist(),[5,4,2]);self.assertEqual(dm.sum(),6);self.assertEqual(sm.sum(),9)
  p=initialize('gru',5175);tp={k:torch.tensor(v,dtype=torch.float64) for k,v in p.items()};_,z,c=torch_forward(tp,'gru',[[4,5]])
  def decode(tokens):
   h=c.clone();out=[]
   for token in tokens:h=torch_cell(torch.nn.functional.embedding(torch.tensor([token]),tp['E'],padding_idx=0),h,tp,'dec','gru');out.append((h@tp['outW']+tp['outb']).numpy())
   return np.stack(out,1)
  original=decode([1,5,4]);changed=decode([1,5,7]);close(original,z);close(changed[:,:2],z[:,:2]);self.assertGreater(float(np.max(np.abs(changed[:,2]-z[:,2].numpy()))),1e-9)
 def test_greedy_eos_cap_and_unfiltered_invalid_control(self):
  p=initialize('rnn',5176);p['outW'][:]=0;p['outb'][:]=-10;p['outb'][2]=10;self.assertEqual(greedy(p,'rnn',[[4,5],[]]),[[2],[2]])
  p['outb'][2]=-10;p['outb'][0]=10;self.assertEqual(greedy(p,'rnn',[[4]],max_steps=3),[[0,0,0]])
  p['outb'][0]=-10;p['outb'][4]=10;self.assertEqual(greedy(p,'rnn',[[4,5]],max_steps=2),[[4,4]])
 def test_original_data_disjoint_and_deterministic(self):
  d=json.loads((ROOT/'data/sequences.json').read_text());self.assertEqual(d,generate_data.generate());sets=[set(map(tuple,d[k])) for k in ['train','validation','test','long_test']]
  for i in range(4):
   self.assertEqual(len(sets[i]),len(d[['train','validation','test','long_test'][i]]))
   for j in range(i):self.assertTrue(sets[i].isdisjoint(sets[j]))
 def test_hand_all14_parameters_three_states_and_input_paths(self):
  for state in ledger()['states']:
   theta=np.array(state['theta']);th=torch.tensor(theta,dtype=torch.float64,requires_grad=True);p={'encW':th[:3,None],'encU':th[3:6,None],'encbi':th[6:9],'encbh':th[9:12]};xs=[];pred=[]
   for seq in SEQUENCES:
    x=torch.tensor(seq,dtype=torch.float64,requires_grad=True);xs.append(x);h=torch.zeros((1,1),dtype=torch.float64)
    for v in x:h=torch_cell(v.reshape(1,1),h,p,'enc','gru')
    pred.append(th[12]*h[0,0]+th[13])
   output=torch.stack(pred);loss=((output-torch.tensor(TARGETS,dtype=torch.float64))**2).mean()/2;loss.backward();close(th.grad,state['gradient']);close(output.detach(),[float(v) for v in state['decimal70_predictions']]);close(float(loss.detach()),float(state['decimal70_loss']))
   for i,x in enumerate(xs):close(x.grad,[b['dx'] for b in sorted(state['samples'][i]['backward'],key=lambda b:b['time'])])
   for j in range(14):
    a=theta.copy();a[j]+=1e-6;plus=objective_gradient(a)[0];a[j]-=2e-6;minus=objective_gradient(a)[0];self.assertAlmostEqual(state['gradient'][j],(plus-minus)/2e-6,delta=1e-9)
 def test_lstm_explicit_gate_values_and_direct_memory_path(self):
  cprev=1.2;i=.25;f=.8;g=-.5;o=.6;c=f*cprev+i*g;h=o*np.tanh(c);self.assertAlmostEqual(c,.835,places=14);cell=torch.nn.LSTMCell(1,1).double()
  with torch.no_grad():
   for p in cell.parameters():p.zero_()
   cell.bias_ih.copy_(torch.tensor([np.log(i/(1-i)),np.log(f/(1-f)),np.arctanh(g),np.log(o/(1-o))],dtype=torch.float64))
  cp=torch.tensor([[cprev]],dtype=torch.float64,requires_grad=True);hp=torch.zeros_like(cp);ht,ct=cell(torch.zeros_like(cp),(hp,cp));close(ht.detach(),[[h]]);close(ct.detach(),[[c]]);ct.backward();close(cp.grad,[[f]])
 def test_final_records_greedy_and_teacher_exact_identity(self):
  r=json.loads((ROOT/'outputs/results.json').read_text());d=json.loads((ROOT/'data/sequences.json').read_text());states=json.loads(gzip.decompress((ROOT/'outputs/final_states.json.gz').read_bytes()))
  for state in states:
   p={k:np.array(v) for k,v in state['parameters'].items()};record=next(v for v in r['runs'] if (v['kind'],v['seed'])==(state['kind'],state['seed']))
   for split in ['test','long_test']:
    check=evaluate(p,state['kind'],d[split]);m=record['metrics'][split];self.assertEqual(check['decoded'],m['decoded']);close(check['teacher_logits'],m['teacher_logits']);close(check['per_sequence_nll'],m['per_sequence_nll']);self.assertEqual(m['teacher_forced_sequence_exact'],m['free_sequence_exact'])
 def test_all_final_free_rollouts_with_official_cells(self):
  data=json.loads((ROOT/'data/sequences.json').read_text());r=json.loads((ROOT/'outputs/results.json').read_text());states=json.loads(gzip.decompress((ROOT/'outputs/final_states.json.gz').read_bytes()))
  for state in states:
   p={k:torch.tensor(v,dtype=torch.float64) for k,v in state['parameters'].items()};kind=state['kind'];cell_type=torch.nn.RNNCell if kind=='rnn' else torch.nn.GRUCell;enc=cell_type(8,12).double();dec=cell_type(8,12).double()
   with torch.no_grad():
    for prefix,cell in [('enc',enc),('dec',dec)]:
     for name,key in [('weight_ih','W'),('weight_hh','U'),('bias_ih','bi'),('bias_hh','bh')]:getattr(cell,name).copy_(p[prefix+key])
    for split in ['test','long_test']:
     seq=data[split];source,sm,din,target,dm=batch(seq);h=torch.zeros((len(seq),12),dtype=torch.float64)
     for t in range(source.shape[1]):h=torch.where(torch.tensor(sm[:,t,None]),enc(p['E'][torch.tensor(source[:,t])],h),h)
     tf=h.clone();zs=[]
     for t in range(din.shape[1]):tf=torch.where(torch.tensor(dm[:,t,None]),dec(p['E'][torch.tensor(din[:,t])],tf),tf);zs.append(tf@p['outW']+p['outb'])
     m=next(v for v in r['runs'] if (v['kind'],v['seed'])==(kind,state['seed']))['metrics'][split];close(torch.stack(zs,1).numpy(),m['teacher_logits'],tol=1e-10)
     done=[False]*len(seq);tokens=torch.full((len(seq),),BOS,dtype=torch.long);decoded=[[] for _ in seq]
     for step in range(10):
      candidate=dec(p['E'][tokens],h);h=torch.where(torch.tensor(done)[:,None],h,candidate);next_ids=(h@p['outW']+p['outb']).argmax(-1)
      for i,v in enumerate(next_ids.tolist()):
       if not done[i]:decoded[i].append(v);done[i]=v==EOS
      tokens=torch.tensor([EOS if done[i] else v for i,v in enumerate(next_ids.tolist())])
      if all(done):break
     self.assertEqual(decoded,m['decoded'])
 def test_training_failure_keeps_old_result_files(self):
  with tempfile.TemporaryDirectory() as folder:
   out=Path(folder);names=['results.json','training_traces.json.gz','final_states.json.gz']
   for name in names:(out/name).write_bytes(b'old result')
   with patch.object(experiment,'torch_forward',side_effect=FloatingPointError('injected nonfinite computation')):
    with self.assertRaises(FloatingPointError):run(out)
   for name in names:self.assertEqual((out/name).read_bytes(),b'old result')
 def test_reasonable_input_contracts(self):
  for seq in [[],[[1]],[[4.0]],[[True]],[[8]],[[4]*33]]:
   with self.assertRaises(ValueError):batch(seq)
  for extra in [-1,True,33]:
   with self.assertRaises(ValueError):batch([[4]],extra)
  p=initialize('gru',5177);p['E'][0,0]=1
  with self.assertRaises(ValueError):numpy_reference(p,'gru',[[4]])
  p=initialize('gru',5177);p['encU'][0,0]=np.nan
  with self.assertRaises(ValueError):greedy(p,'gru',[[4]])
  with self.assertRaises(ValueError):greedy(initialize('rnn',5178),'rnn',[[4]],max_steps=True)
  with self.assertRaises(ValueError):initialize('gru',0,D=True)
if __name__=='__main__':unittest.main(verbosity=2)
