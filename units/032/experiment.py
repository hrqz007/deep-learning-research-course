"""DL032 fixed-gradient sampling and explicitly budgeted CPU SGD experiments."""
from pathlib import Path
import argparse,csv,hashlib,io,itertools,json,math,platform
from fractions import Fraction
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
INPUT_HASHES={'config.json': '3f31d1b740f66a4224b3be2ec56f451d3181e5ccd7a17b6749c5114a2ec51c53', 'order.json': 'f90d8e73c6d8c7db11ce5e39f58ff17c6c1fe55d9b9e42d78ce9b011b65553df', 'samples.csv': '9b2e5c6d0ac24372075c01f817150430c171d1995363051ac5a861496e67f402'}

def require(ok,message):
 if not ok:raise ValueError(message)

def check_inputs(path):
 path=Path(path)
 for name,digest in INPUT_HASHES.items():
  require(hashlib.sha256((path/name).read_bytes()).hexdigest()==digest,'Fixed teaching input differs: '+name)

def load_inputs(path=ROOT/'data'):
 check_inputs(path);path=Path(path)
 with (path/'samples.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 x=np.array([[float(r['x1']),float(r['x2'])] for r in rows],dtype=np.float64)
 y=np.array([[float(r['y'])] for r in rows],dtype=np.float64)
 orders=json.loads((path/'order.json').read_text());cfg=json.loads((path/'config.json').read_text())
 require(x.shape==(32,2) and y.shape==(32,1),'Unexpected dataset shape')
 require(len(orders)==40 and all(sorted(row)==list(range(32)) for row in orders),'Order must contain 40 complete permutations')
 return x,y,orders,cfg

def validate(theta,x,y):
 for a in (theta,x,y):
  require(isinstance(a,np.ndarray) and a.dtype==np.float64 and np.isfinite(a).all(),'Finite float64 arrays required')
 require(theta.shape==(13,) and x.ndim==2 and x.shape[1]==2 and y.shape==(len(x),1),'Expected theta(13), X(B,2), y(B,1)')
 require(1<=len(x)<=256 and max(np.max(np.abs(a)) for a in (theta,x,y))<=100,'Teaching domain exceeded')

def numpy_forward_backward(theta,x,y):
 validate(theta,x,y)
 w=theta[:6].reshape(3,2);b=theta[6:9];v=theta[9:12].reshape(1,3);c=theta[12]
 a=x@w.T+b;h=np.tanh(a);pred=h@v.T+c;residual=pred-y
 loss=float(.5*np.mean(residual**2));dp=residual/len(x);dh=dp@v;da=dh*(1-h*h)
 dw=da.T@x;db=da.sum(0);dv=dp.T@h;dc=dp.sum()
 gradient=np.concatenate([dw.ravel(),db,dv.ravel(),np.array([dc])])
 require(math.isfinite(loss) and np.isfinite(gradient).all(),'Nonfinite calculation')
 return loss,gradient,{'x':x,'target':y,'affine':a,'hidden':h,'prediction':pred,'residual':residual,'d_prediction':dp,'d_hidden':dh,'d_affine':da,'d_input':da@w}

def torch_loss(theta,x,y):
 w=theta[:6].reshape(3,2);b=theta[6:9];v=theta[9:12].reshape(1,3);c=theta[12]
 h=torch.tanh(x@w.T+b);pred=h@v.T+c
 require(pred.shape==y.shape,'No implicit target broadcasting')
 return .5*(pred-y).square().mean()

def train(x,y,orders,initial,batch_size,steps,lr,record=True):
 initial=np.array(initial,dtype=np.float64);validate(initial,x,y)
 require(type(batch_size) is int and 1<=batch_size<=len(x),'Invalid batch size')
 require(type(steps) is int and 1<=steps<=10000,'Invalid steps')
 require(type(lr) in (int,float) and math.isfinite(lr) and 0<lr<=1,'Learning rate outside (0,1]')
 flat=[i for order in orders for i in order]
 require(all(type(i) is int and 0<=i<len(x) for i in flat),'Invalid row index')
 require(steps*batch_size<=len(flat),'Sample order shorter than requested budget')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 xx=torch.tensor(x,dtype=torch.float64);yy=torch.tensor(y,dtype=torch.float64)
 theta=torch.tensor(initial,dtype=torch.float64,requires_grad=True);history=[];used=[]
 if record:history.append({'step':0,'examples':0,'full_training_loss':float(torch_loss(theta,xx,yy).detach()),'parameters':theta.detach().tolist()})
 for t in range(steps):
  ids=flat[t*batch_size:(t+1)*batch_size];theta.grad=None
  loss=torch_loss(theta,xx[ids],yy[ids]);require(torch.isfinite(loss).item(),'Nonfinite batch loss')
  loss.backward();require(torch.isfinite(theta.grad).all().item(),'Nonfinite batch gradient')
  with torch.no_grad():theta.add_(theta.grad,alpha=-lr)
  require(torch.isfinite(theta).all().item() and theta.abs().max().item()<=100,'Updated parameters left teaching domain')
  if record:
   used.extend(ids);history.append({'step':t+1,'examples':(t+1)*batch_size,'full_training_loss':float(torch_loss(theta,xx,yy).detach()),'parameters':theta.detach().tolist()})
 return {'parameters':theta.detach().tolist(),'history':history,'used_ids':used,'steps':steps,'examples':steps*batch_size,'batch_size':batch_size}

def first_batch_trace(x,y,orders,cfg):
 ids=orders[0][:4];theta=np.array(cfg['initial_parameters'],dtype=np.float64);xb=x[ids];yb=y[ids]
 loss,g,values=numpy_forward_backward(theta,xb,yb)
 t=torch.tensor(theta,requires_grad=True);lt=torch_loss(t,torch.tensor(xb),torch.tensor(yb));lt.backward()
 require(np.allclose(t.grad.numpy(),g,atol=2e-15,rtol=2e-14),'Torch/manual gradient mismatch')
 post=theta-cfg['learning_rate']*g;after_loss,_,after=numpy_forward_backward(post,xb,yb)
 return {'ids':ids,'initial':theta,'values':values,'shapes':{k:list(v.shape) for k,v in values.items()},'loss':loss,'gradient':g,'updated':post,'next_forward':after,'next_loss':after_loss,'learning_rate':cfg['learning_rate']}

def matrix_stats(values):
 values=[[Fraction(x) for x in row] for row in values];n=len(values);d=len(values[0]);mean=[sum(row[j] for row in values)/n for j in range(d)]
 covariance=[[sum((row[j]-mean[j])*(row[k]-mean[k]) for row in values)/n for k in range(d)] for j in range(d)]
 return mean,covariance

def sampling_exact():
 # Four fixed gradient vectors; this is a finite population, not training iterates.
 gradients=[[3,1],[1,-1],[-1,2],[-3,-2]];mean,cov=matrix_stats(gradients);cases=[]
 for batch in (1,2,3,4):
  for replace in (True,False):
   selections=list(itertools.product(range(4),repeat=batch) if replace else itertools.combinations(range(4),batch))
   averages=[[sum(Fraction(gradients[i][j]) for i in ids)/batch for j in range(2)] for ids in selections]
   empirical_mean,empirical_cov=matrix_stats(averages)
   factor=Fraction(1,batch) if replace else Fraction(4-batch,3*batch)
   expected=[[z*factor for z in row] for row in cov]
   require(empirical_mean==mean and empirical_cov==expected,'Exact finite-population enumeration failed')
   cases.append({'batch_size':batch,'replacement':replace,'possibilities':len(selections),'mean_exact':[str(z) for z in empirical_mean],'covariance_exact':[[str(z) for z in row] for row in empirical_cov],'covariance':[[float(z) for z in row] for row in empirical_cov]})
 a=[-3,-1,1,3];prob=[Fraction(1,10),Fraction(2,10),Fraction(3,10),Fraction(4,10)];g=[-Fraction(z) for z in a]
 weighted=sum(p*z for p,z in zip(prob,g));corrected=[z/(4*p) for p,z in zip(prob,g)];corrected_mean=sum(p*z for p,z in zip(prob,corrected))
 conditional=[]
 for first in (-1,1):
  theta=Fraction(first,2);remaining=-first
  conditional.append({'first_target':first,'theta_after_first':str(theta),'remaining_target':remaining,'next_gradient':str(theta-remaining),'full_gradient_here':str(theta)})
 return {'fixed_gradients':gradients,'population_mean':[str(z) for z in mean],'population_covariance':[[str(z) for z in row] for row in cov],'cases':cases,'nonuniform':{'targets':a,'probabilities':[str(z) for z in prob],'unweighted_expected_gradient':str(weighted),'uniform_objective_gradient':'0','importance_corrected_values':[str(z) for z in corrected],'importance_corrected_expectation':str(corrected_mean)},'reshuffle_conditional_counterexample':conditional}

def scalar_noise_theory():
 # f_i(theta)=.5(theta-a_i)^2, a=(-3,-1,1,3), mean0 var5.
 rows=[]
 for b in (1,4,16):
  for lr in (.1,.5,1.,2.,2.1):
   m=1.;variance=0.
   for step in range(41):
    rows.append({'batch_size':b,'learning_rate':lr,'step':step,'mean_theta':m,'variance_theta':variance,'expected_excess_loss':.5*(m*m+variance)})
    m=(1-lr)*m;variance=(1-lr)**2*variance+lr*lr*5/b
 return rows

def network_fixed_theta_sampling(x,y,theta):
 # Exact without-replacement distribution on the first eight rows, fixed theta.
 g=np.stack([numpy_forward_backward(theta,x[i:i+1],y[i:i+1])[1] for i in range(8)])
 mu=g.mean(0);centered=g-mu;cov=centered.T@centered/8;cases=[]
 for b in (1,2,4,8):
  samples=np.stack([g[list(ids)].mean(0) for ids in itertools.combinations(range(8),b)])
  empirical=(samples-mu).T@(samples-mu)/len(samples);theory=(8-b)/(7*b)*cov
  require(np.allclose(empirical,theory,atol=2e-16,rtol=2e-13),'Network covariance enumeration mismatch')
  cases.append({'batch_size':b,'possibilities':len(samples),'mean':samples.mean(0),'covariance_trace':float(np.trace(empirical)),'theory_covariance_trace':float(np.trace(theory)),'maximum_covariance_error':float(np.max(np.abs(empirical-theory)))})
 return {'rows':list(range(8)),'theta':theta,'per_example_gradients':g,'population_mean':mu,'population_covariance':cov,'cases':cases}

def serializable(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {str(k):serializable(z) for k,z in v.items()}
 if isinstance(v,(list,tuple)):return [serializable(z) for z in v]
 return v

def run_report(path=ROOT/'data'):
 x,y,orders,cfg=load_inputs(path);theta=np.array(cfg['initial_parameters'],dtype=np.float64);runs={};rows=[]
 for budget in ('equal_examples','equal_steps','linear_scaled_examples'):
  for b in cfg['batch_sizes']:
   steps=cfg['step_budget'] if budget=='equal_steps' else cfg['sample_budget']//b
   lr=cfg['learning_rate']*b/4 if budget=='linear_scaled_examples' else cfg['learning_rate']
   key=f'{budget}_b{b}';r=train(x,y,orders,theta,b,steps,lr);runs[key]=r
   rows.append({'run':key,'budget':budget,'batch_size':b,'steps':steps,'examples':r['examples'],'learning_rate':lr,'final_training_loss':r['history'][-1]['full_training_loss']})
 def js(v):return (json.dumps(serializable(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu','dtype':'float64','threads':1},'runs':rows,'scope':'Optimization on a fixed synthetic training objective, no validation/test/generalization comparison; timings are separate measured runs.'}
 return {'summary.json':js(summary),'budget-comparison.csv':buf.getvalue().encode(),'trajectories.json':js(runs),'sampling-exact.json':js(sampling_exact()),'scalar-theory.json':js(scalar_noise_theory()),'network-sampling.json':js(network_fixed_theta_sampling(x,y,theta)),'first-batch-trace.json':js(first_batch_trace(x,y,orders,cfg))}

def write_report(output,path=ROOT/'data'):
 files=run_report(path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 for name,data in files.items():
  temp=output/(name+'.tmp');temp.write_bytes(data);temp.replace(output/name)
 return json.loads(files['summary.json'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args();print(json.dumps(write_report(a.output,a.data_dir),indent=2))
if __name__=='__main__':main()
