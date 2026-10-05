"""Scalar GRU: two independent sequences, all14 shared parameters, two updates."""
from decimal import Decimal,localcontext
from pathlib import Path
import json,argparse
import numpy as np
from experiment import numpy_cell,numpy_cell_backward
NAMES=['W_ir','W_iz','W_in','W_hr','W_hz','W_hn','b_ir','b_iz','b_in','b_hr','b_hz','b_hn','w_out','b_out']
INITIAL=np.array([.4,-.3,.5,.2,.1,-.4,.1,-.2,.05,-.05,.1,.02,1.2,-.1],np.float64)
SEQUENCES=[[1.,-1.],[.5]];TARGETS=[.2,-.1]
def unpack(theta):return {'encW':theta[:3,None],'encU':theta[3:6,None],'encbi':theta[6:9],'encbh':theta[9:12]}
def objective_gradient(theta):
 p=unpack(theta);samples=[];total=np.zeros(14);loss=0.
 for seq,y in zip(SEQUENCES,TARGETS):
  h=np.zeros((1,1));cache=[];steps=[]
  for t,value in enumerate(seq):
   x=np.array([[value]]);new,c=numpy_cell(x,h,p,'enc','gru');xx,hp,r,z,n,bn=c;steps.append({'time':t+1,'x':value,'h_prev':float(h[0,0]),'r':float(r[0,0]),'z':float(z[0,0]),'n':float(n[0,0]),'hidden_affine_n':float(bn[0,0]),'h':float(new[0,0])});cache.append(c);h=new
  pred=theta[12]*h[0,0]+theta[13];res=pred-y;up=res/2;grad=np.zeros(14);grad[12]=up*h[0,0];grad[13]=up;dh=np.array([[up*theta[12]]]);backward=[]
  for t in reversed(range(len(seq))):
   x,hp,r,z,n,bn=cache[t];dn=dh*(1-z);dz=dh*(hp-n);da_n=dn*(1-n*n);dr=da_n*bn;da_r=dr*r*(1-r);da_z=dz*z*(1-z);dx,prev,g=numpy_cell_backward(dh,cache[t],p,'enc','gru');con=np.r_[g['encW'].ravel(),g['encU'].ravel(),g['encbi'],g['encbh'],0.,0.];grad+=con;backward.append({'time':t+1,'dh':float(dh[0,0]),'dn':float(dn[0,0]),'dz':float(dz[0,0]),'da_n':float(da_n[0,0]),'dr':float(dr[0,0]),'da_r':float(da_r[0,0]),'da_z':float(da_z[0,0]),'dx':float(dx[0,0]),'dh_prev_direct':float((dh*z)[0,0]),'dh_prev_via_gates':float((prev-dh*z)[0,0]),'dh_prev_total':float(prev[0,0]),'parameter_contribution':con.tolist()});dh=prev
  total+=grad;loss+=res*res/4;samples.append({'input':seq,'target':y,'steps':steps,'prediction':float(pred),'residual':float(res),'individual_half_mse':float(res*res/2),'prediction_upstream':float(up),'backward':backward,'gradient_contribution':grad.tolist()})
 return float(loss),total,samples

def decimal_forward(theta):
 with localcontext() as ctx:
  ctx.prec=70;v=[Decimal(str(x)) for x in theta];out=[];one=Decimal(1)
  def sig(x):return one/(one+(-x).exp())
  def tanh(x):e=(2*x).exp();return (e-one)/(e+one)
  for seq,y in zip(SEQUENCES,TARGETS):
   h=Decimal(0)
   for value in seq:
    x=Decimal(str(value));r=sig(v[0]*x+v[3]*h+v[6]+v[9]);z=sig(v[1]*x+v[4]*h+v[7]+v[10]);n=tanh(v[2]*x+v[8]+r*(v[5]*h+v[11]));h=(one-z)*n+z*h
   out.append(v[12]*h+v[13])
  loss=sum((p-Decimal(str(y)))**2 for p,y in zip(out,TARGETS))/4
  return [str(x) for x in out],str(loss)

def ledger():
 theta=INITIAL.copy();states=[]
 for step in range(3):
  loss,g,samples=objective_gradient(theta);pred,dloss=decimal_forward(theta);states.append({'step':step,'parameter_order':NAMES,'theta':theta.tolist(),'loss':loss,'gradient':g.tolist(),'samples':samples,'decimal70_predictions':pred,'decimal70_loss':dloss})
  if step<2:theta=theta-.2*g
 return {'learning_rate':.2,'initial_state_each_sequence':0.,'objective':'mean of two final-output half-MSE losses','states':states}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs/hand_ledger.json');p=a.parse_args().output;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ledger(),indent=2)+'\n');print(p)
