from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from experiment import field
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'figure.dpi':160})
F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(name):plt.savefig(F/name,bbox_inches='tight');plt.close()
def boxes(name,texts):
 fig,ax=plt.subplots(figsize=(9,2.6));ax.axis('off')
 for i,s in enumerate(texts):
  x=.13+i*.36;ax.text(x,.55,s,ha='center',va='center',bbox=dict(boxstyle='round,pad=.8',facecolor=['#e3eff8','#e9f3e7','#fcecd9'][i],edgecolor='#506579'),transform=ax.transAxes)
  if i<2:ax.annotate('',xy=(x+.24,.55),xytext=(x+.12,.55),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2})
 save(name)
boxes('01_pipeline.png',['位置 x\n20个 tanh 单元','u、u′、u″\n可训练参数 θ','方程残差 + 边界\n独立网格评价'])
fig,ax=plt.subplots(figsize=(8,3.3))
for label in ['pinn','pinn_seed82','pinn_sparse']:
 a=np.loadtxt(ROOT/'outputs'/f'{label}_history.csv',delimiter=',',skiprows=1);ax.semilogy(a[:,0],np.maximum(a[:,1],1e-30),label=label)
ax.set(xlabel='参数更新步数',ylabel='总训练损失');ax.legend();ax.grid(alpha=.2);save('02_loss.png')
a=np.loadtxt(ROOT/'data/evaluation.csv',delimiter=',',skiprows=1);x=a[:,0]
fig,ax=plt.subplots(figsize=(8,3.3))
for label in ['pinn','pinn_sparse']:
 z=np.load(ROOT/'outputs'/f'{label}_weights.npz');r=-field([z[k] for k in ['w','b','v','c']],x)[2]-np.pi**2*np.sin(np.pi*x);ax.plot(x,r,label=label)
for p in (np.arange(5)+.5)/5:ax.axvline(p,color='gray',alpha=.25)
ax.set(xlabel='无量纲位置 x',ylabel='方程残差 −u″−f');ax.legend();save('03_residual.png')
fig,axs=plt.subplots(1,2,figsize=(9,3.3))
for col,label in [(1,'真解'),(2,'PINN'),(5,'差分63')]:axs[0].plot(x,a[:,col],label=label)
for col,label in [(2,'PINN'),(4,'五点PINN'),(5,'差分63')]:axs[1].plot(x,a[:,col]-a[:,1],label=label)
for ax in axs:ax.set_xlabel('无量纲位置 x');ax.legend()
axs[0].set_ylabel('无量纲解 u');axs[1].set_ylabel('预测减真解');save('04_solution.png')
fig,axs=plt.subplots(2,1,figsize=(9,3.5))
for ax,texts in zip(axs,[['固定热源 f0','x → uθ(x)','一条解曲线'],['多条输入热源 f','Gθ: f → u','新函数实例测试']]):
 ax.axis('off')
 for i,s in enumerate(texts):
  ax.text(.15+i*.35,.5,s,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',facecolor='#e4eef7'),transform=ax.transAxes)
  if i<2:ax.annotate('',xy=(.39+i*.35,.5),xytext=(.26+i*.35,.5),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':1.8})
save('05_operator.png')
