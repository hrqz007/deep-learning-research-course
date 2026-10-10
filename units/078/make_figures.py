"""Render six original figures from saved, measured results."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
D=np.load(ROOT/'outputs/data.npz');R=json.loads((ROOT/'outputs/results.json').read_text());F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(name):plt.tight_layout();plt.savefig(F/name,dpi=180,bbox_inches='tight');plt.close()
a=D['adjacency'][125];x=D['x'][125,:,0];ang=np.arange(8)*2*np.pi/8;pos=np.c_[np.cos(ang),np.sin(ang)]
fig,ax=plt.subplots(figsize=(8,4.3))
for i in range(8):
 for j in range(i):
  if a[i,j]:ax.plot(pos[[i,j],0],pos[[i,j],1],c='#94a3b8',zorder=1)
sc=ax.scatter(*pos.T,c=x,cmap='coolwarm',s=550,zorder=2,edgecolors='#1e293b')
for i in range(8):ax.text(*pos[i],str(i),ha='center',va='center',zorder=3)
ax.set(aspect='equal',title='一张封存测试图：节点编号不是物理身份');ax.axis('off');fig.colorbar(sc,ax=ax,label='节点读数（无单位）');save('01_graph.png')
fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
for (xx,yy,txt,c) in [(.08,.72,'自身 X\n8 × 1','#dbeafe'),(.08,.18,'邻域 PX\n8 × 1','#ccfbf1'),(.36,.45,'拼接 Z\n8 × 2','#fef3c7'),(.64,.45,'tanh(ZW+b)\n8 × 12','#e0e7ff'),(.9,.45,'预测 Y\n8 × 1','#fce7f3')]:ax.text(xx,yy,txt,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc=c,ec='#64748b'))
for fr,to in [((.16,.7),(.29,.48)),((.16,.2),(.29,.42)),((.43,.45),(.54,.45)),((.74,.45),(.83,.45))]:ax.annotate('',xy=to,xytext=fr,arrowprops=dict(arrowstyle='->',lw=2))
ax.set_title('一跳消息与自身跳接：所有节点共享49个参数');save('02_message.png')
from experiment import transition
p=transition(a);perm=[3,7,1,0,6,4,2,5]
fig,axes=plt.subplots(1,2,figsize=(9,4))
for ax,mat,title in zip(axes,[p,p[perm][:,perm]],['原始 P','同时重排行与列的 P′']):ax.imshow(mat,vmin=0,vmax=.5,cmap='Blues');ax.set(title=title,xlabel='消息来源节点',ylabel='接收节点')
fig.suptitle('预测最大重编号误差 = '+f"{R['permutation_max_error']:.2e}");save('03_permutation.png')
fig,axes=plt.subplots(1,2,figsize=(10,4))
for name,label in [('gnn','消息模型'),('blind','关系无关')]:axes[0].semilogy(np.arange(40,1601,40),D[name+'_history'],label=label);axes[1].scatter([label]*3,R[name+'_test_mse_seeds'],s=60)
axes[0].set(xlabel='优化步数',ylabel='训练MSE');axes[0].legend();axes[1].set(yscale='log',ylabel='封存测试MSE（3种子）');save('04_training.png')
fig,ax=plt.subplots(figsize=(8,4));ax.semilogy(np.arange(41),np.maximum(D['smooth_variance'],1e-20),'o-',color='#0f766e');ax.set(xlabel='纯传播层数',ylabel='节点方差（显示下限 1e-20）',title='反复邻域平均消除节点差异');save('05_smoothing.png')
fig,ax=plt.subplots(figsize=(8,4));ax.bar(np.arange(8),D['intervention_effect'][:,0],color='#d97706');ax.set(xlabel='节点',ylabel='目标均值变化（无单位）',title='已知合成机制：固定图和其他X，对 X₀ 加1');save('06_intervention.png')
