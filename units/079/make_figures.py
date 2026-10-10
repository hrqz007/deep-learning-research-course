from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
from experiment import rotation,truth
D=np.load(ROOT/'outputs/data.npz');R=json.loads((ROOT/'outputs/results.json').read_text());P=np.load(ROOT/'outputs/predictions.npz');F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(n):plt.tight_layout();plt.savefig(F/n,dpi=180,bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(7,4));x=np.array([[1.2,.2]]);q=rotation(1.1)
for z,col,lab in [(x,'#2563eb','原始'),(x@q.T,'#e11d48','旋转后')]:
 ax.scatter(*z.T,c=col,s=70,label=lab);f=truth(z)[0];ax.arrow(*z[0],*(.45*f),width=.015,color=col,length_includes_head=True);ax.plot([0,z[0,0]],[0,z[0,1]],'--',c=col)
ax.set(xlim=(-.3,1.6),ylim=(-.3,1.6),aspect='equal',xlabel='x₁ / m',ylabel='x₂ / m',title='位置转动，力箭头同样转动（箭长缩放0.45）');ax.legend();save('01_vectors.png')
fig,ax=plt.subplots(figsize=(8,3.4));ax.axis('off')
for xx,yy,txt in [(.15,.75,'x'),(.8,.75,'F(x)'),(.15,.15,'Rx'),(.8,.15,'F(Rx) = RF(x)')]:ax.text(xx,yy,txt,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#dbeafe'))
for fr,to,lab in [((.25,.75),(.7,.75),'预测'),((.25,.15),(.65,.15),'预测'),((.15,.63),(.15,.27),'旋转'),((.8,.63),(.8,.27),'旋转')]:ax.annotate('',xy=to,xytext=fr,arrowprops=dict(arrowstyle='->',lw=2));ax.text((fr[0]+to[0])/2+.035,(fr[1]+to[1])/2+.06,lab,ha='center')
ax.set_title('两条路径必须交换，输出力不是旋转不变的标量');save('02_commute.png')
fig,ax=plt.subplots(figsize=(7,4.5));ax.scatter(*D['test_x'].T,s=15,alpha=.4,label='封存整圆测试');ax.scatter(*D['train_x'].T,s=18,label='窄扇区训练');ax.set(aspect='equal',xlabel='x₁ / m',ylabel='x₂ / m');ax.legend();save('03_distribution.png')
fig,ax=plt.subplots(figsize=(8,4))
for n,l in [('equivariant','结构等变'),('ordinary','普通'),('augmented','旋转增强')]:ax.semilogy(D['angles'],np.maximum(D[n+'_equivariance_errors'],1e-16),label=l)
ax.set(xlabel='旋转角 / rad',ylabel='最大绝对等变误差 / N',title='61个角度，256个封存测试位置');ax.legend();save('04_equivariance.png')
fig,axes=plt.subplots(1,2,figsize=(10,4))
for n,l in [('equivariant','结构'),('ordinary','普通'),('augmented','增强')]:axes[0].scatter([l]*3,R[n+'_test_mse_seeds'],s=60)
axes[0].set(yscale='log',ylabel='封存测试MSE / N²');z=D['test_x'][::12];axes[1].quiver(*z.T,*D['test_y'][::12].T,color='#64748b',angles='xy',scale_units='xy',scale=5,label='真值');axes[1].quiver(*z.T,*P['equivariant'][::12].T,color='#0891b2',angles='xy',scale_units='xy',scale=5,label='结构预测');axes[1].set(aspect='equal',xlabel='x₁ / m',ylabel='x₂ / m',title='箭长统一缩小5倍');axes[1].legend();save('05_accuracy.png')
fig,ax=plt.subplots(figsize=(8,4));q=rotation(np.pi/2);z=np.array([[1.,0.]]);a=lambda x:-x*np.array([1.,2.]);v1=a(z@q.T)[0];v2=(a(z)@q.T)[0]
ax.bar(np.array([0,1])-.16,v1,width=.32,label='先转位置再算力');ax.bar(np.array([0,1])+.16,v2,width=.32,label='先算力再旋转');ax.set(xticks=[0,1],xticklabels=['F₁','F₂'],ylabel='力 / N',title='各向异性：F = (-x₁, -2x₂)，旋转90°不再交换');ax.legend();save('06_counterexample.png')
