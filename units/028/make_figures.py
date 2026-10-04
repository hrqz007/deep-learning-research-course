"""Fixed, source-checked original figures; guard before importing matplotlib or writing."""
from experiment import require_teaching_inputs
require_teaching_inputs()
from pathlib import Path
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
p=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if p.exists():font_manager.fontManager.addfont(str(p));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(p)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'axes.unicode_minus':False})
C=['#287f9e','#d68e35','#a64c77','#459c79'];S=json.loads((HERE/'outputs/summary.json').read_text())
def save(name):plt.tight_layout();plt.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close()
def box(ax,x,y,s,color='#e9f1f4',size=12):ax.text(x,y,s,ha='center',va='center',fontsize=size,bbox={'boxstyle':'round,pad=.6','fc':color,'ec':'#a7c0ca'},linespacing=1.6)
def arrow(ax,p,q,color=C[0]):ax.add_patch(FancyArrowPatch(p,q,arrowstyle='->',mutation_scale=14,color=color,lw=1.7))

fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
for x,title,body in [(.16,'符号微分','表达式 → 新表达式\n先展开/化简也可能变大'),(.50,'数值差分','扰动输入 → 相减/相除\n步长与舍入影响结果'),(.84,'自动微分','执行运算 → 局部拉回\n链式法则仍用浮点数')]:
 box(ax,x,.77,title);box(ax,x,.25,body,size=11);arrow(ax,(x,.64),(x,.43))
save('01_three_methods.png')

fig,ax=plt.subplots(1,2,figsize=(9.5,3.6));J=np.array([[-1,2],[1,-2]]);ax[0].imshow(J,cmap='RdBu_r',vmin=-2,vmax=2)
for i in range(2):
 for j in range(2):ax[0].text(j,i,str(J[i,j]),ha='center',va='center',fontsize=22,color='white' if abs(J[i,j])==2 else 'black')
ax[0].set(xticks=[0,1],xticklabels=['x1','x2'],yticks=[0,1],yticklabels=['F1','F2'],xlabel='输入列',ylabel='输出行',title='J：在(2,−1)的局部线性映射')
ax[1].axis('off');box(ax[1],.5,.78,'输出seed v=(3,−1)');box(ax[1],.5,.44,'J^T v = (−4,8)');box(ax[1],.5,.10,'t=(1/2,2) → Jt=(3.5,−3.5)\n两边内积都等于14',size=10);arrow(ax[1],(.5,.67),(.5,.56));save('02_vjp_jacobian.png')

fig,ax=plt.subplots(1,3,figsize=(10.5,3.1),gridspec_kw={'width_ratios':[2.6,1,1.8]});M=np.arange(1,7).reshape(2,3);ax[0].imshow(M,cmap='Blues',vmin=0,vmax=7)
for i in range(2):
 for j in range(3):ax[0].text(j,i,str(M[i,j]),ha='center',va='center',fontsize=17,color='white' if M[i,j]>=5 else 'black')
ax[0].set(xticks=range(3),yticks=range(2),title='输出种子 C：(2,3)',xlabel='列j',ylabel='行i')
ax[1].barh([0,1],[6,15],color=C[0]);ax[1].set(yticks=[0,1],yticklabels=['u0','u1'],title='行内相加');ax[1].invert_yaxis()
ax[2].bar(range(3),[5,7,9],color=C[2]);ax[2].set(xticks=range(3),xticklabels=['v0','v1','v2'],title='列内相加',ylim=(0,11))
for i,v in enumerate([5,7,9]):ax[2].text(i,v+.2,str(v),ha='center')
save('03_broadcast_adjoint.png')

fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off');box(ax,.18,.76,'A：(2,1,3)');box(ax,.51,.76,'A+B：(2,4,3)');box(ax,.84,.76,'B：(1,4,1)');arrow(ax,(.30,.76),(.39,.76));arrow(ax,(.72,.76),(.62,.76))
box(ax,.18,.22,'A的梯度：(2,1,3)\n沿axis1相加，保留维度',size=11);box(ax,.84,.22,'B的梯度：(1,4,1)\n沿axis0和2相加，保留维度',size=11);arrow(ax,(.46,.61),(.23,.39),C[2]);arrow(ax,(.56,.61),(.79,.39),C[2]);ax.text(.5,.08,'反向目标是原输入shape，不能只让代码“广播得过去”',ha='center',fontsize=11);save('04_multiaxis.png')

fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
for x,y,t in [(.13,.78,'X：(2,3)\nw：(3,)，b：()'),(.43,.78,'a=X*w+b\n(2,3)'),(.75,.78,'h=tanh(a)\n(2,3)'),(.75,.23,'z=h*h+h\n共享h，(2,3)'),(.43,.23,'L=sum(z)/6\n标量()'),(.13,.23,'gx：(2,3)\ngw：(3,)，gb：()')]:box(ax,x,y,t,size=11)
for p,q in [((.25,.78),(.33,.78)),((.54,.78),(.65,.78)),((.75,.62),(.75,.40)),((.63,.23),(.54,.23)),((.33,.23),(.25,.23))]:arrow(ax,p,q)
save('05_main_shapes.png')

fig,ax=plt.subplots(1,2,figsize=(9,3.4));g=S['mean_gradients']['w'];v=S['weighted_gradients']['w'];x=np.arange(3)
ax[0].bar(x-.18,g,width=.36,label='L均值seed=1/6',color=C[0]);ax[0].bar(x+.18,v,width=.36,label='给定非均匀seed',color=C[2]);ax[0].axhline(0,color='#666');ax[0].set(xticks=x,xticklabels=['w0','w1','w2'],ylabel='输入梯度',title='同一前向，不同标量化目标');ax[0].legend(fontsize=8)
ax[1].axis('off');box(ax[1],.5,.77,'默认 L≈0.1359075');box(ax[1],.5,.43,'均值梯度 gb≈0.8183867');box(ax[1],.5,.10,'加权VJP gb≈−0.2084006');save('06_seeds.png')

rows=list(csv.DictReader((HERE/'outputs/finite_difference.csv').open()));fig,ax=plt.subplots(figsize=(8,3.5))
for key,col in [('X',C[0]),('w',C[1]),('b',C[2])]:
 steps=sorted({float(r['epsilon']) for r in rows});err=[max(float(r['absolute_error']) for r in rows if r['node']==key and float(r['epsilon'])==h) for h in steps];ax.loglog(steps,err,'o-',color=col,label=key)
ax.set(xlabel='中心差分步长',ylabel='该输入所有坐标最大绝对误差',title='固定默认数据，10个输入坐标');ax.legend();save('07_finite_difference.png')

fig,ax=plt.subplots(1,2,figsize=(9.5,3.5));x=np.linspace(-1,1,201);ax[0].plot(x,x,color=C[0],label='数值函数 ReLU(x)−ReLU(−x)=x');ax[0].set(xlabel='x',ylabel='前向值');ax[0].legend(fontsize=8)
ax[1].plot([-1,0],[1,1],color=C[0]);ax[1].plot([0,1],[1,1],color=C[0]);ax[1].scatter([0],[1],s=65,facecolors='white',edgecolors=C[0],zorder=4);ax[1].scatter([0],[0],s=65,color=C[2],zorder=5);ax[1].annotate('局部ReLU规则组合为0',xy=(0,0),xytext=(-.9,.3),fontsize=10,arrowprops={'arrowstyle':'->'});ax[1].set(xlabel='x',ylabel='返回梯度',ylim=(-.1,1.25),title='普通导数处处1，组合约定在0不符');save('08_nonsmooth_composition.png')

fig,ax=plt.subplots(figsize=(10,3.3));ax.axis('off')
for x,y,t in [(.10,.66,'x=2'),(.37,.66,'a=x*x=4'),(.70,.66,'a*x=8'),(.10,.18,'普通链式\n梯度12'),(.37,.18,'detach(a)：值4\n阻断这条历史'),(.70,.18,'detach(a)*x=8\n对x返回4')]:box(ax,x,y,t,size=11)
arrow(ax,(.18,.66),(.27,.66));arrow(ax,(.48,.66),(.59,.66));arrow(ax,(.48,.18),(.58,.18));ax.text(.90,.48,'同值\n≠\n同导数规则',ha='center',va='center',fontsize=12,color=C[2]);save('09_barrier.png')

fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
for x,y,t in [(.15,.76,'前向图 f\n.5(x²+x)²'),(.51,.76,'一次vjp\n返回数值1.5'),(.85,.76,'不能对该数组\n再调用vjp'),(.15,.20,'手写导数表达式\n(x²+x)(2x+1)'),(.51,.20,'新建导数图'),(.85,.20,'再vjp得到5.5\n本例二阶导数')]:box(ax,x,y,t,size=11)
for y in [.76,.20]:
 for p,q in [((.27,y),(.39,y)),((.63,y),(.73,y))]:arrow(ax,p,q)
ax.text(.50,.47,'x=0.5；重新构图是人工推导，本引擎没有create_graph',ha='center',fontsize=11,color=C[2]);save('10_higher_order.png')

fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
for x,title,body in [(.16,'一张图的分支','多个使用者 → 同一节点\n必须相加'),(.50,'两次vjp调用','各自新字典\n本实现不跨调用相加'),(.84,'跨微批累计','使用者显式组合\n权重与归约另行定义')]:
 box(ax,x,.77,title);box(ax,x,.25,body,size=11);arrow(ax,(x,.64),(x,.43))
save('11_accumulation_levels.png')
