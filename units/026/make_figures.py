"""Original mechanism figures, fixed teaching data verified before any figure write."""
from experiment import require_teaching_inputs
require_teaching_inputs()
from pathlib import Path
import csv,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
p=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if p.exists():font_manager.fontManager.addfont(str(p));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(p)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'axes.unicode_minus':False})
C=['#237c9b','#d68c33','#aa4f78','#379579'];summary=json.loads((HERE/'outputs/summary.json').read_text())
def save(n):plt.tight_layout();plt.savefig(OUT/n,bbox_inches='tight',facecolor='white');plt.close()
def box(ax,x,y,text,color='#e8f1f4',fontsize=11):ax.text(x,y,text,ha='center',va='center',fontsize=fontsize,bbox={'boxstyle':'round,pad=.5','fc':color,'ec':'#b1c5ce'},linespacing=1.5)
def arrow(ax,a,b,text='',rad=0,color=C[0]):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=14,lw=1.7,color=color,connectionstyle=f'arc3,rad={rad}'))
 if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.03,text,ha='center',fontsize=10,color=color)

fig,ax=plt.subplots(figsize=(10,3.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
for x,y,t in [(.08,.72,'w=1/2'),(.08,.26,'x=2'),(.29,.5,'m=wx\n1'),(.29,.15,'b=−1/4'),(.49,.5,'a=m+b\n3/4'),(.67,.5,'h=a·a\n9/16'),(.86,.5,'s=h+w\n17/16')]:box(ax,x,y,t)
for a,b in [((.15,.69),(.23,.56)),((.15,.28),(.23,.44)),((.35,.5),(.42,.5)),((.32,.24),(.46,.41)),((.55,.5),(.61,.5)),((.73,.5),(.8,.5))]:arrow(ax,a,b)
arrow(ax,(.13,.81),(.87,.63),rad=-.15,color=C[2]);ax.text(.65,.96,'同一个w还直接进入s',color=C[2],ha='center');ax.text(.68,.17,'h的两条输入边都来自a；图中合画为a·a',ha='center',fontsize=10);save('01_forward_graph.png')

fig,ax=plt.subplots(figsize=(10,3.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
for x,y,t in [(.09,.66,'s=17/16'),(.09,.24,'y=1/4'),(.36,.5,'e=s−y\n13/16'),(.61,.5,'q=e·e\n169/256'),(.87,.5,'L=q/2\n169/512')]:box(ax,x,y,t)
arrow(ax,(.17,.64),(.29,.55),'1');arrow(ax,(.17,.27),(.29,.45),'−1');arrow(ax,(.43,.5),(.54,.5),'2e');arrow(ax,(.69,.5),(.8,.5),'1/2')
ax.text(.58,.89,'前向值和局部导数是两种量',ha='center',fontsize=14);ax.text(.66,.18,'反向：bar L=1 → bar q=1/2 → bar e=13/16',ha='center',fontsize=11,color=C[2]);save('02_loss_graph.png')

fig,ax=plt.subplots(1,2,figsize=(9.5,3.7));ax[0].axis('off');box(ax[0],.5,.8,'上游：bar z = dL/dz');box(ax[0],.5,.43,'局部：z=f(u)，dz/du');box(ax[0],.5,.08,'下游输入：bar u += bar z · dz/du');arrow(ax[0],(.5,.69),(.5,.55));arrow(ax[0],(.5,.31),(.5,.21));
ax[1].bar(['乘法路径','直接路径','合计'],[39/16,13/16,13/4],color=[C[0],C[2],C[3]])
for i,v in enumerate([39/16,13/16,13/4]):ax[1].text(i,v+.08,['39/16','13/16','13/4'][i],ha='center')
ax[1].set(ylabel='dL/dw 的贡献',ylim=(0,3.8),title='同一个参数，贡献必须相加');save('03_local_and_shared.png')

fig,ax=plt.subplots(1,2,figsize=(9.5,3.7));ax[0].set(xlim=(0,1),ylim=(0,1));ax[0].axis('off');box(ax[0],.15,.5,'a=3/4');box(ax[0],.78,.5,'h=a·a\nbar h=13/16')
arrow(ax[0],(.26,.57),(.63,.58),rad=-.35);arrow(ax[0],(.26,.43),(.63,.42),rad=.35)
ax[0].text(.43,.82,'输入位置0：局部3/4',ha='center',fontsize=10);ax[0].text(.43,.18,'输入位置1：局部3/4',ha='center',fontsize=10)
ax[1].bar(['错误去重一条边','正确保留两条边'],[39/64,39/32],color=[C[2],C[0]])
for i,v in enumerate([39/64,39/32]):ax[1].text(i,v+.045,['39/64','39/32'][i],ha='center')
ax[1].set(ylabel='dL/da',ylim=(0,1.5),title='节点去重，不等于输入位置去重');save('04_repeated_edges.png')

fig,ax=plt.subplots(figsize=(10,3.2));ax.axis('off');labels=['w,x,b','m','a','h','s','e','q','L'];xs=[.06+i*.125 for i in range(8)]
for x,t in zip(xs,labels):box(ax,x,.68,t,fontsize=12)
for i in range(7):arrow(ax,(xs[i]+.043,.68),(xs[i+1]-.043,.68))
for i in range(7,0,-1):arrow(ax,(xs[i]-.035,.25),(xs[i-1]+.035,.25),color=C[2])
ax.text(.5,.95,'前向拓扑：输入在使用者之前（固定常量略）',ha='center',fontsize=13);ax.text(.5,.03,'反向拓扑：先收齐所有使用者的贡献，再处理这个节点',ha='center',fontsize=12,color=C[2]);save('05_topological_order.png')

rows=list(csv.DictReader((HERE/'outputs/gradient_check.csv').open()));fig,ax=plt.subplots(1,2,figsize=(9.5,3.5))
for key,col in [('w',C[0]),('b',C[1])]:
 r=[v for v in rows if v['parameter']==key];ax[0].loglog([float(v['step']) for v in r],[max(float(v['absolute_error']),1e-18) for v in r],'o-',label=key,color=col)
ax[0].set(xlabel='中心差分步长',ylabel='绝对误差（0显示为1e−18）',title='差分是检查器，不是反传引擎');ax[0].legend()
z=[-1+i*.01 for i in range(201)];ax[1].plot(z,[max(v,0) for v in z],color=C[0]);ax[1].plot([-.8,.8],[-.4,.4],ls='--',color=C[2],label='跨0对称差分斜率1/2');ax[1].scatter([0],[0],color=C[1],zorder=5,label='代码在0取局部值0');ax[1].set(xlabel='u',ylabel='ReLU(u)',title='尖点没有唯一普通导数');ax[1].legend(fontsize=9);save('06_checks_and_kinks.png')

fig,ax=plt.subplots(1,2,figsize=(9.5,3.5));ts=[i/2 for i in range(61)]
slope=[4*math.exp(-2*t)/(1+math.exp(-2*t))**2 for t in ts];naive=[1-math.tanh(t)**2 for t in ts];ax[0].semilogy(ts,slope,color=C[0],label='稳定的等价导数式');nz=[i for i,v in enumerate(naive) if v>0];ax[0].semilogy([ts[i] for i in nz],[naive[i] for i in nz],'.',color=C[2],label='1−tanh²，非零点');ax[0].set(xlabel='u',ylabel='tanh局部导数',title='实数链式法则仍经过浮点运算');ax[0].legend(fontsize=9)
ax[1].axis('off');box(ax[1],.5,.8,'新参数数值 → 新叶子 → 新图');box(ax[1],.5,.45,'backward(L) → 新梯度字典');box(ax[1],.5,.1,'优化器使用字典，显式更新数值');arrow(ax[1],(.5,.68),(.5,.57));arrow(ax[1],(.5,.33),(.5,.23));save('07_numeric_and_lifecycle.png')

hist=list(csv.DictReader((HERE/'outputs/training.csv').open()));fig,ax=plt.subplots(1,2,figsize=(9.5,3.5));steps=[int(v['step']) for v in hist]
ax[0].semilogy(steps,[float(v['loss']) for v in hist],color=C[0]);ax[0].set(xlabel='更新次数',ylabel='单样本平方损失',title='40次更新：0.330078 → 0.00012754')
ax[1].plot(steps,[float(v['w']) for v in hist],color=C[0],label='w');ax[1].plot(steps,[float(v['b']) for v in hist],color=C[2],label='b');ax[1].set(xlabel='更新次数',ylabel='参数',title='η=0.02；两个参数同步更新');ax[1].legend();save('08_training.png')

fig,ax=plt.subplots(figsize=(9.5,3.4));ax.axis('off')
for x,y,t in [(.15,.7,'同一数值≠同一节点'),(.5,.7,'同一节点≠只留一条边'),(.83,.7,'算出梯度≠更新参数'),(.15,.24,'叶子身份决定\n参数是否共享'),(.5,.24,'每个输入位置\n各贡献一次'),(.83,.24,'梯度方向与步长\n分开检查')]:box(ax,x,y,t,fontsize=12)
for x in [.15,.5,.83]:arrow(ax,(x,.57),(x,.39),color=C[2])
save('09_debugging_invariants.png')

fig,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(-.04,1.05),ylim=(0,1));ax.axis('off')
for x,y,t in [(.03,.6,'w\n13/4'),(.19,.6,'m\n39/32'),(.35,.6,'a\n39/32'),(.51,.6,'h\n13/16'),(.67,.6,'s\n13/16'),(.83,.6,'e\n13/16'),(.99,.6,'L\n1'),(.19,.15,'x\n39/64'),(.35,.15,'b\n39/32'),(.83,.15,'y\n−13/16')]:box(ax,x,y,t,fontsize=12)
for i in range(6):
 x=.99-i*.16;arrow(ax,(x-.055,.6),(x-.105,.6),color=C[2])
for x in [.19,.35,.83]:arrow(ax,(x,.47),(x,.28),color=C[2])
ax.plot([.67,.67,.03,.03],[.73,.88,.88,.74],color=C[2],lw=1.7);arrow(ax,(.03,.80),(.03,.73),color=C[2])
ax.text(.35,.94,'s直接给w：13/16；m再给w：39/16',ha='center',fontsize=12,color=C[2])
ax.text(.64,.13,'框内均为 dL/d(节点)\n平方的重复边贡献已经合并显示',ha='center',fontsize=10)
save('10_complete_backward.png')
