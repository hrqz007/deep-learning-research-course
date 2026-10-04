"""Original explanatory diagrams, no downloaded images or network calls."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Rectangle
import experiment as lab
BASE=Path(__file__).resolve().parent
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False,'figure.facecolor':'white','savefig.facecolor':'white'})
OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
blue='#215e9c';orange='#b64f16';green='#287b58';gray='#5e6b76'

def save(fig,name):fig.savefig(OUT/name,dpi=180,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color=blue):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='#f0f5f8',edgecolor=color,lw=1.6));ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=color,fontsize=11)
def arrow(ax,a,b,color=gray):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=color,lw=1.6))

J=np.array([[3,2],[1,1],[4,0]])
fig,axs=plt.subplots(1,2,figsize=(9,3.1),gridspec_kw={'width_ratios':[1,1.3]})
axs[0].imshow(J,cmap='Blues',vmin=0,vmax=5)
for i in range(3):
 for j in range(2):axs[0].text(j,i,str(J[i,j]),ha='center',va='center',color='white' if J[i,j]>2 else 'black',fontsize=17)
axs[0].set_xticks([0,1],['输入 x1','输入 x2']);axs[0].set_yticks([0,1,2],['输出 x1x2','输出 x1+x2','输出 x1²']);axs[0].set_title('J 的形状 (3,2)')
axs[1].barh(['f1','f2','f3'],[-1,-1,4],color=[orange,orange,blue]);axs[1].axvline(0,color=gray,lw=.8);axs[1].set_xlabel('v=(1,-2) 产生的输出变化率');axs[1].set_title('JVP = (-1,-1,4)');fig.tight_layout();save(fig,'01_jacobian.png')

fig,ax=plt.subplots(figsize=(9,3.4));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
box(ax,.2,2.35,2.3,1,'输入方向 v\n(1,n)',blue);box(ax,3.9,2.35,2,1,'局部前推\nv J^T',blue);box(ax,7.1,2.35,2.6,1,'输出变化\n(1,m)',blue);arrow(ax,(2.5,2.85),(3.9,2.85),blue);arrow(ax,(5.9,2.85),(7.1,2.85),blue)
box(ax,.2,.35,2.3,1,'输入权重\n(1,n)',orange);box(ax,3.9,.35,2,1,'局部回传\nc J',orange);box(ax,7.1,.35,2.6,1,'输出权重 c\n(1,m)',orange);arrow(ax,(7.1,.85),(5.9,.85),orange);arrow(ax,(3.9,.85),(2.5,.85),orange);ax.text(5,1.85,'同一点的 J；方向与权重的角色不同',ha='center',color=gray);save(fig,'02_products.png')

fig,ax=plt.subplots(figsize=(10,3.3));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
xs=[.1,2.55,5.0,7.45,9.9];labels=['X\n(N,D)','A=XW1+b1\n(N,H)','H=A⊙A\n(N,H)','P=HW2+b2\n(N,C)','损失 L\n标量']
for x,label in zip(xs,labels):box(ax,x,2.1,1.95,1.15,label)
for i in range(4):arrow(ax,(xs[i]+1.95,2.67),(xs[i+1],2.67),blue)
for i in range(4):arrow(ax,(xs[i+1],1.35),(xs[i]+1.95,1.35),orange)
for x,label in zip(xs,['G_X (N,D)','G_A (N,H)','G_H (N,H)','G_P (N,C)','起点权重 1']):ax.text(x+.95,.93,label,ha='center',color=orange)
ax.text(6,.15,'W1、W2 和共享偏置也各自收到梯度；参数梯度形状与参数相同',ha='center',color=gray);save(fig,'03_network.png')

cfg,X,p,Y=lab.load_config();g=lab.gradients(X,p,Y)
contrib=[np.outer(X[i],g['A'][i]) for i in range(2)]
fig,axs=plt.subplots(1,3,figsize=(9,2.8))
for ax,a,title in zip(axs,contrib+[g['W1']],['样本 1 的外积','样本 2 的外积','相加得到 G_W1']):
 ax.imshow(a,cmap='coolwarm',vmin=-60,vmax=60)
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{a[i,j]:g}',ha='center',va='center',fontsize=12)
 ax.set_xticks([0,1],['隐藏 1','隐藏 2']);ax.set_yticks([0,1],['输入 1','输入 2']);ax.set_title(title)
fig.tight_layout();save(fig,'04_contributions.png')

fig,axs=plt.subplots(1,2,figsize=(9,3.1))
for ax in axs:ax.axis('off');ax.set(xlim=(0,5),ylim=(0,4))
box(axs[0],.1,1.3,1.5,1,'共享偏置\nb11',blue);box(axs[0],3,2.5,1.8,.8,'样本1: -2',orange);box(axs[0],3,.4,1.8,.8,'样本2: 5.375',orange);arrow(axs[0],(3,2.9),(1.6,2),orange);arrow(axs[0],(3,.8),(1.6,1.6),orange);axs[0].text(2.5,3.65,'广播的反向：贡献相加 = 3.375',ha='center')
box(axs[1],3,1.3,1.8,1,'求和结果\n上游权重 c',orange);box(axs[1],.1,2.5,1.5,.8,'加数1: c',blue);box(axs[1],.1,.4,1.5,.8,'加数2: c',blue);arrow(axs[1],(3,2),(1.6,2.9),blue);arrow(axs[1],(3,1.6),(1.6,.8),blue);axs[1].text(2.5,3.65,'求和的反向：每项局部系数 1',ha='center');fig.tight_layout();save(fig,'05_broadcast.png')

N=np.array([1,2,4,8,16,32,64,128]);dense=8*(N*128)*(N*256);vectors=8*N*(128+256)
fig,ax=plt.subplots(figsize=(8.3,3.7));ax.loglog(N,dense/2**20,'o-',color=orange,label='稠密输入雅可比');ax.loglog(N,vectors/2**20,'s-',color=blue,label='一次乘积的输入与输出数组');ax.set(xlabel='批次数 N',ylabel='存储量 MiB（公式值）',title='只比较这两类数组；不包含前向缓存或框架内存');ax.set_xticks(N,[str(v) for v in N]);ax.grid(alpha=.2);ax.legend();ax.annotate('N=32: 256 MiB',(32,256),xytext=(9,1400),arrowprops={'arrowstyle':'->'},fontsize=10);ax.annotate('N=32: 96 KiB',(32,.09375),xytext=(9,.012),arrowprops={'arrowstyle':'->'},fontsize=10);fig.tight_layout();save(fig,'06_memory.png')

rows=list(csv.DictReader((BASE/'outputs/difference_scan.csv').open()));good=[r for r in rows if r['status']=='computed']
fig,ax=plt.subplots(figsize=(8.3,3.5));ax.loglog([float(r['h']) for r in good],[float(r['max_absolute_error']) for r in good],'o-',color=blue);ax.set(xlabel='差分步长 h',ylabel='13 个坐标最大绝对误差',title='解析梯度与中心差分的实测偏差');ax.grid(alpha=.2);ax.text(.03,.92,'h=1e-17：输入无法分辨，明确拒绝',transform=ax.transAxes,color=orange);fig.tight_layout();save(fig,'07_difference.png')
