"""由真实outputs和明示数学示例重建原创彩色图，不下载素材。"""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-dl-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
font=FontProperties(fname=os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
C=['#2563eb','#e76f51','#10a380','#9b5de5']
R=json.loads((ROOT/'outputs/results.json').read_text())
F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(name):
    plt.tight_layout();plt.savefig(F/name,dpi=170,bbox_inches='tight');plt.close()
def flow(labels,name):
    fig,ax=plt.subplots(figsize=(10,2.3));ax.axis('off')
    for i,s in enumerate(labels):
        xx=.12+i*.255
        ax.text(xx,.52,s,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.65',fc=['#dbeafe','#d1fae5','#ffedd5','#ede9fe'][i],ec=C[i]))
        if i<3:ax.annotate('',xy=(xx+.17,.52),xytext=(xx+.08,.52),arrowprops=dict(arrowstyle='->',color='#475569',lw=2))
    save(name)

A=np.load(ROOT/'outputs/trajectories.npz')
flow(['平滑：限制曲率\nL给出安全步长','凸性：切线在下\n连接当前位置和最优点','距离平方望远镜\n累加消去中间项','最后一步界\n需损失单调'],'01_proof.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.6));t=np.arange(1,len(A['quad_loss']))
ax[0].semilogy(t,A['quad_loss'][1:],label='真实目标差',color=C[0]);ax[0].semilogy(t,A['quad_bound'],label='理论上界40/t',color=C[1]);ax[0].set(xlabel='更新次数t',ylabel='目标值（对数轴）');ax[0].legend()
for eta,label,c in [(.25,'稳定步长0.25',C[0]),(.6,'过大步长0.6',C[1])]:
    vals=A['quad_loss'] if eta==.25 else A['bad_loss'];ax[1].semilogy(vals,label=label,color=c)
ax[1].set(xlabel='更新次数t',ylabel='目标值');ax[1].legend();save('02_bound.png')
fig,ax=plt.subplots(figsize=(8,3.8));x=A['x'];y=A['y'];ax.scatter(x[:,0],x[:,1],c=np.where(y>0,C[0],C[1]),s=60)
xx=np.linspace(-2,2,50);ax.plot(xx,-2*xx,color=C[2],label='硬间隔边界：x0+0.5x1=0')
ax.set(xlim=(-2.5,2.5),ylim=(-2.5,2.5),xlabel='特征x0',ylabel='特征x1');ax.legend();save('03_margin.png')
fig,ax=plt.subplots(1,3,figsize=(11,3.3))
for i in range(3):
    m=A[f'run{i}_metrics'];t=np.arange(1,len(m));
    for panel,col,title in [(0,0,'logistic损失'),(1,1,'参数长度'),(2,2,'到最大间隔方向角度（度）')]:ax[panel].semilogx(t,m[1:,col],color=C[i],label='初值'+str(R['logistic']['runs'][i]['initial']));ax[panel].set(xlabel='更新次数t',ylabel=title)
ax[0].legend(fontsize=8);save('04_directions.png')
fig,ax=plt.subplots(figsize=(8,3.6))
for i in range(3):
    w=A[f'run{i}_weights'];ax.plot(w[:,0],w[:,1],color=C[i],label=str(R['logistic']['runs'][i]['initial']));ax.scatter(*w[0],c=C[i]);ax.scatter(*w[-1],c=C[i],marker='x')
xx=np.linspace(0,10,50);ax.plot(xx,.5*xx,'--',color='#64748b',label='最大间隔射线');ax.set(xlabel='w0',ylabel='w1');ax.legend(fontsize=8);save('05_paths.png')
