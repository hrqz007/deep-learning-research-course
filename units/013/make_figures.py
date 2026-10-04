"""Nine original, deterministic explanatory figures; CPU and local data only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p]
if not fonts: raise RuntimeError('Noto Sans CJK font is required for Chinese figures')
font_manager.fontManager.addfont(fonts[0])
plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=fonts[0]).get_name(), 'DejaVu Sans'], 'font.size':11, 'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})
B,G,R,O='#2466a8','#24816c','#b94258','#c88620'
def save(fig,name):
 fig.tight_layout(pad=1.5); fig.savefig(OUT/name,dpi=175,facecolor='white'); plt.close(fig)
def arrow(ax,v,color,label):
 ax.annotate('',xy=v,xytext=(0,0),arrowprops={'arrowstyle':'-|>','color':color,'lw':2.5})
 ax.annotate(label,xy=v,xytext=(6,5),textcoords='offset points',color=color)
fig,axs=plt.subplots(1,2,figsize=(10.5,3.8))
for ax in axs:
 ax.axhline(0,c='#d0d8e0',lw=.8);ax.axvline(0,c='#d0d8e0',lw=.8);ax.set(xlim=(-1,3),ylim=(-1,5),xlabel='第一坐标',ylabel='第二坐标');ax.grid(alpha=.15)
arrow(axs[0],(1,2),B,'a');arrow(axs[0],(2,4),R,'b=2a');axs[0].set_title('线性相关：只有一个自由方向')
arrow(axs[1],(1,2),B,'a');arrow(axs[1],(0,1),G,'c');axs[1].set_title('a与c无关：张成二维平面')
for a in np.linspace(-1,2,7): axs[1].plot([a,a],[2*a-2,2*a+2],c=G,alpha=.18)
save(fig,'01_dependence.png')
fig,axs=plt.subplots(1,3,figsize=(11,3.7));t=np.linspace(0,2*np.pi,360);circle=np.c_[np.cos(t),np.sin(t)]
U=np.array([[1,1],[1,-1]])/np.sqrt(2);S=np.diag([4,2]);final=circle@U@S@U.T
for ax,pts,title in zip(axs,[circle,circle@U@S,final],['输入单位圆','奇异坐标：伸缩4与2','输出前两维：椭圆']):
 ax.plot(pts[:,0],pts[:,1],c=B,lw=2);ax.axhline(0,c='#cdd5df',lw=.8);ax.axvline(0,c='#cdd5df',lw=.8);ax.set_aspect('equal');ax.set(title=title,xlabel='坐标1',ylabel='坐标2',xlim=(-4.7,4.7),ylim=(-4.7,4.7));ax.grid(alpha=.15)
arrow(axs[0],U[:,0],G,'u₁');arrow(axs[0],U[:,1],R,'u₂');arrow(axs[2],4*U[:,0],G,'4v₁');arrow(axs[2],2*U[:,1],R,'2v₂')
save(fig,'02_ellipse.png')
fig,axs=plt.subplots(1,3,figsize=(10.8,3));parts=[np.array([[2,2,0],[2,2,0]]),np.array([[1,-1,0],[-1,1,0]]),np.array([[3,1,0],[1,3,0]])]
for ax,A,title in zip(axs,parts,['4u₁v₁ᵀ，秩1','2u₂v₂ᵀ，秩1','相加得W，秩2']):
 ax.imshow(A,vmin=-3,vmax=3,cmap='RdBu');ax.set_title(title);ax.set_xticks([0,1,2],['输出1','输出2','输出3']);ax.set_yticks([0,1],['输入1','输入2'])
 for (i,j),v in np.ndenumerate(A):ax.text(j,i,str(v),ha='center',va='center',fontsize=17,color='white' if abs(v)>1.5 else 'black')
save(fig,'03_components.png')
fig,axs=plt.subplots(1,2,figsize=(10.8,3.6))
for s,color,label in [(np.array([8,2,.5,.125]),B,'快衰减'),(np.array([4,3,2,1]),R,'慢衰减')]:
 axs[0].plot(range(1,5),s,'o-',c=color,label=label);errs=[np.sqrt(np.sum(s[k:]**2)) for k in range(5)];axs[1].plot(range(5),errs,'o-',c=color,label=label)
for ax in axs:ax.grid(alpha=.2);ax.legend()
axs[0].set(xlabel='方向编号i',ylabel='奇异值σᵢ',title='两种合成奇异值谱',xticks=[1,2,3,4]);axs[1].set(xlabel='保留方向数k',ylabel='Frobenius误差',title='尾部平方和开根',xticks=list(range(5)))
save(fig,'04_spectrum.png')
fig,axs=plt.subplots(1,2,figsize=(10.5,3.4))
for ax in axs:ax.set(xlim=(9.5,10.5),ylim=(-.16,.16),xlabel='第一输出（共同为10）',ylabel='第二输出');ax.grid(alpha=.2)
axs[0].scatter([10,10],[.1,-.1],s=95,c=[B,R]);axs[0].annotate('标签 +',(10,.1),xytext=(10.12,.1));axs[0].annotate('标签 −',(10,-.1),xytext=(10.12,-.1));axs[0].set_title('完整输出可区分两类')
axs[1].scatter([10],[0],s=130,c=O);axs[1].annotate('两类都在(10,0)',(10,0),xytext=(9.61,.07),arrowprops={'arrowstyle':'->'});axs[1].set_title('相对矩阵误差不足1%，两点仍重合')
save(fig,'05_task_loss.png')
fig,axs=plt.subplots(1,2,figsize=(10.8,3.6));eps=np.array([1,.01,.0001,.000001]);delta=1e-6
axs[0].loglog(eps,np.full(4,delta),'o-',c=B,label='右端相对变化');axs[0].loglog(eps,delta/eps,'o-',c=R,label='解相对变化');axs[0].invert_xaxis();axs[0].set(xlabel='ε从大到小',ylabel='相对变化',title='相同扰动，弱方向越来越敏感');axs[0].legend();axs[0].grid(alpha=.2)
axs[1].bar(['第一方向','第二方向'],[1,1e6],color=[B,R]);axs[1].set_yscale('log');axs[1].set(ylabel='相对误差放大倍数',title='ε=10⁻⁶时的方向差异');axs[1].grid(axis='y',alpha=.2)
save(fig,'06_perturbation.png')
fig,ax=plt.subplots(figsize=(8,3.5));ax.semilogy([1,2],[1,1e-8],'o-',c=B,lw=2);ax.axhline(1e-6,c=R,ls='--',label='τ=10⁻⁶：数值秩1');ax.axhline(1e-10,c=G,ls='--',label='τ=10⁻¹⁰：数值秩2');ax.set(xticks=[1,2],xlabel='奇异值编号',ylabel='奇异值或阈值',title='diag(1,10⁻⁸)的数学秩始终为2',ylim=(1e-11,10));ax.legend(loc='upper right');ax.grid(alpha=.2);save(fig,'07_threshold.png')
fig,axs=plt.subplots(1,2,figsize=(10.8,3.7));x=np.linspace(-2,2,300);y=np.linspace(-15,15,400);X,Y=np.meshgrid(x,y)
for ax,e in zip(axs,[1,.1]):
 C=ax.contour(X,Y,.5*(X*X+e*e*Y*Y),levels=[.125,.5,1],colors=[G,B,R]);limit=1.7 if e==1 else 15;ax.set(xlabel='z₁',ylabel='z₂',title=f'ε={e:g}；等比例坐标',xlim=(-limit,limit),ylim=(-limit,limit));ax.set_aspect('equal');ax.grid(alpha=.15)
save(fig,'08_landscape.png')
fig,ax=plt.subplots(figsize=(11,3.5));ax.set(xlim=(0,11),ylim=(0,4));ax.axis('off')
boxes=[(.7,2,'x\n(1,d)'),(3,3,'原权重 W₀\n(d,h)'),(3,1,'A\n(d,r)'),(6,1,'B\n(r,h)'),(8.4,2,'相加'),(10.2,2,'输出\n(1,h)')]
for x,y,lab in boxes:
 ax.add_patch(FancyBboxPatch((x-.65,y-.42),1.3,.84,boxstyle='round,pad=.05',fc='#eef4fa',ec=B));ax.text(x,y,lab,ha='center',va='center')
for a,b in [((1.4,2.2),(2.3,3)),((1.4,1.8),(2.3,1)),((3.7,1),(5.3,1)),((6.7,1),(7.7,1.8)),((3.7,3),(7.7,2.2)),((9.1,2),(9.5,2))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':G,'lw':2})
ax.text(4.5,.25,'中间只有r个数',ha='center',color=G);ax.set_title('低秩的是增量AB，原权重W₀可为满秩');save(fig,'09_adapter.png')
