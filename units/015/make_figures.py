"""Ten original figures for the quadratic optimization geometry lesson."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiment import trajectory
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'figures'; OUT.mkdir(exist_ok=True)
fonts = [p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p]
if not fonts: raise RuntimeError('Noto Sans CJK is required')
font_manager.fontManager.addfont(fonts[0])
plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=fonts[0]).get_name(), 'DejaVu Sans'], 'font.size':11, 'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})
B,G,R,O = '#2466a8','#24816c','#b94258','#c88620'
H=np.array([[13.,-12.],[-12.,13.]])
Q=np.array([[1,1],[1,-1]])/np.sqrt(2)
def save(fig,name):
    fig.tight_layout(pad=1.3); fig.savefig(OUT/name,dpi=175,facecolor='white'); plt.close(fig)
def grid(ax): ax.grid(alpha=.18)
def contours(ax,limit=2.5):
    a=np.linspace(-limit,limit,400); X,Y=np.meshgrid(a,a)
    C=ax.contour(X,Y,.5*(13*X*X-24*X*Y+13*Y*Y),levels=[.25,1,4,16,40,80],colors='#a8bdcc',linewidths=.9)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel='x',ylabel='y');ax.set_aspect('equal');grid(ax)

fig,axs=plt.subplots(1,2,figsize=(10.7,3.6))
for ax,width in zip(axs,[.35,1.1]):
    h=np.linspace(-width,width,300);ax.plot(1+h,(1+h)**4,c=B,label='真实 x⁴');ax.plot(1+h,1+4*h,c=R,ls='--',label='一阶');ax.plot(1+h,1+4*h+6*h*h,c=G,ls='-.',label='二阶');ax.scatter([1],[1],c='black',s=25,zorder=4);ax.set(xlabel='x',ylabel='函数或模型值',title='基点附近' if width<1 else '范围扩大后的偏离');ax.legend();grid(ax)
save(fig,'01_taylor.png')

fig,axs=plt.subplots(1,2,figsize=(10.7,3.9))
contours(axs[0],2.3)
for vec,c,label in [(Q[:,0],G,'λ₁=1'),(Q[:,1],R,'λ₂=25')]:
    axs[0].annotate('',xy=1.7*vec,xytext=(0,0),arrowprops={'arrowstyle':'-|>','lw':2.5,'color':c});axs[0].text(*(1.85*vec),label,color=c,ha='center')
axs[0].set_title('主方向对齐长轴和短轴')
t=np.linspace(0,2*np.pi,300);curv=13-24*np.cos(t)*np.sin(t)
axs[1].plot(t*180/np.pi,curv,c=B);axs[1].scatter([45,135],[1,25],c=[G,R],zorder=3);axs[1].set(xlabel='单位方向角度（度）',ylabel='uHuᵀ',title='方向曲率在1与25之间',xticks=[0,45,90,135,180,270,360]);grid(axs[1]);save(fig,'02_directions.png')

fig,axs=plt.subplots(1,2,figsize=(10.7,3.6));hs=np.array([.2,.1,.05,.025]);actual=1.5+6*hs+8*hs**2+4*hs**3+hs**4
axs[0].plot(hs,actual,'o-',c=B,label='真实值');axs[0].plot(hs,1.5+6*hs,'s--',c=R,label='一阶');axs[0].plot(hs,1.5+6*hs+8*hs**2,'^-.',c=G,label='二阶');axs[0].set(xlabel='h',ylabel='F(1+h,1+2h)',title='同一条增量路径');axs[0].legend();grid(axs[0])
axs[1].loglog(hs,8*hs**2+4*hs**3+hs**4,'s-',c=R,label='一阶误差');axs[1].loglog(hs,4*hs**3+hs**4,'^-',c=G,label='二阶误差');axs[1].set(xlabel='h（对数刻度）',ylabel='绝对误差（对数刻度）',title='本例可由多项式精确核算');axs[1].legend();grid(axs[1]);save(fig,'03_errors.png')

fig,axs=plt.subplots(1,3,figsize=(11,3.5));a=np.linspace(-1,1,200);X,Y=np.meshgrid(a,a)
for ax,Z,title in zip(axs,[X*X+Y*Y,-X*X-Y*Y,X*X-Y*Y],['严格最小：x²+y²','严格最大：−x²−y²','鞍点：x²−y²']):
    ax.contourf(X,Y,Z,levels=np.linspace(-2,2,17),cmap='RdBu_r');ax.contour(X,Y,Z,levels=[-.75,-.25,.25,.75],colors='#666666',linewidths=.6);ax.scatter([0],[0],c='black',s=18);ax.axhline(0,c='black',lw=.6);ax.axvline(0,c='black',lw=.6);ax.set(xlabel='x',ylabel='y',title=title);ax.set_aspect('equal')
fig.colorbar(axs[0].collections[0],ax=axs.tolist(),label='函数值',fraction=.025,pad=.03) if False else None
save(fig,'04_stationary.png')

fig,axs=plt.subplots(1,2,figsize=(10.6,3.5));a=np.linspace(-1.25,1.25,350)
for ax,vals,title,end in [(axs[0],a*a,'凸函数：弦不低于曲线',1),(axs[1],a*a-3*a**4+a**6,'非凸：中心高于端点的弦',-1)]:
    ax.plot(a,vals,c=B);ax.plot([-1,1],[end,end],c=R,ls='--',label='端点连线');ax.scatter([-1,0,1],[end,0,end],c=[R,G,R],zorder=4);ax.set(xlabel='x（固定y=0）',ylabel='函数值',title=title);ax.legend();grid(ax)
save(fig,'05_convex.png')

fig,ax=plt.subplots(figsize=(8.5,3.7));eta=np.linspace(0,.105,300)
ax.axhspan(-1,1,color='#eaf5ef');ax.axhline(1,c='#777777',ls='--');ax.axhline(-1,c='#777777',ls='--');ax.axvline(.08,c=O,ls=':',label='边界0.08');ax.plot(eta,1-eta,c=G,lw=2,label='温和方向 1−η');ax.plot(eta,1-25*eta,c=R,lw=2,label='陡峭方向 1−25η');ax.set(xlabel='步长η',ylabel='每步乘数',title='严格稳定要求两个乘数的绝对值都小于1',xlim=(0,.105));ax.legend(loc='lower left');grid(ax);save(fig,'06_factors.png')

fig,axs=plt.subplots(1,3,figsize=(11.5,3.65))
for ax,eta,n,limit in zip(axs,[.02,.075,.09],[20,20,6],[2.3,2.3,5.2]):
    contours(ax,limit);r=trajectory([[2,0]],H,eta,n);x=[v['x'] for v in r];y=[v['y'] for v in r];ax.plot(x,y,'o-',ms=3,c=R);ax.scatter([0],[0],marker='*',c=G,s=90);ax.annotate('0',(x[0],y[0]),xytext=(4,6),textcoords='offset points');ax.annotate(str(n),(x[-1],y[-1]),xytext=(4,6),textcoords='offset points');ax.set_title(f'η={eta:g}，显示{n}步')
save(fig,'07_trajectories.png')

fig,axs=plt.subplots(1,2,figsize=(10.8,3.7))
for eta,c in zip([.02,.04,.075,.08,.09],[B,G,O,'#8156a3',R]):
    r=trajectory([[2,0]],H,eta,60);axs[0].semilogy([v['iteration'] for v in r],[v['loss'] for v in r],c=c,label=f'η={eta:g}')
axs[0].set(xlabel='迭代k',ylabel='损失（对数刻度）',title='下降 平台 发散来自同一个H');axs[0].legend(fontsize=9);grid(axs[0])
r=trajectory([[2,0]],H,.08,20);p=np.array([[v['x'],v['y']] for v in r]);z=p@Q
axs[1].plot(range(21),z[:,0],'o-',c=G,label='z₁ 衰减');axs[1].plot(range(21),z[:,1],'o-',c=R,label='z₂ 等幅翻转');axs[1].set(xlabel='迭代k',ylabel='主方向坐标',title='η=0.08：参数没有收敛');axs[1].legend();grid(axs[1]);save(fig,'08_loss_modes.png')

fig,axs=plt.subplots(1,2,figsize=(10.6,4));contours(axs[0],2.5);axs[0].plot([2,0],[0,0],'o-',c=G);axs[0].set_title('映回θ：一步从(2,0)到0')
a=np.linspace(-8,8,350);X,Y=np.meshgrid(a,a);axs[1].contour(X,Y,.5*(X*X+Y*Y),levels=[1,4,16,26,40],colors='#a8bdcc');axs[1].plot([np.sqrt(2),0],[5*np.sqrt(2),0],'o-',c=G);axs[1].set(xlabel='w₁=z₁',ylabel='w₂=5z₂',title='w坐标：曲率都为1，η=1',xlim=(-8,8),ylim=(-8,8));axs[1].set_aspect('equal');grid(axs[1]);save(fig,'09_scaling.png')

fig,axs=plt.subplots(1,2,figsize=(10.7,3.6));x=np.linspace(-1,0,250)
axs[0].plot(x,x+.5*x*x+10*x**4,c=B,label='真实R(x,0)');axs[0].plot(x,x+.5*x*x,c=G,ls='--',label='原点二阶模型');axs[0].scatter([-1],[9.5],c=R);axs[0].set(xlabel='x',ylabel='函数或模型值',title='原点H=I仍不能保护大步长');axs[0].legend();grid(axs[0])
axs[1].plot(x,1+120*x*x,c=R);axs[1].axhline(1,c=G,ls='--',label='仅原点的曲率1');axs[1].set(xlabel='x（更新线段−1到0）',ylabel='x方向二阶导数',title='曲率沿更新线段迅速增大');axs[1].legend();grid(axs[1]);save(fig,'10_local_bound.png')
print('10 original figure files written')
