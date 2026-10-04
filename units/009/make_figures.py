"""Build nine original pedagogical figures. Matplotlib is a build-only dependency."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

HERE=Path(__file__).resolve().parent
OUT=HERE/"figures"; OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in p]
if not fonts:
    raise RuntimeError("Install Noto Sans CJK font before rebuilding figures")
font_manager.fontManager.addfont(fonts[0])
font=font_manager.FontProperties(fname=fonts[0]).get_name()
plt.rcParams.update({"font.family":font,"font.size":12,"axes.unicode_minus":False})
B,G,O,R="#2563a6","#25806c","#d57b18","#b84552"
GRAY="#596878"
Wc=np.array([[4.,3.],[0.,1.]])
bc=np.array([-2.,-.5])


def canvas(h=4):
    fig,ax=plt.subplots(figsize=(11,h)); ax.set(xlim=(0,11),ylim=(0,h)); ax.axis("off"); return fig,ax


def arrow(ax,a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=17,lw=1.8,color=color))


def array(ax,x,y,values,w=.7,h=.55,color=B,highlight=None):
    for i,row in enumerate(values):
        for j,value in enumerate(row):
            fill="#e6f0f8" if highlight and highlight(i,j) else "white"
            ax.add_patch(Rectangle((x+j*w,y-i*h),w,h,facecolor=fill,edgecolor=color,lw=1.4))
            ax.text(x+(j+.5)*w,y+(.5-i)*h,str(value),ha="center",va="center")


def box(ax,x,y,w,h,text,color=B):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.06",facecolor="#f3f7fa",edgecolor=color,lw=1.6))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=color)


def save(fig,name):
    fig.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white");plt.close(fig)


fig,ax=canvas(3.6)
array(ax,.4,2.3,[[1,2],[0,-1],[3,1]],w=.75,h=.55)
array(ax,4.2,2.3,[[1,2,-1],[2,0,1]],w=.72,h=.55)
array(ax,8.2,2.3,[[5,2,1],[-2,0,-1],[5,6,-2]],w=.7,h=.55)
ax.text(1.15,3.15,"X (3,2) · V",ha="center",color=G)
ax.text(5.28,3.15,"W1 (2,3) · R/V",ha="center",color=O)
ax.text(9.25,3.15,"Z1 (3,3) · R",ha="center",color=B)
ax.text(2.95,2.4,"@",fontsize=25,ha="center");ax.text(7.1,2.4,"=",fontsize=25,ha="center")
ax.text(1.1,.7,"3条样本",ha="center",color=G)
ax.text(5.3,.7,"输入2项 → 输出3项",ha="center",color=O)
ax.text(9.2,.7,"仍是同3条样本",ha="center",color=G)
ax.text(5.5,.1,"保留N，累加D，得到H： (N,D) @ (D,H) → (N,H)",ha="center",fontsize=12)
save(fig,"01_flow.png")

fig,ax=canvas(3.8)
array(ax,.5,2.1,[[1,2]],w=.8,h=.65,color=G,highlight=lambda i,j:True)
array(ax,4,2.65,[[1,2,-1],[2,0,1]],w=.8,h=.65,color=B,highlight=lambda i,j:j==1)
box(ax,8.1,1.9,2,.85,"通道B = 2",O)
arrow(ax,(2.35,2.4),(3.75,2.4));arrow(ax,(6.6,2.4),(7.85,2.4))
ax.text(1.3,3.3,"P01 一行",ha="center",color=G)
ax.text(5.2,3.55,"W1 的三列",ha="center",color=B)
ax.text(5.5,1.1,"固定第1行、第2列： 1 × 2 + 2 × 0 = 2",ha="center",color=O)
ax.text(5.5,.35,"沿输入j求和；输出索引n=1、k=2保留",ha="center",color=GRAY)
save(fig,"02_cell.png")

fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
for ax,title,unit in zip(axs,["输入：标准基与x=(1,2)","输出：Wc的行就是基像"],["V","mm"]):
    ax.axhline(0,color="#b9c2cb",lw=.8);ax.axvline(0,color="#b9c2cb",lw=.8);ax.grid(alpha=.18);ax.set_aspect("equal")
    ax.set_title(title,fontsize=13);ax.set_xlabel("第1坐标 / "+unit);ax.set_ylabel("第2坐标 / "+unit)
for v,col,label in [(np.array([1,0]),B,"e1"),(np.array([0,1]),G,"e2"),(np.array([1,2]),R,"x")]:
    arrow(axs[0],(0,0),v,col);axs[0].text(v[0]+.08,v[1]+.08,label,color=col)
    mapped=v@Wc;arrow(axs[1],(0,0),mapped,col)
axs[0].set(xlim=(-.4,2.3),ylim=(-.4,2.5))
axs[1].set(xlim=(-.7,6.2),ylim=(-.7,5.8))
axs[1].text(4.15,2.85,"e1Wc=(4,3)",color=B,fontsize=10)
axs[1].text(.15,1.1,"e2Wc=(0,1)",color=G,fontsize=10)
axs[1].text(2.1,5.22,"xWc=(4,5)",color=R,fontsize=10)
axs[1].plot([4,4],[3,5],"--",color=G)
fig.tight_layout();save(fig,"03_basis.png")

fig,ax=canvas(4)
array(ax,.75,2.65,[[1,2],[0,-1],[3,1]],w=.8,h=.65,color=G)
array(ax,6.75,2.25,[[1,0,3],[2,-1,1]],w=.9,h=.65,color=B)
ax.text(1.55,3.55,"X (3,2)",ha="center",color=G);ax.text(8.1,3.55,"X^T (2,3)",ha="center",color=B)
arrow(ax,(3,2.3),(6.2,2.3));ax.text(4.65,2.9,"交换两条轴",ha="center")
ax.text(1.55,.85,"行：P01、P02、P03\n列：旋钮1、旋钮2",ha="center",fontsize=11)
ax.text(8.1,.85,"行：旋钮1、旋钮2\n列：P01、P02、P03",ha="center",fontsize=11)
ax.text(5.5,.08,"原X[2,0]=3 → 转置X^T[0,2]=3；不是把6个数按原顺序重排",ha="center",fontsize=11,color=R)
save(fig,"04_transpose.png")

fig,axs=plt.subplots(1,2,figsize=(10.5,4.3))
for ax,shift,title in zip(axs,[np.zeros(2),bc],["线性：xWc","仿射：xWc + bc"]):
    for t in [-1,-.5,0,.5,1]:
        horizontal=np.array([[-1,t],[1,t]])@Wc+shift
        vertical=np.array([[t,-1],[t,1]])@Wc+shift
        ax.plot(horizontal[:,0],horizontal[:,1],color=B,alpha=.7)
        ax.plot(vertical[:,0],vertical[:,1],color=G,alpha=.7)
    ax.scatter(*shift,c=R,zorder=5,s=35)
    label="原点像 (0,0)" if np.all(shift==0) else "原点像 (-2,-0.5)"
    ax.annotate(label,xy=shift,xytext=(shift[0]+1,shift[1]-2),fontsize=10,color=R,arrowprops={"arrowstyle":"->","color":R})
    ax.axhline(0,c="#a6b1bc",lw=.6);ax.axvline(0,c="#a6b1bc",lw=.6)
    ax.set(xlim=(-7,5),ylim=(-5,5),xlabel="水平坐标 / mm",ylabel="竖直坐标 / mm",title=title)
    ax.set_aspect("equal");ax.grid(alpha=.12)
fig.tight_layout();save(fig,"05_affine_grid.png")

fig,ax=canvas(4.8)
array(ax,.7,2.85,[[6,0,1.5],[-1,-2,-.5],[6,4,-1.5]],w=.85,h=.65,color=G)
array(ax,6.9,2.85,[[6,3,2],[-4,-2,-3],[5.5,6.5,-1.5]],w=.85,h=.65,color=R)
ax.text(2,4.2,"正确：b1 (1,3) 按列共享",ha="center",color=G)
ax.text(8.2,4.2,"错误：b1 (3,1) 按行共享",ha="center",color=R)
for j,t in enumerate(["+1","-2","+0.5"]):ax.text(1.125+.85*j,3.68,t,ha="center",fontsize=11,color=G)
for i,t in enumerate(["整行+1","整行-2","整行+0.5"]):ax.text(10.05,3.175-.65*i,t,ha="center",fontsize=10,color=R)
ax.text(5.5,.92,"N=H=3 时，两者shape都为(3,3)；P01通道B：0 ≠ 3",ha="center",color=R)
ax.text(5.5,.3,"广播不识别“样本”或“通道”这些含义，程序员必须识别",ha="center",color=GRAY,fontsize=11)
save(fig,"06_bias.png")

fig,ax=canvas(4.4)
box(ax,.2,2.8,1.5,.8,"X\n(N,2)",G);box(ax,2.9,2.8,2,.8,"@W1 + b1\n2 → 3",B)
box(ax,6,2.8,1.3,.8,"U\n(N,3)",O);box(ax,8.4,2.8,2.2,.8,"@W2 + b2\n3 → 2",B)
for a,b in [((1.8,3.2),(2.8,3.2)),((5,3.2),(5.9,3.2)),((7.4,3.2),(8.3,3.2))]:arrow(ax,a,b)
box(ax,.2,1.15,1.5,.8,"X\n(N,2)",G);box(ax,3.2,1.15,4.1,.8,"Wc=W1W2，bc=b1W2+b2",B);box(ax,9,1.15,1.5,.8,"Y\n(N,2)",R)
arrow(ax,(1.8,1.55),(3.1,1.55));arrow(ax,(7.4,1.55),(8.9,1.55));arrow(ax,(9.75,2.7),(9.75,2.05))
ax.text(5.5,.4,"合并后 Wc：(2,2) · mm/V；bc：(1,2) · mm",ha="center",color=GRAY)
save(fig,"07_composition.png")

fig,axs=plt.subplots(1,2,figsize=(10.5,4.1))
paths=[[(1,1),(2,1),(2,3)],[(1,1),(1,2),(2,2)]]
for ax,pts,title,labs in zip(axs,paths,["先S，再T：xST","先T，再S：xTS"],[["S: 水平×2","T: 竖直加水平"],["T: 竖直加水平","S: 水平×2"]]):
    ax.set(xlim=(0,3.25),ylim=(0,3.65),xlabel="第1坐标",ylabel="第2坐标",title=title);ax.set_aspect("equal");ax.grid(alpha=.2)
    for a,b,col in [(pts[0],pts[1],B),(pts[1],pts[2],O)]:arrow(ax,a,b,col)
    for pt,lab in zip(pts,["输入","中间","输出"]):
        ax.scatter(*pt,color=R,s=22,zorder=5);ax.text(pt[0]+.08,pt[1]+.08,f"{lab}{pt}",fontsize=10)
    ax.text(.12,3.15,labs[0]+"\n"+labs[1],fontsize=10,color=GRAY)
fig.tight_layout();save(fig,"08_order.png")

fig,ax=plt.subplots(figsize=(10.5,3.6))
x=np.arange(3)
ax.bar(x-.18,[.25,0,.75],.34,color=B,label="W1[0,1] 增加0.25 R/V")
ax.bar(x+.18,[.25,.25,.25],.34,color=O,label="b1[1] 增加0.25 R")
for loc,val in zip(x-.18,[.25,0,.75]):ax.text(loc,val+.025,f"{val:g}",ha="center",fontsize=10)
for loc,val in zip(x+.18,[.25,.25,.25]):ax.text(loc,val+.025,f"{val:g}",ha="center",fontsize=10)
ax.set(xticks=x,xticklabels=["P01：x1=1 V","P02：x1=0 V","P03：x1=3 V"],ylim=(0,1.12),ylabel="通道B响应变化 / R")
ax.legend(loc="upper left",fontsize=10);ax.grid(axis="y",alpha=.15);fig.tight_layout();save(fig,"09_parameter.png")
print("generated 9 original figures")
