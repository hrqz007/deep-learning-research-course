"""Original vector geometry diagrams, with equal axis scales where angles matter."""
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle, Arc
import numpy as np

OUT=Path(__file__).resolve().parent/"figures";OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in p]
if fonts:
    font_manager.fontManager.addfont(fonts[0]);font=font_manager.FontProperties(fname=fonts[0]).get_name()
else:font="sans-serif"
plt.rcParams.update({"font.family":font,"font.size":12,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
B,G,O,R="#2563a6","#25806c","#d57b18","#bc4951"


def arrow(ax,end,color,label=None,start=(0,0),offset=(.1,.1)):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=17,color=color,lw=2))
    if label:ax.text(end[0]+offset[0],end[1]+offset[1],label,color=color,fontsize=11)


def geometry(ax,xlim,ylim):
    ax.set_aspect("equal",adjustable="box");ax.set(xlim=xlim,ylim=ylim,xlabel="第一坐标",ylabel="第二坐标")
    ax.axhline(0,color="#9ca3af",lw=.7);ax.axvline(0,color="#9ca3af",lw=.7);ax.grid(alpha=.15)


def save(name):
    plt.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white");plt.close()


fig,ax=plt.subplots(figsize=(7,5));geometry(ax,(-.5,4.6),(-.4,4.8))
arrow(ax,(2,1),B,"u=(2,1)");arrow(ax,(1,3),O,"v=(1,3)")
arrow(ax,(3,4),G,"u+v=(3,4)")
ax.plot([2,3],[1,4],"--",color=O,alpha=.65);ax.plot([1,3],[3,4],"--",color=B,alpha=.65)
save("01_vectors.png")

fig,ax=plt.subplots(figsize=(9,3.8));ax.set(xlim=(0,9),ylim=(0,3.8));ax.axis("off")
headers=["位置","u分量","v分量","对应乘积"]
for j,h in enumerate(headers):ax.text(1.1+1.65*j,3.1,h,ha="center",color=B)
for i,row in enumerate([[1,2,1,2],[2,1,3,3]]):
    for j,value in enumerate(row):ax.text(1.1+1.65*j,2.25-.7*i,str(value),ha="center")
ax.text(4.15,.55,"内积 = 2 + 3 = 5（标量）",ha="center",color=G,fontsize=15)
save("02_dot.png")

fig,ax=plt.subplots(figsize=(7.5,4));geometry(ax,(-.3,3.2),(-.3,1.9))
arrow(ax,(2,1),B,"长度 √5",offset=(.15,.05));ax.plot([0,2,2],[0,0,1],color=O,lw=2)
ax.text(1,-.2,"2",ha="center",color=O);ax.text(2.1,.45,"1",color=O)
ax.text(.55,1.6,"长度平方 = 2² + 1² = 5",color=G)
save("03_norm.png")

fig,ax=plt.subplots(figsize=(7,5.3));geometry(ax,(-.6,2.7),(-.3,3.5))
arrow(ax,(1,3),O,"v",offset=(.1,.05));arrow(ax,(2,1),B,"u",offset=(.1,0))
arrow(ax,(.5,1.5),G,"p=(0.5,1.5)",offset=(-.9,.15))
arrow(ax,(2,1),R,start=(.5,1.5));ax.text(1.15,1.5,"残差 r",color=R)
ax.add_patch(Arc((0,0),1,1,theta1=math.degrees(math.atan2(1,2)),theta2=math.degrees(math.atan2(3,1)),color="#5b6b7e"))
ax.text(.42,.52,"45°",fontsize=11)
ax.text(1.1,2.6,"r · v = 0",color=R)
save("04_projection.png")

fig,ax=plt.subplots(figsize=(9,4.5));geometry(ax,(-.4,5),(-.5,3))
arrow(ax,(4,2),O,"2u=(4,2)",offset=(.1,.05));arrow(ax,(2,1),B,"u=(2,1)",offset=(-.1,-.4))
ax.text(.35,2.65,"余弦(u,2u)=1     距离=√5",color=G)
ax.text(.4,2.2,"余弦(u,-u)=-1，可为负数",color=R)
save("05_cosine.png")

fig,ax=plt.subplots(figsize=(10,4.1));ax.set(xlim=(0,10),ylim=(0,4.1));ax.axis("off")
lines=[("输入","a=(1,2,-1)    b=(2,0,3)"),("对应乘积","2、0、-3；合计 -1"),("平方长度","a：6，b：13"),("投影系数与向量","-1/13；(-2/13, 0, -3/13)"),("残差与方向内积","30/13 - 30/13 = 0")]
for i,(label,value) in enumerate(lines):
    y=3.6-.68*i;ax.text(.2,y,label,color=B,va="center");ax.text(3.2,y,value,color=G if i==4 else "#18202a",va="center")
save("06_3d_table.png")

fig,axs=plt.subplots(1,2,figsize=(9,4.8))
for ax,scale,title in zip(axs,[1,2],["原坐标：余弦0","第二坐标乘2：余弦-0.6"]):
    geometry(ax,(-.5,2),(-2.4,2.4));arrow(ax,(1,scale),B,"s");arrow(ax,(1,-scale),O,"t",offset=(.1,-.05));ax.set_title(title,fontsize=12)
fig.tight_layout(w_pad=2);save("07_scaling.png")

fig,axs=plt.subplots(1,2,figsize=(10,4.7))
ax=axs[0];ax.imshow(np.eye(6),cmap="Blues",vmin=0,vmax=1)
ax.set(xticks=range(6),yticks=range(6),xticklabels=[str(i) for i in range(1,7)],yticklabels=["e"+str(i) for i in range(1,7)],xlabel="只显示前6个坐标",title="100维基向量的一小块")
for i in range(6):
    for j in range(6):ax.text(j,i,str(int(i==j)),ha="center",va="center",color="white" if i==j else "#415066")
ax=axs[1];geometry(ax,(-.45,1.5),(-.45,1.5));ax.scatter([1,0,0],[0,1,0],s=[80,80,120],color=[B,O,R])
ax.text(1.03,.03,"e1",color=B);ax.text(.05,1.04,"e2",color=O)
ax.annotate("e3至e100\n截断后重合",xy=(0,0),xytext=(.35,.45),arrowprops={"arrowstyle":"->","color":R},color=R,fontsize=11)
ax.set_title("只保留前两个坐标",fontsize=12)
fig.tight_layout(w_pad=3);save("08_high_dim.png")
