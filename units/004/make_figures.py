"""Original small-array visual explanations; matplotlib is a build-only extra."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures";OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in p]
if fonts:
    font_manager.fontManager.addfont(fonts[0]);font=font_manager.FontProperties(fname=fonts[0]).get_name()
else:font="sans-serif"
plt.rcParams.update({"font.family":font,"font.size":12,"axes.unicode_minus":False})
B,G,O,R="#2563a6","#25806c","#d57b18","#bc4951"


def canvas(height=4):
    fig,ax=plt.subplots(figsize=(10,height));ax.set(xlim=(0,10),ylim=(0,height));ax.axis("off");return ax


def array(ax,x,y,values,width=.85,height=.55,color=B):
    for i,row in enumerate(values):
        for j,value in enumerate(row):
            ax.add_patch(Rectangle((x+j*width,y-i*height),width,height,facecolor="white",edgecolor=color,lw=1.5))
            ax.text(x+(j+.5)*width,y+(.5-i)*height,str(value),ha="center",va="center",fontsize=12)


def arrow(ax,start,end,color="#718096"):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=15,color=color,lw=1.6))


def save(name):
    plt.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white");plt.close()


ax=canvas(3.7);array(ax,3.55,2.3,[[1,10],[2,20],[3,30]],1.25,.65)
ax.text(4.8,3.35,"特征轴 axis=1：两个旋钮",ha="center",color=O)
ax.text(1.9,1.9,"样本轴 axis=0\n三条记录",ha="center",color=G)
for i,s in enumerate(["A01","A02","A03"]):ax.text(7,2.62-.65*i,s,color=G)
ax.text(4.8,.45,"shape = (3, 2)        size = 6",ha="center")
save("01_axes.png")

ax=canvas(3.9);array(ax,.7,1.9,[[1,2,3]],.8,.7);array(ax,5.6,2.6,[[1],[2],[3]],1,.6)
ax.text(1.9,3.15,'X[:, 0]  →  shape (3,)',ha="center",color=B)
ax.text(6.1,3.5,'X[:, 0:1]  →  shape (3, 1)',ha="center",color=O)
ax.text(1.9,1.05,"只有一条轴，不自带行列方向",ha="center",fontsize=10)
ax.text(6.1,.85,"两条轴，第二条长度为1",ha="center",fontsize=10)
ax.text(5,.15,"三个数相同，后续广播的对齐方式却可能不同",ha="center",color=R)
save("02_keep_axes.png")

ax=canvas(4);array(ax,.4,2.55,[[1,10],[2,20],[3,30]],.8,.6);array(ax,3.2,2.55,[[2,1],[4,2],[6,3]],.8,.6);array(ax,6,2.55,[[3],[6],[9]],.8,.6);array(ax,8.55,2.55,[[4],[7],[10]],.8,.6)
for x,s in [(1.2,"X (3,2)"),(4,"贡献 (3,2)"),(6.4,"合计 (3,)"),(8.95,"预测 (3,)")]:ax.text(x,3.45,s,ha="center",fontsize=11)
for start,end in [(2.05,3.1),(4.85,5.9),(6.85,8.45)]:arrow(ax,(start,2.15),(end,2.15))
ax.text(2.55,1.2,"乘两项系数",ha="center",fontsize=10,color=O)
ax.text(5.4,1.2,"沿特征求和",ha="center",fontsize=10,color=G)
ax.text(7.65,1.2,"每条加1",ha="center",fontsize=10,color=B)
ax.text(5,.35,"图中竖排只为便于对应；合计与预测实际采用一维 shape (3,)",ha="center",fontsize=10,color=R)
save("03_vectorize.png")

ax=canvas(4.3);array(ax,3.55,2.8,[[2,1],[4,2],[6,3]],.85,.6)
for i,s in enumerate([3,6,9]):arrow(ax,(5.35,3.1-.6*i),(6.3,3.1-.6*i),G);ax.text(6.6,3.1-.6*i,str(s),ha="center",va="center",color=G)
for j,s in enumerate([12,6]):arrow(ax,(3.98+.85*j,1.5),(3.98+.85*j,.9),O);ax.text(3.98+.85*j,.55,str(s),ha="center",color=O)
ax.text(8,2.3,"axis=1\n每条样本",ha="center",color=G)
ax.text(1.5,1.35,"axis=0\n每个特征",ha="center",color=O)
ax.text(4.4,3.85,"同一组逐项贡献",ha="center")
save("04_reduction.png")

ax=canvas(4.5);values=[[-1,-3,-5],[2,0,-2],[5,3,1]]
array(ax,3.4,2.8,values,1,.7,color=R)
for i in range(3):ax.add_patch(Rectangle((3.4+i,2.8-.7*i),1,.7,facecolor="#d8eee8",edgecolor=G,lw=2.1));ax.text(3.9+i,3.15-.7*i,str(values[i][i]),ha="center",va="center")
for j,s in enumerate([5,7,9]):ax.text(3.9+j,3.8,f"标签{s}",ha="center",color=O)
for i,s in enumerate([4,7,10]):ax.text(2.5,3.15-.7*i,f"预测{s}",ha="center",color=B)
ax.text(7.9,2.5,"绿色对角线\n才是同样本配对",ha="center",color=G)
ax.text(5,.65,"逐样本：2 ÷ 3     错误两两平均：22 ÷ 9",ha="center",color=R)
save("05_broadcast_bug.png")

ax=canvas(4.3);array(ax,.25,2.7,[[1,10],[2,20],[3,30]],.75,.6);array(ax,3.65,2.7,[[2,-1],[.1,.2]],.9,.6);array(ax,7.4,2.7,[[3,1],[6,2],[9,3]],.8,.6)
ax.text(2.7,2.7,"@",fontsize=24,ha="center");ax.text(6.25,2.7,"=",fontsize=24,ha="center")
for x,s in [(1,"X (3,2)"),(4.55,"W (2,2)"),(8.2,"输出 (3,2)")]:ax.text(x,3.65,s,ha="center")
ax.text(5,.8,"第一行第一列：1 × 2 + 10 × 0.1 = 3",ha="center",color=G)
ax.text(5,.25,"第一行第二列：1 × (-1) + 10 × 0.2 = 1",ha="center",color=O)
save("06_matmul.png")

ax=canvas(4.1)
for n,x in [(0,.8),(1,5.6)]:
 for c,col in [(2,O),(1,G),(0,B)]:
  ax.add_patch(Rectangle((x+c*.27,1.2+c*.24),2.2,1.55,facecolor="white",edgecolor=col,lw=2))
  if c==0:
   ax.plot([x+1.1,x+1.1],[1.2,2.75],color=col,lw=1)
   ax.plot([x,x+2.2],[1.98,1.98],color=col,lw=1)
 ax.text(x+1.4,3.7,f"样本 {n}：3个通道",ha="center")
ax.text(5,.55,"NCHW (2,3,2,2)  →  NHWC (2,2,2,3)",ha="center",color=R)
ax.text(5,.05,"轴换位置，样本、通道与像素的对应身份不变",ha="center",fontsize=11)
save("07_channels.png")
