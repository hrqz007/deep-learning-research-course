"""Rebuild nine original explanatory figures, using local data and CPU only."""
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import experiment as lab

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
fonts = [p for p in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in p]
if not fonts:
    raise RuntimeError("Install Noto Sans CJK fonts to rebuild Chinese figures")
font_manager.fontManager.addfont(fonts[0])
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=fonts[0]).get_name(),
                     "font.size": 11.5, "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False})
B, G, R, O, GRAY = "#2466a8", "#24816c", "#bc4156", "#d58a23", "#687585"


def save(fig, name):
    fig.tight_layout(pad=1.6)
    fig.savefig(OUT / name, dpi=170, facecolor="white")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
x = np.linspace(1.5, 2.5, 150)
y = np.linspace(2.5, 3.5, 150)
for ax, value, curve, tangent, xlabel, title in [
    (axes[0], x, 16*x*x, 64+64*(x-2), "x（y固定为3）", "只改变x：斜率64"),
    (axes[1], y, 4*(y+1)**2, 64+32*(y-3), "y（x固定为2）", "只改变y：斜率32")]:
    ax.plot(value, curve, color=B, lw=2.4, label="真实切片")
    ax.plot(value, tangent, color=O, ls="--", lw=2, label="局部切线")
    ax.scatter([2 if ax is axes[0] else 3], [64], c=R, zorder=5)
    ax.set(xlabel=xlabel, ylabel="损失L", title=title, ylim=(25, 105))
    ax.grid(alpha=.15); ax.legend(fontsize=10)
save(fig, "01_slices.png")

fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
e = np.linspace(0, .15, 120)
remainder = 96*e**2 + 32*e**3 + 4*e**4
axes[0].plot(e, 128*e+remainder, c=B, lw=2.4, label="真实变化")
axes[0].plot(e, 128*e, c=O, ls="--", lw=2, label="线性预测")
axes[0].set(xlabel="ε，增量为(ε,2ε)", ylabel="损失变化", title="两者在原点接近")
axes[0].legend(fontsize=10)
es = np.array([.1, .01, .001, .0001])
rs = 96*es**2+32*es**3+4*es**4
axes[1].loglog(es, rs/(np.sqrt(5)*es), "o-", c=G)
axes[1].set(xlabel="ε", ylabel="|余项| / 增量长度", title="相对一阶尺度的余项趋小")
for ax in axes: ax.grid(alpha=.2)
save(fig, "02_linear.png")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
xs, ys = np.linspace(.4, 4.5, 280), np.linspace(.5, 4.6, 280)
X, Y = np.meshgrid(xs, ys)
C = ax.contour(X, Y, X*X*(Y+1)**2, levels=[8, 16, 32, 64, 96, 144], colors=B, linewidths=1.1)
ax.clabel(C, inline=True, fontsize=10)
p = np.array([2, 3.])
for v, color, label in [(np.array([2, 1])/np.sqrt(5), G, "梯度方向"),
                         (-np.array([2, 1])/np.sqrt(5), R, "负梯度方向"),
                         (np.array([-1, 2])/np.sqrt(5), GRAY, "等高线切向")]:
    q=p+.65*v
    ax.annotate("", xy=q, xytext=p, arrowprops={"arrowstyle":"-|>", "color":color, "lw":2.5})
    offset = {"梯度方向": (.12, .15), "负梯度方向": (-.9, -.25), "等高线切向": (-.5, .18)}[label]
    ax.text(q[0]+offset[0], q[1]+offset[1], label, color=color, fontsize=11)
ax.scatter(*p,c="black",s=35,zorder=6); ax.text(2.15,2.55,"p=(2,3)，L=64",fontsize=10)
ax.set(xlabel="x",ylabel="y",title="损失等高线  L=x²(y+1)²",xlim=(.4,4.5),ylim=(.5,4.6)); ax.set_aspect("equal")
save(fig,"03_contours.png")

fig,axes=plt.subplots(1,2,figsize=(10.8,3.7))
ax=axes[0]; angles=np.linspace(0,2*np.pi,240)
ax.plot(np.cos(angles),np.sin(angles),c="#cbd4de",lw=1)
vecs=[(np.array([.6,.8]),B,"u"),(np.array([-1,2])/np.sqrt(5),GRAY,"切向"),(-np.array([2,1])/np.sqrt(5),R,"最陡下降")]
for v,c,name in vecs:
    ax.annotate("",xy=v,xytext=(0,0),arrowprops={"arrowstyle":"-|>","lw":2.2,"color":c})
    ax.text(v[0]*1.13,v[1]*1.13,name,ha="center",color=c)
ax.axhline(0,c="#dce2e8",lw=.8);ax.axvline(0,c="#dce2e8",lw=.8)
ax.set(xlim=(-1.4,1.4),ylim=(-1.35,1.35),xlabel="x方向分量",ylabel="y方向分量",title="固定长度为1");ax.set_aspect("equal")
vals=[64,0,-32*np.sqrt(5)]
axes[1].barh(["u=(3/5,4/5)","单位切向","负梯度单位方向"],vals,color=[B,GRAY,R],height=.5)
axes[1].axvline(0,c=GRAY,lw=.8); axes[1].set(xlim=(-92,88),xlabel="方向导数",title="同一梯度(64,32)")
for i,v in enumerate(vals):
    axes[1].text(v+3 if v>=0 else v/2, i, f"{v:.2f}", ha="left" if v>=0 else "center", va="center", color="black" if v>=0 else "white")
save(fig,"04_directions.png")

fig,ax=plt.subplots(figsize=(10.8,3.8));ax.set(xlim=(0,11),ylim=(0,4));ax.axis("off")
positions={"x":(.8,2.7),"y":(.8,.8),"a":(4,2),"z":(7,2),"L":(10,2)}
for name,(x,y) in positions.items():
    label={"x":"x = 2","y":"y = 3","a":"a = xy\n值为6","z":"z = a+x\n值为8","L":"L = z²\n值为64"}[name]
    ax.add_patch(FancyBboxPatch((x-.6,y-.43),1.2,.86,boxstyle="round,pad=.06",fc="#f0f5fa",ec=B,lw=1.5))
    ax.text(x,y,label,ha="center",va="center",fontsize=12)
for start,end,label,color,offset in [("x","a","y=3",B,.18),("y","a","x=2",GRAY,-.25),("a","z","1",B,.2),("z","L","2z=16",B,.2)]:
    s=np.array(positions[start]); e=np.array(positions[end]);sv=s+np.array([.66,0]);ev=e-np.array([.66,0])
    ax.annotate("",xy=ev,xytext=sv,arrowprops={"arrowstyle":"-|>","color":color,"lw":2})
    mid=(sv+ev)/2;ax.text(mid[0],mid[1]+offset,label,ha="center",color=color)
ax.add_patch(FancyArrowPatch((1.05,3.2),(7.0,2.51),connectionstyle="arc3,rad=-.22",arrowstyle="-|>",mutation_scale=17,lw=2.3,color=R))
ax.text(4.9,3.82,"x的直接路径：局部系数1",ha="center",color=R)
ax.text(5.5,.25,"x贡献：48 + 16 = 64        y贡献：32",ha="center",fontsize=14,color=G)
save(fig,"05_graph.png")

fig,axes=plt.subplots(1,2,figsize=(10.8,3.5))
s=np.linspace(-.3,.3,160)
for ax,c,curve,label in [(axes[0],B,(2+s)**2*16,"固定y=3，x=2+s"),(axes[1],G,(2+s)**2*(4+s)**2,"x=2+s，y=3+s")]:
    slope=64 if ax is axes[0] else 96
    ax.plot(s,curve,c=c,lw=2.5,label="实际损失")
    ax.plot(s,64+s*slope,c=O,ls="--",label=f"起点斜率{slope}")
    ax.scatter([0],[64],c=R,zorder=5);ax.set(xlabel="s（从当前点出发的路径参数）",ylabel="L",title=label,ylim=(35,100));ax.legend(fontsize=10);ax.grid(alpha=.15)
save(fig,"06_routes.png")

lab.run(ROOT/"outputs")
rows=list(csv.DictReader((ROOT/"outputs/difference_scan.csv").open()))
valid=[r for r in rows if r["status"]=="computed"]
h=np.array([float(r["h"]) for r in valid]);ge=np.array([float(r["gradient_max_error"]) for r in valid]);ce=np.array([float(r["curve_error"]) for r in valid])
fig,axes=plt.subplots(1,2,figsize=(10.8,3.5))
for ax,error,title in [(axes[0],ge,"坐标中心差分"),(axes[1],ce,"路径中心差分")]:
    positive=error>0;ax.loglog(h[positive],error[positive],"o-",color=B,label="浮点绝对误差")
    ax.set(xlabel="h",ylabel="绝对误差",title=title);ax.grid(alpha=.2)
    if (~positive).any():ax.text(.53,.94,f"另有{(~positive).sum()}个精确零误差点\n未放入对数轴",transform=ax.transAxes,va="top",fontsize=9,color=G)
axes[1].loglog(h,12*h*h,"--",color=O,label="精确截断项12h²");axes[1].legend(fontsize=9)
axes[0].text(.04,.06,"h=1e-16：输入无法分辨，拒绝计算",transform=axes[0].transAxes,fontsize=9,color=R)
save(fig,"07_difference.png")

fig,axes=plt.subplots(1,2,figsize=(10.8,3.5));t=np.linspace(-.3,.3,201)
axes[0].plot(t,np.zeros_like(t),c=B,lw=3,label="沿x轴或y轴")
axes[0].plot(t,np.abs(t)/np.sqrt(2),c=R,lw=2.4,label="沿x=y=t")
axes[0].set(xlabel="t",ylabel="f",title="轴上平坦，对角线上有尖点");axes[0].legend(fontsize=10);axes[0].grid(alpha=.15)
axes[1].plot(t[t<0],np.full((t<0).sum(),-1/np.sqrt(2)),c=B,lw=2.5,label="左侧差商")
axes[1].plot(t[t>0],np.full((t>0).sum(),1/np.sqrt(2)),c=R,lw=2.5,label="右侧差商")
axes[1].axhline(0,c=GRAY,ls="--",label="中心差分=0")
axes[1].set(xlabel="t",ylabel="[f(t,t)-f(0,0)]/t",title="两侧极限不一致",ylim=(-1,1));axes[1].legend(fontsize=9)
save(fig,"08_counterexample.png")

fig,axes=plt.subplots(1,2,figsize=(10.8,3.5))
et=np.linspace(0,.06,200);v=(2-64*et)**2*(4-32*et)**2
axes[0].plot(et,v,c=B,lw=2.5,label="真实损失");axes[0].plot(et,64-5120*et,c=O,ls="--",label="一阶预测")
axes[0].axhline(0,c=GRAY,lw=.8);axes[0].set(xlabel="η",ylabel="新损失",title="局部预测很快失准",ylim=(-20,80));axes[0].legend(fontsize=10)
eta=[.001,.01,.1,.5];ls=[lab.gradient_step([[2,3]],e)["new_loss"] for e in eta]
axes[1].bar([str(e) for e in eta],ls,color=[G,G,G,R]);axes[1].set_yscale("log");axes[1].axhline(64,c=GRAY,ls="--",label="原损失64")
axes[1].set(xlabel="η（每次都从同一起点）",ylabel="新损失（对数轴）",title="大步长可以使损失暴涨");axes[1].legend(fontsize=9)
for i,value in enumerate(ls):axes[1].text(i,value*1.2,f"{value:.2f}",ha="center",fontsize=9)
axes[1].set_ylim(3,1e6)
save(fig,"09_steps.png")
print("Saved 9 original figures")
