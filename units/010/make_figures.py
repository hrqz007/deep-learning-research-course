"""Original scalar-calculus diagrams. Core calculations remain stdlib-only."""
from pathlib import Path
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from experiment import run

BASE=Path(__file__).resolve().parent;OUT=BASE/"figures";OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in p]
if fonts:
    font_manager.fontManager.addfont(fonts[0]);font=font_manager.FontProperties(fname=fonts[0]).get_name()
else:font="sans-serif"
plt.rcParams.update({"font.family":font,"font.size":12,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
B,G,O,R="#2563a6","#25806c","#d57b18","#bc4951"


def save(name):
    plt.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white");plt.close()


def grid(a,b,n=201):return [a+(b-a)*i/(n-1) for i in range(n)]


xs=grid(1.2,3)
fig,ax=plt.subplots(figsize=(8.5,4.5));ax.plot(xs,[x*x for x in xs],color="#263445",lw=2,label="f(x)=x²")
for h,c in [(.8,O),(.3,B),(.05,G)]:
    slope=4+h;ax.plot(xs,[4+slope*(x-2) for x in xs],color=c,label=f"割线 h={h}")
    ax.scatter([2+h],[(2+h)**2],color=c,s=35)
ax.plot(xs,[4+4*(x-2) for x in xs],"--",color=R,label="切线斜率4")
ax.scatter([2],[4],color="#18202a",zorder=5);ax.set(xlabel="输入 x",ylabel="函数值",ylim=(.5,9.5));ax.grid(alpha=.15);ax.legend(fontsize=10)
save("01_secants.png")

fig,axs=plt.subplots(1,2,figsize=(10,4.2));xs=grid(1.5,2.6)
axs[0].plot(xs,[x*x for x in xs],color=B,label="精确平方");axs[0].plot(xs,[4+4*(x-2) for x in xs],"--",color=O,label="在2处的切线")
axs[0].set(xlabel="x",ylabel="输出",title="当前位置附近的近似");axs[0].legend(fontsize=10);axs[0].grid(alpha=.15)
ds=grid(0,.55);axs[1].plot(ds,[d*d for d in ds],color=R)
axs[1].scatter([.1,.5],[.01,.25],color=R);axs[1].annotate("0.1 → 0.01",(.1,.01),(.15,.06),fontsize=10);axs[1].annotate("0.5 → 0.25",(.5,.25),(.23,.26),fontsize=10)
axs[1].set(xlabel="增量 δ",ylabel="近似误差 δ²",title="切线漏掉的部分");axs[1].grid(alpha=.15)
fig.tight_layout(w_pad=2);save("02_local.png")

fig,ax=plt.subplots(figsize=(8,4.3));xs=grid(-1,1)
ax.plot(xs,[abs(x) for x in xs],color=B,lw=2);ax.plot([-.4,.4],[.4,.4],"o--",color=O,label="中心割线斜率0")
ax.scatter([0],[0],color=R,zorder=3);ax.text(-.8,.65,"左斜率 -1",color=G);ax.text(.4,.75,"右斜率 1",color=R)
ax.set(xlabel="x",ylabel="|x|",ylim=(-.1,1.2));ax.legend(loc="upper center",fontsize=11);ax.grid(alpha=.15)
save("03_cusp.png")

run()
rows=list(csv.DictReader((BASE/"outputs/difference_scan.csv").open()))
valid=[row for row in rows if row["status"]=="computed"]
fig,ax=plt.subplots(figsize=(9,4.4))
for key,label,c in [("forward_error","前向差分",O),("central_error","中心差分",B)]:
    shown=[r for r in valid if float(r[key])>0]
    ax.loglog([float(r["h"]) for r in shown],[float(r[key]) for r in shown],"o-",color=c,label=label,markersize=4)
ax.invert_xaxis();ax.set(xlabel="步长 h（向右更小）",ylabel="对解析导数数值的绝对误差");ax.legend();ax.grid(alpha=.2,which="both")
ax.text(.5,-.23,"h=1e-17 输入不可分辨，单独标为失败，不填成零误差",transform=ax.transAxes,ha="center",color=R,fontsize=10)
save("04_steps.png")

fig,axs=plt.subplots(1,3,figsize=(10.5,4));width=.5
for ax,offset,title,c in zip(axs,[0,1,.5],["左端点 总量3","右端点 总量5","中点 总量4"],[O,B,G]):
    starts=[k*width for k in range(4)];heights=[2*(k+offset)*width for k in range(4)]
    ax.bar(starts,heights,width=width,align="edge",alpha=.25,edgecolor=c,color=c)
    ax.plot([0,2],[0,4],color=c,lw=2);ax.scatter([(k+offset)*width for k in range(4)],heights,color=c,s=20)
    ax.set(title=title,xlabel="时间（分钟）",xlim=(0,2.1),ylim=(0,4.5));ax.grid(alpha=.12)
axs[0].set_ylabel("流量（升/分钟）");fig.tight_layout(w_pad=2);save("05_rectangles.png")

fig,ax=plt.subplots(figsize=(7.8,4.3));xs=grid(-1,1)
ax.plot(xs,xs,color=B);ax.fill_between(xs,xs,0,where=[x<=0 for x in xs],color=R,alpha=.25);ax.fill_between(xs,xs,0,where=[x>=0 for x in xs],color=G,alpha=.25)
ax.axhline(0,color="#64748b",lw=.8);ax.axvline(0,color="#64748b",lw=.8)
ax.text(-.72,-.32,"贡献 -1/2",color=R);ax.text(.48,.28,"贡献 +1/2",color=G)
ax.text(-.8,1.02,"净积分0；绝对面积1",color="#18202a");ax.set(xlabel="t",ylabel="f(t)=t",ylim=(-1.1,1.2));ax.grid(alpha=.15)
save("06_signed_area.png")

fig,ax=plt.subplots(figsize=(10,4.2));ax.set(xlim=(0,10),ylim=(0,4.2));ax.axis("off")
for x,text,c in [(0.3,"速率 q(t)=2t\n单位：升/分钟",B),(6.8,"累计 F(x)=x²\n单位：升",G)]:
    ax.add_patch(Rectangle((x,2.2),2.8,1.3,facecolor="white",edgecolor=c,lw=2));ax.text(x+1.4,2.85,text,ha="center",va="center")
ax.add_patch(FancyArrowPatch((3.2,3.1),(6.7,3.1),arrowstyle="-|>",mutation_scale=17,color=O,lw=1.8));ax.text(5,3.55,"从0累计到x",ha="center",color=O)
ax.add_patch(FancyArrowPatch((6.7,2.55),(3.2,2.55),arrowstyle="-|>",mutation_scale=17,color=R,lw=1.8));ax.text(5,1.85,"连续条件下求变化率",ha="center",color=R)
ax.text(5,.9,"G(x)=x²+7 也有 G′(x)=2x",ha="center");ax.text(5,.35,"G(2)-G(0)=11-7=4",ha="center",color=G)
save("07_fundamental.png")

fig,ax=plt.subplots(figsize=(9,4.8));ws=grid(-.4,7)
ax.plot(ws,[(w-3)**2 for w in ws],color="#657589",lw=2);ax.axhline(9,color="#999999",ls="--",lw=1,label="起点损失9")
ax.scatter([0],[9],color="#18202a",s=70,zorder=3);ax.annotate("起点w=0",(0,9),(.2,14),arrowprops={"arrowstyle":"->","color":"#18202a"})
for step,c in [(.1,B),(.5,G),(1,O),(1.1,R)]:
    w=6*step;loss=(w-3)**2;ax.scatter([w],[loss],color=c,s=65,label=f"η={step}，损失{loss:.2f}",zorder=4)
ax.set(xlabel="参数 w",ylabel="损失 (w-3)²",ylim=(-.8,19),xlim=(-.5,7.1));ax.legend(fontsize=10,loc="upper center");ax.grid(alpha=.15)
save("08_descent.png")
