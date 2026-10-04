"""Original process and result diagrams for the reproducibility exercise."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT=Path(__file__).resolve().parent/"figures";OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in p]
if fonts:
    font_manager.fontManager.addfont(fonts[0]);font=font_manager.FontProperties(fname=fonts[0]).get_name()
else:font="sans-serif"
plt.rcParams.update({"font.family":font,"font.size":12,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
B,G,O,R="#2563a6","#25806c","#d57b18","#bc4951"


def canvas(height=4):
    fig,ax=plt.subplots(figsize=(10,height));ax.set(xlim=(0,10),ylim=(0,height));ax.axis("off");return ax


def box(ax,x,y,w,h,text,color=B):
    ax.add_patch(Rectangle((x,y),w,h,facecolor="white",edgecolor=color,lw=1.8))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)


def arrow(ax,start,end,color="#718096"):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=15,color=color,lw=1.6))


def save(name):
    plt.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white");plt.close()


ax=canvas(3.4)
for x,w,text,c in [(.1,2.1,"训练 A B C D\n16行中抽8行\n选择偏移量",B),(2.75,2.05,"验证 E\n主种子101比较\n选择模型类型",G),(5.45,1.85,"冻结\n规则与预测\n先保存",O),(7.9,2,"测试 F\n揭示标签\n逐条核对",R)]:box(ax,x,1.2,w,1.6,text,c)
for start,end in [(2.25,2.7),(4.85,5.4),(7.35,7.85)]:arrow(ax,(start,2),(end,2))
ax.text(5,.45,"102和103保留为描述运行，不能事后替换主种子",ha="center",color=R)
save("01_protocol.png")

ax=canvas(3.9);box(ax,.25,1.5,2.65,1.7,"配置\n抽8行，种子101\n候选0、1、2",B)
box(ax,3.8,1.5,2.5,1.7,"实际执行\n独立随机发生器\n训练子集与拟合",G)
box(ax,7.1,1.5,2.65,1.7,"运行记录\n8个实际编号\n全部候选分数",O)
arrow(ax,(2.95,2.35),(3.75,2.35));arrow(ax,(6.35,2.35),(7.05,2.35))
ax.text(5,.6,"计划与记录相互核对，不能把未执行的步骤算作已完成",ha="center",color=R)
save("02_config.png")

fig,ax=plt.subplots(figsize=(9,4));x=[0,1]
ax.bar([-.18,.82],[2.25,2],.34,color=O,label="训练均值常数")
ax.bar([.18,1.18],[.25,0],.34,color=B,label="所选仿射规则")
for px,y in [(-.18,2.25),(.82,2),(.18,.25),(1.18,0)]:ax.text(px,y+.08,str(y),ha="center")
ax.set(xticks=x,xticklabels=["所选训练子集 8行","验证设备 E 4行"],ylim=(0,3.1),ylabel="MAE（读数单位）")
ax.legend(loc="upper right");ax.grid(axis="y",alpha=.15);save("03_comparison.png")

fig,ax=plt.subplots(figsize=(8.5,4.2));xs=[1,2,3,4];pred=[3,5,7,9];actual=[5,7,9,11]
ax.plot(xs,pred,"o-",color=B,label="冻结预测 2x+1");ax.plot(xs,actual,"s-",color=O,label="F标签")
for x,p,y in zip(xs,pred,actual):
    ax.annotate("",xy=(x,y),xytext=(x,p),arrowprops={"arrowstyle":"<->","color":R})
    ax.text(x+.08,p+.8,"低2",color=R,fontsize=10)
ax.set(xlabel="输入旋钮位置（格）",ylabel="读数单位",xlim=(.65,4.5),ylim=(2,12));ax.legend();ax.grid(alpha=.15);save("04_errors.png")

fig,ax=plt.subplots(figsize=(8.5,4.1));seeds=["101 主运行","102 描述","103 描述"];vals=[.25,.625,.5]
ax.bar(seeds,vals,color=[B,G,G],width=.45,label="仿射训练MAE")
ax.scatter(seeds,[0,0,0],color=O,s=75,marker="s",zorder=3,label="仿射验证MAE")
for i,y in enumerate(vals):ax.text(i,y+.03,str(y),ha="center")
ax.set(ylabel="MAE（读数单位）",ylim=(-.06,.9));ax.legend();ax.grid(axis="y",alpha=.15);save("05_seeds.png")

ax=canvas(4.8);box(ax,.2,2.8,2.65,1.35,"输入与配置\nCSV、config.json\n摘要记录版本",B)
box(ax,3.65,2.8,2.7,1.35,"冻结结果\n模型、编号、预测\n说明选择依据",G)
box(ax,7.15,2.8,2.6,1.35,"逐条评价\n标签与正负误差\n汇总仍可追查",O)
arrow(ax,(2.9,3.45),(3.6,3.45));arrow(ax,(6.4,3.45),(7.1,3.45))
box(ax,2.5,.4,5,1.2,"运行清单与日志\n文件摘要、软件版本、全部步骤",R)
arrow(ax,(1.6,2.75),(3.3,1.65));arrow(ax,(5,2.75),(5,1.65));arrow(ax,(8.45,2.75),(6.7,1.65))
save("06_artifacts.png")
