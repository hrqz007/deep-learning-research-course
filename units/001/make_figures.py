"""Rebuild original teaching diagrams (optional: matplotlib required)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
# Use an installed CJK font when available; do not bundle third-party fonts.
font_candidates = [path for path in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in path]
if font_candidates:
    font_manager.fontManager.addfont(font_candidates[0])
    font_name = font_manager.FontProperties(fname=font_candidates[0]).get_name()
else:
    font_name = "sans-serif"
plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
BLUE, ORANGE, GREEN, RED = "#2563a6", "#d57b18", "#25806c", "#bc4951"


def save(name):
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def node(ax, x, y, w, h, text, color=BLUE):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.8))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", color="#18202a", fontsize=12)


def arrow(ax, start, end, color="#657589"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.7, color=color))


fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set(xlim=(0, 10), ylim=(0, 3)); ax.axis("off")
for x, text, c in [(0.15,"输入已确定\nx = 2.5 格",BLUE),(3.55,"作出预测\n预测读数 = 6",GREEN),(6.95,"屏幕显示答案\ny = 6 读数单位",ORANGE)]:
    node(ax, x, 1.25, 2.85, 1.1, text, c)
arrow(ax, (3,1.8), (3.5,1.8)); arrow(ax, (6.4,1.8), (6.9,1.8))
ax.plot([6.7,6.7],[0.6,2.7],"--",color=ORANGE)
ax.text(5,0.35,"预测之前不能读取这一次的答案",ha="center",color=RED)
save("01_timeline.png")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter([1,2,3,4,5],[3,5,8,9,11],s=75,color=BLUE,label="训练输入与已知标签",zorder=3)
for x,y,i in zip([1,2,3,4,5],[3,5,8,9,11],range(1,6)):ax.annotate(f"T0{i}",(x,y),xytext=(7,4),textcoords="offset points",fontsize=10)
for x in [1.5,2.5,3.5,4.5]:ax.text(x,0.35,"?",color=ORANGE,fontsize=23,ha="center")
ax.plot([1,5],[3,11],"--",color="#b4c7d8",label="视觉参照：2x + 1（尚未选偏移）")
ax.set(xlabel="旋钮位置 x（格）",ylabel="屏幕读数 y（读数单位）",ylim=(-.2,13),xlim=(.6,5.5))
ax.text(3.1,1.6,"测试输入只有位置，问号不表示标签为0",color=ORANGE,ha="center",fontsize=10)
ax.legend(loc="upper left",fontsize=10);ax.grid(alpha=.15)
save("02_split.png")

fig,ax=plt.subplots(figsize=(10,4.5));ax.set(xlim=(0,10),ylim=(0,4.5));ax.axis("off")
node(ax,.15,2.65,2.1,1.1,"规则 A 输入\nx = 2.5",BLUE)
node(ax,3.1,2.65,3.1,1.1,"固定系数 2\n所选偏移量 b = 1",BLUE)
node(ax,7.15,2.65,2.55,1.1,"2 × 2.5 + 1\n输出 6",GREEN)
node(ax,.15,.55,2.1,1.1,"规则 B 输入\nx = 2.5",ORANGE)
node(ax,3.1,.55,3.1,1.1,"查询训练表\n没有完全相同输入",ORANGE)
node(ax,7.15,.55,2.55,1.1,"采用后备规则\n输出 0",RED)
for y in [3.2,1.1]:arrow(ax,(2.3,y),(3.05,y));arrow(ax,(6.25,y),(7.1,y))
ax.text(5,2.15,"在 x = 3 时：A 输出 7；B 查表输出 8",ha="center",fontsize=11)
save("03_rules.png")

fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,vals,title in zip(axs,[[-1,1],[1,1]],["有符号偏差：平均为 0","绝对误差：平均为 1"]):
    ax.bar(["预测 7\n真实 8","预测 9\n真实 8"],vals,color=[BLUE,ORANGE],width=.45)
    ax.axhline(0,color="#667085",linewidth=.8);ax.set(ylim=(-1.5,1.6),title=title,ylabel="读数单位")
    for x,v in enumerate(vals):ax.text(x,v+(.1 if v>0 else -.25),str(v),ha="center")
fig.tight_layout(w_pad=3);save("04_loss.png")

fig,ax=plt.subplots(figsize=(8.7,4))
for offset,values,label,c in [(-.18,[.2,0],"规则 A",BLUE),(.18,[0,7],"规则 B",ORANGE)]:
    ax.bar([offset,1+offset],values,width=.34,label=label,color=c)
    for x,y in zip([offset,1+offset],values):ax.text(x,y+.15,str(y),ha="center")
ax.set(xticks=[0,1],xticklabels=["训练资料：5条","测试资料：4条"],ylabel="平均绝对误差（读数单位）",ylim=(0,8.7))
ax.legend();ax.grid(axis="y",alpha=.15);save("05_scores.png")

fig,ax=plt.subplots(figsize=(10,3.8));ax.set(xlim=(0,10),ylim=(0,3.8));ax.axis("off")
for x,w,text,c in [(.05,1.5,"输入\n原始数字",BLUE),(2.2,2,"可学习变换1\n中间表示",GREEN),(4.9,2,"可学习变换2\n中间表示",GREEN),(7.6,2.2,"输出预测\n任务所需的数",ORANGE)]:node(ax,x,1.8,w,1.2,text,c)
for start,end in [(1.55,2.15),(4.2,4.85),(6.9,7.55)]:arrow(ax,(start,2.4),(end,2.4))
ax.text(5,.95,"训练：利用标签与损失调整参数",ha="center",color=GREEN)
ax.text(5,.35,"评价：方案确定后检查未参与开发的资料",ha="center",color=RED)
save("06_depth.png")
