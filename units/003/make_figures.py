"""Original diagrams for functions, records and small object interfaces."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
fonts = [p for p in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in p]
if fonts:
    font_manager.fontManager.addfont(fonts[0])
    font = font_manager.FontProperties(fname=fonts[0]).get_name()
else:
    font = "sans-serif"
plt.rcParams.update({"font.family": font, "font.size": 12, "axes.unicode_minus": False})
B, G, O, R = "#2563a6", "#25806c", "#d57b18", "#bc4951"


def canvas(height=4):
    fig, ax = plt.subplots(figsize=(10,height));ax.set(xlim=(0,10),ylim=(0,height));ax.axis("off")
    return ax


def box(ax,x,y,w,h,text,color=B):
    ax.add_patch(Rectangle((x,y),w,h,facecolor="white",edgecolor=color,lw=1.8))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)


def arrow(ax,start,end,color="#67798e"):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=15,color=color,lw=1.6))


def save(name):
    plt.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white");plt.close()


ax=canvas(3)
for i,(s,c) in enumerate([("读取文本\nCSV → 字段",B),("核对记录\n名字与数值",G),("产生预测\n输入 → 输出",B),("对齐评分\n标签与误差",O),("保存报告\n逐行与汇总",G)]):
    box(ax,.08+2*i,1.2,1.75,1.1,s,c)
    if i<4:arrow(ax,(1.83+2*i,1.75),(2.05+2*i,1.75))
ax.text(5,.5,"每个箭头旁都应能说清类型、数量和意义",ha="center",color=R)
save("01_pipeline.png")

ax=canvas(3.7)
box(ax,.2,2.3,2,1,"rows 列表\n位置 0、1",B)
box(ax,3.3,2.3,3.1,1,"rows[1]\n第二条记录字典",G)
box(ax,7.2,2.3,2.5,1,'["x"]\n取出数值 4',O)
arrow(ax,(2.25,2.8),(3.25,2.8));arrow(ax,(6.45,2.8),(7.15,2.8))
ax.text(5,1.45,'位置 0 → {"id": "A", "x": 2}     位置 1 → {"id": "B", "x": 4}',ha="center")
ax.text(5,.6,"超出列表位置：IndexError       字典中无此键：KeyError",ha="center",color=R)
save("02_records.png")

ax=canvas(4.4)
box(ax,.15,2.8,2.3,1.1,"从单元目录启动\n当前目录 A",B)
box(ax,.15,.65,2.3,1.1,"从其他目录启动\n当前目录 B",O)
box(ax,3.8,1.65,2.5,1.1,"同一份脚本\n以自身位置定位",G)
box(ax,7.4,1.65,2.4,1.1,"随附的数据\ndata/train.csv",B)
arrow(ax,(2.5,3.3),(3.75,2.4));arrow(ax,(2.5,1.2),(3.75,2.05));arrow(ax,(6.35,2.2),(7.35,2.2))
ax.text(5,.15,"Notebook另约定从本单元目录打开，不能假设有 __file__",ha="center",fontsize=10,color=R)
save("03_paths.png")

ax=canvas(4)
box(ax,.15,2.4,2.4,1.05,'原始字段\nx = "bad"',O)
box(ax,3.55,2.4,2.6,1.05,"尝试转换成数值\nValueError",R)
box(ax,7.25,2.4,2.5,1.05,"补充行号字段\n保留原因再抛出",R)
arrow(ax,(2.6,2.93),(3.5,2.93));arrow(ax,(6.2,2.93),(7.2,2.93))
ax.text(5,1.45,"正常实验：拒绝无效输入并停止",ha="center",color=R)
ax.text(5,.65,"故障测试：明确预期的类型，捕捉后记录已识别",ha="center",color=G)
save("04_errors.png")

ax=canvas(3.8)
for x,text,c in [(.2,"正常例子\n手算 MAE = 1\n检查公式",B),(3.5,"边界例子\n空、错长、非数值\n检查输入约定",O),(6.8,"故障例子\n错索引、错路径\n检查定位能力",R)]:box(ax,x,1.35,3,1.7,text,c)
ax.text(5,.55,"不同长度的手算例子，可以抓住固定除数的静默错误",ha="center",color=G)
save("05_tests.png")

ax=canvas(4)
box(ax,3.8,2.6,2.4,1,"AffineRule 类\n创建规则实例",B)
box(ax,.8,.6,3.1,1.2,"first 实例\nbias = 1，输入3 → 7",G)
box(ax,6.1,.6,3.1,1.2,"second 实例\nbias = 2，输入3 → 8",O)
arrow(ax,(4.1,2.55),(2.65,1.85));arrow(ax,(5.9,2.55),(7.3,1.85))
ax.text(5,.1,"self 指向本次调用所属的实例",ha="center",color=R)
save("06_objects.png")
