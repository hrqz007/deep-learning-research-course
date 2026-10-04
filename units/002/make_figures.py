"""Regenerate original Unit 002 figures; only the authoring tool needs matplotlib.
Set CJK_FONT to a Chinese-capable font file on other systems.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dl002-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/dl002-cache')
try:
    (Path(os.environ['XDG_CACHE_HOME']) / 'fontconfig').mkdir(parents=True, exist_ok=True)
except OSError:
    os.environ['XDG_CACHE_HOME'] = '/tmp/dl002-cache'
    (Path(os.environ['XDG_CACHE_HOME']) / 'fontconfig').mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
font = Path(os.environ.get('CJK_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if not font.exists():
    raise RuntimeError('Set CJK_FONT to a Chinese-capable font file')
F = FontProperties(fname=str(font))
BLUE, ORANGE, GREEN, RED = '#2563a6', '#d97724', '#218777', '#b64855'
plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})
def canvas(h=4.8):
    fig, ax = plt.subplots(figsize=(10, h), dpi=160)
    ax.set_xlim(0, 10); ax.set_ylim(0, h); ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    return fig, ax

def txt(ax, x, y, s, size=13, color='#17283b', ha='center', weight=None):
    ax.text(x, y, s, fontproperties=F, fontsize=size, color=color,
            ha=ha, va='center', weight=weight, linespacing=1.6)

def box(ax, x, y, w, h, s, color=BLUE, size=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.1',
                              facecolor=color+'13',edgecolor=color,lw=1.5))
    txt(ax,x+w/2,y+h/2,s,size,color)

def arrow(ax,x1,y1,x2,y2,color=BLUE):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,color=color,lw=1.6))

def save(fig, name):
    fig.savefig(OUT/name,facecolor='white',dpi=160);plt.close(fig)

fig,ax=canvas()
txt(ax,5,4.48,'同一份规则可以从两条入口执行',17)
box(ax,.15,2.85,2.1,.85,'终端\n启动 Python')
box(ax,3.25,2.85,2.9,.85,'dl002 环境\n解释器 + 标准库')
box(ax,7.05,2.85,2.65,.85,'读取 experiment.py\n逐行执行',GREEN)
arrow(ax,2.3,3.28,3.2,3.28);arrow(ax,6.2,3.28,7,3.28)
box(ax,.15,1.15,2.1,.85,'Notebook 网页\n发送代码单元格',ORANGE)
box(ax,3.25,1.15,2.9,.85,'所选 Python 内核\n保留会话状态',ORANGE)
box(ax,7.05,1.15,2.65,.85,'预测与误差\n回到界面显示',GREEN)
arrow(ax,2.3,1.58,3.2,1.58,ORANGE);arrow(ax,6.2,1.58,7,1.58,ORANGE)
txt(ax,5,.4,'检查实际解释器路径；浏览器外观不能证明内核属于哪个环境',12)
save(fig,'01_execution_path.png')

fig,ax=canvas(4.5)
txt(ax,5,4.2,'代码沿时间前进  已保存的值不会自动回算',17)
rows=[('x 取 3，bias 取 1','3','1','尚未定义'),('prediction = 2*x + bias','3','1','7'),('x = 4','4','1','7'),('prediction = 2*x + bias','4','1','9')]
for j,(code,x,b,p) in enumerate(rows):
    y=3.25-j*.73
    box(ax,.3,y,5.1,.55,code,BLUE,13)
    for k,(val,c) in enumerate([(x,ORANGE),(b,GREEN),(p,RED)]):
        box(ax,5.9+k*1.3,y,1.15,.55,val,c,13)
    if j<3:arrow(ax,.12,y,.12,y-.25)
for x,t in [(6.47,'x'),(7.77,'bias'),(9.07,'prediction')]:txt(ax,x,3.99,t,12)
txt(ax,5,.3,'第三步只更新 x；第四步重新计算 prediction',12)
save(fig,'02_assignment_time.png')

fig,ax=canvas(4.3)
txt(ax,5,3.95,'同一索引同时取出一对记录',17)
for y,label in [(2.95,'索引'),(2.05,'train_x'),(1.15,'train_y')]:txt(ax,.95,y,label,14)
for i in range(5):
    x=2.1+i*1.5
    color=ORANGE if i==2 else BLUE
    txt(ax,x+.6,2.95,str(i),14,color)
    box(ax,x,1.7,1.2,.7,str(i+1),color,16)
    box(ax,x,.8,1.2,.7,str([3,5,8,9,11][i]),color,16)
    arrow(ax,x+.6,1.65,x+.6,1.55,color)
txt(ax,5,.25,'T03 是第三条记录，因此使用索引 2：输入 3，标签 8',12)
save(fig,'03_list_alignment.png')

fig,ax=canvas(4.7)
txt(ax,5,4.36,'先累计五次  再除以记录数',17)
txt(ax,.65,3.2,'本轮\n误差',12);txt(ax,.65,1.65,'累计\ntotal',12)
for i,e in enumerate([0,0,1,0,0]):
    x=1.45+i*1.65
    box(ax,x,2.8,1.15,.8,str(e),ORANGE,18)
    box(ax,x,1.25,1.15,.8,str([0,0,1,1,1][i]),GREEN,18)
    arrow(ax,x+.58,2.7,x+.58,2.15,ORANGE)
    txt(ax,x+.58,3.87,'i = '+str(i),11)
    if i<4:arrow(ax,x+1.19,1.65,x+1.57,1.65,GREEN)
txt(ax,5,.48,'total 初值 0  →  结束时 1  →  1 / 5 = 0.2',14)
save(fig,'04_accumulator.png')

fig,ax=canvas(4.8)
txt(ax,5,4.48,'训练信息进入选择  测试标签留在选择之外',17)
for i,(b,score,errs) in enumerate([(0,'1.2','1 + 1 + 2 + 1 + 1'),(1,'0.2','0 + 0 + 1 + 0 + 0'),(2,'0.8','1 + 1 + 0 + 1 + 1')]):
    y=3.18-i*.87;c=GREEN if i==1 else BLUE
    box(ax,.35,y,1.35,.61,'b = '+str(b),c)
    box(ax,2.15,y,4.2,.61,'五条绝对误差：'+errs,c,12)
    box(ax,6.95,y,1.4,.61,'MAE '+score,c,12)
    arrow(ax,1.75,y+.3,2.08,y+.3,c);arrow(ax,6.4,y+.3,6.88,y+.3,c)
box(ax,8.8,1.83,.85,1.1,'选\n1',GREEN,17)
arrow(ax,8.4,2.63,8.75,2.39,GREEN)
txt(ax,5,.54,'外层换 b；内层算五行。候选升序 + 严格小于 → 同分留较小 b',12)
save(fig,'05_candidate_loops.png')

fig,ax=canvas(5)
txt(ax,5,4.7,'编辑文字不会更新内核中的变量',17)
labels=['顺序执行两格','仅编辑第一格','执行第一格和第二格','重启后只执行第二格']
visible=['bias = 1','bias = 2','bias = 2','bias = 2']
state=['内核 bias 为 1','内核仍为 1','内核 bias 为 2','内核没有 bias']
output=['输出 7','仍输出 7','输出 8','NameError']
for i in range(4):
    y=3.7-i*.86
    txt(ax,.2,y+.24,str(i+1),12,ha='left')
    box(ax,.65,y,3.05,.53,labels[i],BLUE,12)
    box(ax,4.05,y,2.65,.53,state[i],ORANGE,12)
    box(ax,7.1,y,2.55,.53,output[i],RED if i==3 else GREEN,13)
    arrow(ax,3.75,y+.26,3.99,y+.26);arrow(ax,6.75,y+.26,7.04,y+.26)
txt(ax,5,.35,'清除输出 ≠ 清除变量；正式验收：重启内核，再从头运行全部',13)
save(fig,'06_notebook_state.png')
print('6 original figures saved')
