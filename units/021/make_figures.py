"""Build eight original figures from the checked finite examples; matplotlib only."""
from experiment import require_default_figure_inputs
require_default_figure_inputs()
from pathlib import Path
import json
import math
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dl021-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
HERE=Path(__file__).resolve().parent
fonts=font_manager.findSystemFonts()
for p in fonts:
    if 'NotoSansCJK' in p:
        font_manager.fontManager.addfont(p)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
COL={'train':'#287c9e','val':'#d38b13','test':'#7755a6','bad':'#ce5254','good':'#3b9170','ink':'#203444'}
(HERE/'figures').mkdir(exist_ok=True)
s=json.loads((HERE/'outputs/summary.json').read_text())

def save(fig,name):
    fig.savefig(HERE/'figures'/name,dpi=180,bbox_inches='tight',metadata={'Software':'Unit 021 original figure'})
    plt.close(fig)

def box(ax,x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',fc=color,ec='none',alpha=.13))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=COL['ink'],fontsize=11)

fig,ax=plt.subplots(figsize=(11.6,4));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
box(ax,.2,1.65,2.3,1.4,'训练集\n只拟合规则与预处理\n本例仅确定多数类基线',COL['train'])
box(ax,3.2,1.65,2.3,1.4,'验证集\n按事先规定的代价\n选择阈值 0.4',COL['val'])
box(ax,6.2,1.65,2.3,1.4,'冻结方案\n规则、阈值、指标\n分组、分箱、并列规则',COL['ink'])
box(ax,9.2,1.65,2.3,1.4,'封存测试集\n冻结后首次读取\n记录所有规定结果',COL['test'])
for x in (2.55,5.55,8.55):ax.annotate('',xy=(x+.55,2.35),xytext=(x,2.35),arrowprops={'arrowstyle':'->','lw':2,'color':COL['ink']})
ax.text(6,.7,'测试结果若促使改方案，该测试集就已参与开发；下一轮需要新的独立评价。',ha='center',color=COL['bad'])
ax.set_title('图 1  信息流决定一个数字是否有未见数据的含义',loc='left',fontweight='bold',pad=12)
save(fig,'01_isolation.png')

fig,axes=plt.subplots(1,2,figsize=(10.8,4),gridspec_kw={'width_ratios':[1.1,1]})
a=axes[0];a.axis('off');a.set(xlim=(0,4),ylim=(0,3.3))
for i,(label,num,color) in enumerate([('TN',3,COL['good']),('FP',2,COL['bad']),('FN',1,COL['bad']),('TP',2,COL['good'])]):
    x=1.1+(i%2)*1.3;y=1.4-(i//2)*1.2
    a.add_patch(Rectangle((x,y),1.2,1.1,fc=color,alpha=.18))
    a.text(x+.6,y+.55,f'{label} = {num}',ha='center',va='center',fontsize=15)
a.text(1.7,2.7,'预测 0',ha='center');a.text(3,2.7,'预测 1',ha='center')
a.text(.9,1.95,'实际 0',ha='right');a.text(.9,.75,'实际 1',ha='right')
a.set_title('8 条验证记录，阈值 0.5',loc='left')
a=axes[1];a.axis('off')
a.text(.03,.85,'Precision 以“发出告警”为分母',color=COL['val'],fontsize=14)
a.text(.03,.68,'2 / (2 + 2) = 1/2  →  4 次告警中的 2 次正确',fontsize=11)
a.text(.03,.47,'Recall 以“真正需要告警”为分母',color=COL['train'],fontsize=14)
a.text(.03,.30,'2 / (2 + 1) = 2/3  →  3 次事件中的 2 次找到',fontsize=11)
a.text(.03,.08,'Accuracy = 5/8；F1 = 4/7',fontsize=13)
fig.suptitle('图 2  同一张混淆矩阵，不同问题选择不同分母',x=.05,ha='left',fontweight='bold')
fig.tight_layout(rect=(0,0,1,.91));save(fig,'02_confusion.png')

import csv
with (HERE/'outputs/thresholds.csv').open(newline='', encoding='utf-8') as stream:
    rows=list(csv.DictReader(stream))
fig,axes=plt.subplots(1,2,figsize=(11,4))
x=[float(r['threshold']) for r in rows]
axes[0].plot(x,[float(r['mean_cost']) for r in rows],'o-',color=COL['val'])
axes[0].scatter([.4],[.25],s=130,color=COL['good'],zorder=4)
axes[0].set(xlabel='阈值（等于阈值也判正）',ylabel='验证平均代价',ylim=(0,.85),xticks=x)
axes[0].annotate('验证集选择 0.4',(.4,.25),(.48,.12),arrowprops={'arrowstyle':'->'})
axes[1].plot(x,[int(r['fp']) for r in rows],'o-',label='FP 误报',color=COL['bad'])
axes[1].plot(x,[int(r['fn']) for r in rows],'s-',label='FN 漏报',color=COL['train'])
axes[1].set(xlabel='阈值',ylabel='记录数',ylim=(-.2,3.5),yticks=range(4),xticks=x);axes[1].legend()
fig.suptitle('图 3  代价 = (3 × FN + FP) / 8；最优仅指这组候选与这批验证数据',x=.04,ha='left',fontweight='bold')
fig.tight_layout(rect=(0,0,1,.91));save(fig,'03_thresholds.png')

fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(['类别 0\n10 条','类别 1\n2 条','类别 2\n1 条'],[8/9,.5,.5],color=[COL['train'],COL['val'],COL['test']])
axes[0].set(ylabel='逐类 F1',ylim=(0,1.05))
for i,v in enumerate([8/9,.5,.5]):axes[0].text(i,v+.025,f'{v:.3f}',ha='center')
axes[1].bar(['宏平均\n每类等权','微平均\n合并计数','加权平均\n按真实支持数'],[17/27,10/13,187/234],color=[COL['train'],COL['val'],COL['test']])
axes[1].set(ylabel='F1 汇总',ylim=(0,1.05))
for i,v in enumerate([17/27,10/13,187/234]):axes[1].text(i,v+.025,f'{v:.3f}',ha='center')
fig.suptitle('图 4  平均方法改变被回答的问题；所有真实与预测类别均纳入',x=.04,ha='left',fontweight='bold')
fig.tight_layout(rect=(0,0,1,.91));save(fig,'04_averaging.png')

fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(range(1,5),[1,0,-2,1],color=[COL['val'],COL['train'],COL['bad'],COL['val']]);axes[0].axhline(0,c=COL['ink'],lw=.8)
axes[0].set(xlabel='记录',ylabel='预测值 − 真实值（分钟）',xticks=range(1,5),ylim=(-2.5,2))
axes[0].text(1,1.55,'平均带符号误差 = 0，但 MAE = 1 分钟',fontsize=10)
axes[1].bar([0,1],[1,.75],width=.32,label='MAE',color=COL['train'])
axes[1].bar([.35,1.35],[1,2.25],width=.32,label='MSE',color=COL['bad'])
axes[1].set(xticks=[.175,1.175],xticklabels=['A: 四个误差均为 −1','B: 0, 0, 0, 3'],ylabel='数值（单位分别为分钟与分钟²）',ylim=(0,2.7));axes[1].legend()
fig.suptitle('图 5  绝对值防止抵消；平方让少数大误差更显著',x=.04,ha='left',fontweight='bold')
fig.tight_layout(rect=(0,0,1,.91));save(fig,'05_regression.png')

fig,axes=plt.subplots(1,2,figsize=(11,4))
bins=s['calibration']['bins'];a=axes[0]
a.plot([0,1],[0,1],'--',c='#94a3b8',label='总体理想参考线')
a.scatter([b['mean_probability'] for b in bins],[b['positive_frequency'] for b in bins],color=COL['test'],s=80,zorder=3)
for b in bins:a.annotate('n = 2',(b['mean_probability'],b['positive_frequency']),xytext=(6,8 if b['positive_frequency']<1 else -18),textcoords='offset points',fontsize=10)
a.set(xlabel='箱内平均预测概率',ylabel='箱内观察到的正例比例',xlim=(-.04,1.04),ylim=(-.08,1.08));a.legend(fontsize=9,loc='lower right')
a=axes[1]
for n,color in [(4,COL['val']),(40,COL['train'])]:
    k=list(range(n+1));a.plot([v/n for v in k],[math.comb(n,v)*.5**n for v in k],'o-',ms=3,label=f'独立样本 n = {n}',color=color)
a.set(xlabel='观察正例频率',ylabel='该频率的概率',ylim=(0,.45));a.legend(fontsize=9)
fig.suptitle('图 6  左侧仅为小样本描述；右侧为真实概率 0.5 时的精确重复抽样分布',x=.04,ha='left',fontweight='bold')
fig.tight_layout(rect=(0,0,1,.91));save(fig,'06_calibration.png')

fig,axes=plt.subplots(1,2,figsize=(11,4))
colors=[COL['train'],COL['val'],COL['test']]
for a,kind in zip(axes,['entity','time']):
    for i in range(3):
        for j in range(3):
            c=colors[i if kind=='entity' else j]
            a.add_patch(Rectangle((j,i),.94,.94,fc=c,alpha=.8));a.text(j+.47,i+.47,f'{chr(65+i)}{j+1}',ha='center',va='center',color='white',fontsize=14)
    a.set(xlim=(-.05,3),ylim=(3,-.05),xticks=[.47,1.47,2.47],xticklabels=['时点 1','时点 2','时点 3'],yticks=[.47,1.47,2.47],yticklabels=['实体 A','实体 B','实体 C'])
    a.set_aspect('equal')
    a.set_title('按实体：评价新实体' if kind=='entity' else '按时间：评价未来记录')
fig.suptitle('图 7  蓝 = 训练，橙 = 验证，紫 = 测试；实体隔离与时间隔离是不同约束',x=.04,ha='left',fontweight='bold')
fig.tight_layout(rect=(0,0,1,.91));save(fig,'07_splits.png')

fig,axes=plt.subplots(1,2,figsize=(11,4))
a=axes[0];names=['A 组\nTP=2, FN=0','B 组\nTP=0, FN=1'];a.bar(names,[1,0],color=[COL['train'],COL['bad']]);a.set(ylabel='测试 recall',ylim=(-.08,1.28))
a.text(0,1.07,'2/2 正例；总数 4',ha='center',fontsize=10);a.text(1,.12,'0/1 正例；总数 4',ha='center',fontsize=10)
a=axes[1];a.bar(['逐行平均\n8/10','逐实体等权\n(1+0)/2'],[.8,.5],color=[COL['val'],COL['test']]);a.set(ylabel='另一独立例的 accuracy',ylim=(0,1.15))
a.text(.5,1.03,'实体甲 8 行全对；实体乙 2 行全错',ha='center',fontsize=10)
fig.suptitle('图 8  报告分组分母和加权单位；组间差异在这些小样本上仅是描述',x=.04,ha='left',fontweight='bold')
fig.tight_layout(rect=(0,0,1,.91));save(fig,'08_groups.png')
print('Built 8 original figures.')
