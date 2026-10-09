"""Rebuild six original color figures from local saved evidence."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-matplotlib')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
CJK=os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(CJK)
FONT=font_manager.FontProperties(fname=CJK).get_name()
plt.rcParams.update({'font.family':FONT,'axes.unicode_minus':False,'font.size':10})
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
BLUE='#1768ac'; ORANGE='#dc772a'; GREEN='#23856d'; RED='#b54555'
def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/name,dpi=160,bbox_inches='tight')
    plt.close(fig)
def box(ax,x,y,text,color=BLUE,w=.24,h=.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.018',fc=color,ec='none',alpha=.13))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=color,fontsize=11)
def arrow(ax,a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#526075','lw':1.7})

from experiment import ig,init
r=json.loads((ROOT/'outputs/results.json').read_text());d=np.load(ROOT/'outputs/data.npz');w=np.load(ROOT/'outputs/weights.npz');p=[w[f'p{k}'] for k in range(4)]
fig,ax=plt.subplots(figsize=(9,3.2));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));box(ax,.03,.4,'输入字段 X、Z');box(ax,.37,.4,'网络计算\nf(X,Z)',GREEN);box(ax,.73,.4,'梯度/归因\n描述模型依赖',ORANGE);arrow(ax,(.29,.5),(.35,.5));arrow(ax,(.63,.5),(.71,.5));ax.text(.5,.13,'从这里走向现实干预，需要额外生成机制与识别假设',ha='center',color=RED);save(fig,'01_model_question.png')
fig,ax=plt.subplots(figsize=(9,3.5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));
box(ax,.38,.72,'共同原因 U',ORANGE,w=.2,h=.15);box(ax,.06,.32,'X = U + εx',BLUE,w=.24,h=.17);box(ax,.4,.32,'Y = 2X + 3U + εy',GREEN,w=.27,h=.17);box(ax,.75,.32,'Z = Y + εz',RED,w=.21,h=.17)
arrow(ax,(.39,.73),(.22,.52));arrow(ax,(.5,.71),(.52,.52));arrow(ax,(.32,.405),(.38,.405));arrow(ax,(.69,.405),(.73,.405));ax.text(.5,.08,'Z 是结果的下游测量：预测有用，改变 Z 不会反向改变 Y',ha='center',color=RED);save(fig,'02_causal_graph.png')
fig,ax=plt.subplots(figsize=(8.8,3.5));a=np.linspace(0,1,80);point=np.array(r['point']);
for b,c,label in [(np.zeros(2),BLUE,'零基线'),(np.array([.5,2.5]),ORANGE,'替代基线')]:
 path=b+a[:,None]*(point-b);ax.plot(path[:,0],path[:,1],color=c,label=label);ax.scatter(*b,color=c,s=60)
ax.scatter(*point,c=GREEN,s=80,label='同一目标点');ax.set(xlabel='输入 X',ylabel='输入 Z',title='积分梯度沿明确路径累计；路径不等于现实干预');ax.legend();save(fig,'03_ig_path.png')
fig,axes=plt.subplots(1,2,figsize=(9,3.6));loc=np.arange(2);axes[0].bar(loc-.18,r['ig_zero'],.36,color=BLUE,label='零基线');axes[0].bar(loc+.18,r['ig_alternative'],.36,color=ORANGE,label='替代基线');axes[0].set_xticks(loc,['X归因','Z归因']);axes[0].set_ylabel('带符号归因');axes[0].legend(fontsize=8)
point=np.array(r['point']);rand=init(77);head=[a.copy() for a in p];head[2:]=rand[2:]
for j,(model,label,c) in enumerate([(p,'已训练',BLUE),(head,'随机输出层',ORANGE),(rand,'全随机',RED)]): axes[1].bar(loc+(j-1)*.24,ig(point,np.zeros(2),model),.24,label=label,color=c)
axes[1].set_xticks(loc,['X归因','Z归因']);axes[1].set_ylabel('同一点的归因');axes[1].legend(fontsize=8);save(fig,'04_attribution_checks.png')
fig,axes=plt.subplots(1,2,figsize=(9,3.4));q=r['perturbation'];axes[0].loglog(q['delta'],q['actual_change'],'o-',color=BLUE,label='实际输出变化');axes[0].loglog(q['delta'],q['linearized_change'],'x--',color=ORANGE,label='局部梯度预测');axes[0].set(xlabel='对输入 Z 的增量',ylabel='模型输出增量');axes[0].legend(fontsize=8)
err=np.abs(np.array(q['actual_change'])-q['linearized_change']);axes[1].loglog(q['delta'],err,'o-',color=RED);axes[1].set(xlabel='对输入 Z 的增量',ylabel='线性近似绝对误差');save(fig,'05_input_perturbation.png')
fig,axes=plt.subplots(1,2,figsize=(9,3.7));c=r['causal'];axes[0].bar(['观察回归','调整 U','真正 do(X)'],[c['observational_x_slope'],c['u_adjusted_x_slope'],c['do_x_effect_per_unit']],color=[ORANGE,BLUE,GREEN]);axes[0].set(ylabel='X 增加一单位的斜率/效应');axes[1].bar(['模型输入 Z+1','世界 do(Z)'],[c['model_response_to_z_plus_one'],c['do_z_effect_per_unit']],color=[BLUE,GREEN]);axes[1].set(ylabel='预测变化 / 真实结果变化',ylim=(-.1,1.2));axes[1].axhline(0,color='#555',lw=.8);save(fig,'06_effect_comparison.png')
