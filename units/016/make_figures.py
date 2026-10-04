"""Original figures for a finite probability tutorial."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl016-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle,FancyArrowPatch
BASE=Path(__file__).resolve().parent;OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False})
blue='#215e9c';orange='#b64f16';green='#287b58';gray='#596a78'
def save(fig,name):fig.savefig(OUT/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
def box(ax,x,y,w,h,text,color=blue):ax.add_patch(Rectangle((x,y),w,h,edgecolor=color,facecolor='#f0f5f8',lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=color,fontsize=11)
def arrow(ax,a,b,color=gray):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,lw=1.3))
fig,ax=plt.subplots(figsize=(8,3.1));ax.axis('off')
colours=['#aab7c2']*882+[blue]*98+['#f1b569']*2+[orange]*18
for start,count,color,label in [(0,882,'#aab7c2','合格未标记 882'),(882,98,blue,'合格标记 98'),(980,2,'#f1b569','缺陷未标记 2'),(982,18,orange,'缺陷标记 18')]:
 ids=np.arange(start,start+count);ax.scatter(ids%50,19-ids//50,s=10,color=color,label=label)
ax.set(xlim=(-1,50),ylim=(-1,21));ax.legend(loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,fontsize=9,frameon=False);ax.set_title('1000个等概率编号；每个点表示一件零件')
save(fig,'01_table.png')
fig,axs=plt.subplots(1,2,figsize=(8.4,3.2))
for ax,vals,labels,title in zip(axs,[[18,2],[18,98]],[['被标记18','未标记2'],['缺陷18','合格98']],['已知缺陷：分母20','已知标记：分母116']):
 ax.barh([0,1],vals,color=[orange,blue]);ax.set_yticks([0,1],labels);ax.set_xlabel('零件件数');ax.set_title(title);ax.invert_yaxis()
fig.tight_layout();save(fig,'02_condition.png')
fig,ax=plt.subplots(figsize=(10,4.2));ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off');box(ax,.1,2.05,1.7,.9,'总体\n概率1');box(ax,3.6,3.25,1.9,.8,'缺陷 D');box(ax,3.6,.95,1.9,.8,'合格 非D')
for start,end,label,y in [((1.8,2.65),(3.6,3.65),'1/50',3.25),((1.8,2.45),(3.6,1.35),'49/50',1.55)]:arrow(ax,start,end);ax.text(2.6,y,label,ha='center',color=blue)
leaves=[(4.1,'标记 F','9/10','9/500',3.65),(3.05,'未标记 非F','1/10','1/500',3.65),(1.6,'标记 F','1/10','49/500',1.35),(.55,'未标记 非F','9/10','441/500',1.35)]
for y,label,prob,joint,starty in leaves:box(ax,8.25,y-.3,2.35,.7,label+'\n联合 '+joint,orange if '未' not in label else gray);arrow(ax,(5.5,starty),(8.25,y+.05));ax.text(6.9,(starty+y)/2+.1,prob,ha='center',fontsize=10)
save(fig,'03_tree.png')
p=np.linspace(.001,.5,200);posterior=.9*p/(.9*p+.1*(1-p));fig,ax=plt.subplots(figsize=(8,3.4));ax.plot(p,posterior,color=blue);ax.scatter([.001,.02,.1,.5],[1/112,9/58,.5,.9],color=orange,zorder=4)
for x,y,text in [(.02,9/58,'2% → 15.5%'),(.1,.5,'10% → 50%'),(.5,.9,'50% → 90%')]:ax.annotate(text,(x,y),xytext=(8,-23 if x==.5 else 10),textcoords='offset points',ha='right' if x==.5 else 'left',fontsize=10)
ax.set(xlabel='缺陷先验 P(D)',ylabel='已知标记后的缺陷概率',title='固定两种标记率：P(F|D)=0.9，P(F|非D)=0.1',xlim=(0,.54),ylim=(0,1));ax.grid(alpha=.2);fig.tight_layout();save(fig,'04_prior.png')
fig,axs=plt.subplots(1,2,figsize=(8.5,3.2));axs[0].axis('off');t=axs[0].table(cellText=[[0,0,0,'1/4'],[0,1,1,'1/4'],[1,0,1,'1/4'],[1,1,0,'1/4']],colLabels=['X','Y','Z','权重'],cellLoc='center',loc='center',bbox=[.04,.1,.92,.82]);t.auto_set_font_size(False);t.set_fontsize(12);axs[0].set_title('Z=(X+Y)除2的余数')
for (i,j),cell in t.get_celld().items():cell.set_edgecolor('#a4b1bc');cell.set_facecolor('#eef4f8' if i==0 else 'white')
axs[1].bar(['真实三者均1','三个边缘相乘'],[0,1/8],color=[orange,blue]);axs[1].set(ylim=(0,.16),ylabel='概率',title='两两独立仍不足');axs[1].text(0,.008,'0',ha='center');axs[1].text(1,.135,'1/8',ha='center');fig.tight_layout();save(fig,'05_xor.png')
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,title in zip(axs,['世界甲：X=U，Y=X','世界乙：X=U，Y=U']):ax.axis('off');ax.set(xlim=(0,5),ylim=(0,5));ax.set_title(title)
for ax in axs:box(ax,.3,3.2,1.2,.8,'U');box(ax,2,3.2,1.2,.8,'X');box(ax,3.7,3.2,1.2,.8,'Y');arrow(ax,(1.5,3.6),(2,3.6));ax.text(2.5,2.6,'观测：(0,0)、(1,1) 各 1/2',ha='center',fontsize=10)
arrow(axs[0],(3.2,3.6),(3.7,3.6));arrow(axs[1],(.9,4.15),(4.3,4.15));axs[1].text(2.6,4.35,'共同背景决定Y',ha='center',fontsize=10)
for ax,value in zip(axs,['P(Y=1 | do(X=1)) = 1','P(Y=1 | do(X=1)) = 1/2']):box(ax,.25,1,4.55,1,'替换X规则为常数1\n'+value,orange)
fig.tight_layout();save(fig,'06_intervention.png')
