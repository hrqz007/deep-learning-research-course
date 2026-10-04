"""Original risk illustrations, optional NumPy/matplotlib figure builder."""
from pathlib import Path
import csv
from experiment import require_teaching_inputs
require_teaching_inputs()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiment import fit,risk,empirical_risk,candidates,load_training
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
f=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if f.exists():font_manager.fontManager.addfont(str(f));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'axes.unicode_minus':False})
C=['#227e9f','#e99532','#ad496d','#54a57e'];train=load_training(HERE/'data/hand_training.csv')
def save(n):plt.tight_layout();plt.savefig(OUT/n,bbox_inches='tight',facecolor='white');plt.close()

fig,ax=plt.subplots(1,2,figsize=(9,3.1));x=np.arange(8);q=np.array([.1]*4+[.9]*4)
ax[0].bar(x,1-q,color=C[0],label='Y=0');ax[0].bar(x,q,bottom=1-q,color=C[1],label='Y=1');ax[0].set(xlabel='输入类别 x',ylabel='条件概率',xticks=x,title='每个 x 的标签分布');ax[0].legend(fontsize=9)
ax[1].bar(x,np.minimum(q,1-q)/8,color=C[3]);ax[1].set(xlabel='输入类别 x',ylabel='对总体风险的贡献',xticks=x,title='最优规则仍有噪声风险0.1',ylim=(0,.02))
save('01_population.png')

fig,ax=plt.subplots(3,1,figsize=(8,5.1),sharex=True)
for a,kind,col in zip(ax,('constant','threshold','lookup'),C):
 h=fit(kind,train);a.step(x,h,where='mid',color=col,lw=2,label=f'{kind} 拟合结果');a.scatter([xx for xx,yy in train],[yy for xx,yy in train],c='black',s=45,zorder=4,label='4个训练观察');a.set(ylim=(-.2,1.3),yticks=[0,1],ylabel='预测/标签',title=f'训练误差 {float(empirical_risk(h,train)):.2f}，总体风险 {float(risk(h)):.2f}');a.legend(fontsize=9,loc='upper left',bbox_to_anchor=(1.01,1),borderaxespad=0,frameon=False)
ax[-1].set(xlabel='输入类别 x',xticks=x)
save('02_three_fits.png')

fig,ax=plt.subplots(figsize=(8,3.3));ts=list(range(9));models=[tuple(int(xx>=t) for xx in range(8)) for t in ts]
ax.plot(ts,[float(empirical_risk(h,train)) for h in models],'o-',color=C[0],label='这4条数据的经验风险')
ax.plot(ts,[float(risk(h)) for h in models],'s-',color=C[2],label='源总体精确风险')
ax.axvline(1,color=C[0],ls=':',alpha=.6);ax.axvline(4,color=C[2],ls=':',alpha=.6);ax.set(xlabel='阈值 t，预测 1 当 x≥t',ylabel='错误概率/比例',xticks=ts,ylim=(-.03,.8));ax.legend(fontsize=9)
save('03_erm_population.png')

fig,ax=plt.subplots(figsize=(9,3.1));ax.axis('off')
items=[(.12,'训练集 D','只据它拟合3个候选'),(.5,'验证集 V','按预定规则选择1个'),(.87,'封存测试集 T','方案冻结后只做评估')]
for at,title,body in items:ax.text(at,.69,title,ha='center',fontsize=16,color='#111');ax.text(at,.4,body,ha='center',fontsize=11)
for a,b in [(0.24,.37),(.64,.75)]:ax.annotate('',xy=(b,.62),xytext=(a,.62),arrowprops={'arrowstyle':'->','lw':2,'color':C[0]})
ax.text(.5,.07,'总体风险只作为合成实验的审计参照，不进入拟合或选择',ha='center',fontsize=11,color=C[2]);save('04_data_roles.png')

with (HERE/'outputs/summary.csv').open() as stream:summ=list(csv.DictReader(stream))
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
for kind,col in zip(('constant','threshold','lookup'),C):
 rs=[r for r in summ if r['class']==kind];ns=[int(r['n']) for r in rs]
 ax[0].plot(ns,[float(r['mean_train_risk']) for r in rs],'o-',color=col,label=kind)
 ax[1].plot(ns,[float(r['mean_population_risk']) for r in rs],'o-',color=col,label=kind)
for a,title in zip(ax,['800次重复的平均训练误差','800次重复的平均总体风险']):
 a.set(xscale='log',xticks=[8,32,128],xticklabels=['8','32','128'],xlabel='训练样本量 n',ylabel='错误比例/概率',ylim=(0,.55),title=title);a.axhline(.1,color='#666',ls=':',label='最小总体风险 0.1');a.legend(fontsize=9)
save('05_sample_capacity.png')

fig,ax=plt.subplots(figsize=(8,3.4));rs=[r for r in summ if r['class']=='validation_selected'];ns=np.arange(3);w=.23
for i,key,col,label in zip([-1,0,1],['mean_validation_risk','mean_test_risk','mean_population_risk'],[C[0],C[1],C[2]],['已用于选择的验证误差','独立测试误差','总体精确风险']):ax.bar(ns+i*w,[float(r[key]) for r in rs],width=w,color=col,label=label)
ax.set(xticks=ns,xticklabels=['n=8','n=32','n=128'],ylabel='800次平均错误比例/概率',ylim=(0,.23));ax.legend(fontsize=9);save('06_selection_test.png')

fig,ax=plt.subplots(figsize=(8,3.2));labels=['只学1个标签','恒定预测0.5','知道真实概率0.9'];bias=[0,.16,0];var=[.09,0,0];noise=[.09]*3
ax.bar(labels,noise,label='新标签噪声',color='#bbbbbb');ax.bar(labels,var,bottom=noise,label='训练集导致的预测方差',color=C[0]);ax.bar(labels,bias,bottom=np.array(noise)+var,label='预测偏差平方',color=C[2]);ax.set(ylabel='x=4处期望平方损失',ylim=(0,.31));ax.legend(fontsize=9)
save('07_bias_variance.png')

fig,ax=plt.subplots(1,2,figsize=(9,3.3));h=tuple(int(xx>=4) for xx in range(8))
ax[0].plot(x,q,'o-',color=C[0],label='源 P(Y=1|X)');ax[0].plot(x,1-q,'s-',color=C[2],label='变更后 Q(Y=1|X)');ax[0].step(x,h,where='mid',ls='--',color='#222',label='冻结的旧规则');ax[0].set(xlabel='x',ylabel='概率/输出',xticks=x);ax[0].legend(fontsize=8)
ax[1].bar(['源总体 P','目标总体 Q'],[.1,.9],color=[C[0],C[2]]);ax[1].set(ylabel='同一冻结规则的总体风险',ylim=(0,1));save('08_distribution_shift.png')
