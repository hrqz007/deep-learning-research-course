"""Optional original figures; intentionally guarded fixed teaching configuration."""
from pathlib import Path
from experiment import require_teaching_inputs,exact_bootstrap,paired_t7,DEFAULT_ROWS
require_teaching_inputs()
import csv,json,random,math
from fractions import Fraction as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
f=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if f.exists():font_manager.fontManager.addfont(str(f));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'axes.unicode_minus':False})
C=['#287f9e','#da923c','#aa4f74','#479f7c'];r=json.loads((HERE/'outputs/results.json').read_text())
def save(name):plt.tight_layout();plt.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close()

fig,ax=plt.subplots(figsize=(9,3.3));ax.axis('off')
for x,title,body,col in zip([.15,.5,.85],['只重复训练','只重采测试实体','重采数据并训练'],['固定 D 与 T\n改变随机状态 S\n条件种子平均效果','冻结两个模型\n抽新的目标实体\n测试抽样不确定性','新 D、新 T、新 S\n多层变化需对应设计\n更广泛算法比较'],C):
 ax.text(x,.82,title,ha='center',fontsize=16,color=col);ax.text(x,.47,body,ha='center',va='center',linespacing=1.7,bbox={'boxstyle':'round,pad=.7','facecolor':col,'alpha':.1,'edgecolor':col})
ax.text(.5,.05,'重复单位决定所能推广的范围；不能将训练种子当成新数据集',ha='center',color=C[2]);save('01_targets.png')

fig,ax=plt.subplots(1,2,figsize=(9,3.6));a=[x[1] for x in DEFAULT_ROWS];b=[x[2] for x in DEFAULT_ROWS];d=[x-y for x,y in zip(a,b)]
for i,(x,y) in enumerate(zip(a,b)):ax[0].plot([i,i],[x,y],color='#aaa');
ax[0].scatter(range(8),a,label='A',color=C[0],zorder=3);ax[0].scatter(range(8),b,label='B',color=C[1],marker='s',zorder=3);ax[0].set(xlabel='配对种子ID',ylabel='错误率（%）',xticks=range(8),ylim=(10,45));ax[0].legend(loc='lower right')
ax[1].bar(range(8),d,color=[C[3] if v>0 else C[2] for v in d]);ax[1].axhline(0,color='#444',lw=1);ax[1].axhline(2,color=C[0],ls=':',label='平均差异 2');ax[1].set(xlabel='配对种子ID',ylabel='A−B（百分点）',xticks=range(8),ylim=(-3,7.5));ax[1].legend(loc='upper right');save('02_pairs.png')

rng=random.Random(1729);fig,ax=plt.subplots(figsize=(8.5,4.8));hits=0
for i in range(50):
 ds=[rng.gauss(2,3) for _ in range(8)];m=sum(ds)/8;v=sum((x-m)**2 for x in ds)/7;h=2.365*math.sqrt(v/8);hit=m-h<=2<=m+h;hits+=hit
 ax.plot([m-h,m+h],[i,i],lw=1.6,color=C[0] if hit else C[2]);ax.scatter([m],[i],s=8,color=C[0] if hit else C[2])
ax.axvline(2,color='#222',ls='--');ax.set(xlabel='均值区间端点（每次8个独立正态差异）',ylabel='独立重复编号',title=f'真实均值2；本次50条中{hits}条覆盖，临界值2.365已舍入');save('03_coverage.png')

fig,ax=plt.subplots(figsize=(9,3.1));ax.axis('off');inds=[3,3,0,4,1,6,2,7]
for j in range(8):
 x=.08+j*.117
 ax.text(x,.85,str(j),ha='center',fontsize=14,bbox={'boxstyle':'round','fc':'#e9f1f5','ec':'none'});ax.text(x,.65,f'{d[j]:+d}',ha='center',color=C[0],fontsize=14)
 ax.text(x,.35,str(inds[j]),ha='center',fontsize=14,bbox={'boxstyle':'round','fc':'#faecd9','ec':'none'});ax.text(x,.15,f'{d[inds[j]]:+d}',ha='center',color=C[1],fontsize=14)
ax.text(.01,.85,'ID',ha='right');ax.text(.01,.65,'差',ha='right');ax.text(.01,.35,'抽ID',ha='right');ax.text(.01,.15,'差',ha='right');ax.set_title('整行重采样：抽到(3,3,0,4,1,6,2,7)，均值=18/8=2.25');save('04_resampling.png')

exact=exact_bootstrap(d);xs=[s/8 for s in exact['counts']];probs=[c/8**8 for c in exact['counts'].values()]
with (HERE/'outputs/bootstrap.csv').open() as f:boots=[float(F(row['mean_difference'])) for row in csv.DictReader(f)]
from collections import Counter
observed=Counter(boots)
fig,ax=plt.subplots(figsize=(9,3.6));ax.bar(xs,probs,width=.2,color=C[0],alpha=.6,label='全部8^8样本压缩后的精确质量');ax.plot(xs,[observed[x]/len(boots) for x in xs],'.',color=C[1],label='4000次重采样相对频率')
for x in [.25,3.75]:ax.axvline(x,color=C[2],ls='--')
ax.set(xlabel='bootstrap平均差异（百分点）',ylabel='概率/频率',title='固定八行之后的条件重采样分布');ax.legend(fontsize=9,loc='upper left');save('05_bootstrap.png')

with (HERE/'outputs/rare_coverage.csv').open() as f:rare=list(csv.DictReader(f))
fig,ax=plt.subplots(1,2,figsize=(9,3.7));ks=[int(x['positive_count']) for x in rare];p=[float(F(x['sample_probability'])) for x in rare]
ax[0].bar(ks,p,color=C[0]);ax[0].set(yscale='log',xlabel='八个样本中稀有值20的个数K',ylabel='真实抽样概率（对数轴）',xticks=ks,title='K服从Binomial(8,0.01)')
for row in rare:
 k=int(row['positive_count']);lo=float(F(row['lower']));hi=float(F(row['upper']));col=C[3] if row['covers_true_mean']=='1' else C[2];ax[1].plot([lo,hi],[k,k],'o-',color=col,markersize=3)
ax[1].axvline(.2,color='#111',ls='--',label='真实均值0.2');ax[1].set(xlabel='条件percentile区间',ylabel='K',yticks=ks,title='仅K=1或2时覆盖');ax[1].legend(fontsize=9,loc='lower right');save('06_rare_failure.png')

fig,ax=plt.subplots(1,2,figsize=(9,3.5));vals=[-2,0,2,4]
for j,v in enumerate(vals):ax[0].scatter(range(j*8,(j+1)*8),[v]*8,color=C[j],s=25,label=f'实体{j+1}')
ax[0].set(xlabel='复制的32行位置',ylabel='观察差异',title='每个实体内8行完全相同');ax[0].legend(fontsize=8,ncol=2,loc='upper left',bbox_to_anchor=(0,1.0))
ax[1].bar(['4个实体正确单位','32行错误独立'],[math.sqrt(5/3),math.sqrt(5/31)],color=[C[3],C[2]]);ax[1].set(ylabel='均值标准误估计',ylim=(0,1.6),title='同一个均值1，不同的方差估计');save('07_dependence.png')

fig,ax=plt.subplots(1,2,figsize=(9,3.6));summ=r['selection_summary'];xx=np.arange(3)
ax[0].bar(xx-.18,[x['mean_selected_validation'] for x in summ],width=.36,label='挑过的验证噪声',color=C[1]);ax[0].bar(xx+.18,[x['mean_independent_test'] for x in summ],width=.36,label='独立复测噪声',color=C[0]);ax[0].plot(xx,[float(F(x['theoretical_selected_mean'])) for x in summ],'o',color='#222',label='验证理论期望');ax[0].set(xticks=xx,xticklabels=['1','5','20'],xlabel='候选数m',ylabel='2000次平均噪声',ylim=(-.15,1.35));ax[0].legend(fontsize=8,loc='upper left')
ms=list(range(1,41));ax[1].plot(ms,[1-.95**m for m in ms],color=C[2]);ax[1].axhline(.05,ls=':',color='#555');ax[1].set(xlabel='独立有效检验数m',ylabel='至少一次误报概率',ylim=(0,1),title='另一个机制：每项误报0.05');save('08_selection.png')

fig,ax=plt.subplots(2,1,figsize=(8.5,5.4));ax[0].axvspan(-3,3,color=C[3],alpha=.15,label='预定等效范围 ±3');
for i,(lo,hi,label) in enumerate([(-10,10,'示意：证据很不精确'),(-.5,.5,'示意：支持小差异'),(-.19,4.19,'本表t计算：仍不确定')]):ax[0].plot([lo,hi],[i,i],'o-',color=C[i],lw=2);ax[0].text(10.7,i,label,va='center',fontsize=9)
ax[0].axvline(0,color='#666',ls=':');ax[0].set(xlim=(-11,18),yticks=[],xlabel='效果区间（百分点）；解释以方法有效为前提');ax[0].legend(fontsize=9,loc='upper left',bbox_to_anchor=(0,1.25))
ax[1].axis('off');table=ax[1].table(cellText=[['10','9'],['8','4']],rowLabels=['P关闭','P开启'],colLabels=['Q关闭','Q开启'],cellLoc='center',bbox=[.12,.24,.62,.67]);table.auto_set_font_size(False);table.set_fontsize(13)
ax[1].text(.8,.65,'P减少2',color=C[0],fontsize=12);ax[1].text(.8,.35,'或减少5',color=C[2],fontsize=12);ax[1].text(.43,.05,'损失越小越好；差中差=5−2=3，作用依赖上下文',ha='center',fontsize=10);save('09_effect_ablation.png')
