"""Optional original figure builder; core experiment does not import these packages."""
from pathlib import Path
import csv,math,random
from experiment import require_teaching_config
require_teaching_config()  # Before figure directories or files can be changed.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiment import exact_mean_distribution
HERE=Path(__file__).resolve().parent
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.dpi':160})
COLORS=['#237b9a','#ef9b32','#ad4e70','#59a785']
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
def save(name):
    plt.tight_layout();plt.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close()

fig,ax=plt.subplots(1,2,figsize=(9,3.1))
ax[0].bar([0,2,6],[.5,.25,.25],width=.7,color=COLORS[:3]);ax[0].set(xlabel='损失 X',ylabel='概率质量',xticks=[0,2,6],ylim=(0,.62),title='先看每个取值有多大权重')
ax[1].bar([0,2,6],[0,.5,1.5],width=.7,color=COLORS[:3]);ax[1].axhline(2,color='#444',ls='--',label='贡献之和 μ = 2');ax[1].set(xlabel='损失 X',ylabel='x × P(X=x)',xticks=[0,2,6],ylim=(0,2.3),title='加权贡献相加得到期望');ax[1].legend(fontsize=9)
save('01_expectation.png')

fig,ax=plt.subplots(1,2,figsize=(9,3))
for a,x,p,title in [(ax[0],[1,3],[.5,.5],'相同均值 2，方差 1'),(ax[1],[0,2,6],[.5,.25,.25],'相同均值 2，方差 6')]:
    a.bar(x,p,color=COLORS[0],width=.6);a.axvline(2,color=COLORS[2],ls='--');a.set(xlim=(-.8,6.8),ylim=(0,.62),xlabel='损失 X',ylabel='概率质量',title=title)
    for v,pr in zip(x,p):a.annotate('',xy=(v,.57),xytext=(2,.57),arrowprops={'arrowstyle':'<->','color':COLORS[1]})
save('02_variance.png')

fig,ax=plt.subplots(1,3,figsize=(9,2.8))
x=np.array([-1,0,1])
for a,y,title in zip(ax,[x,-x,x*x],['正协方差 2/3','负协方差 -2/3','零协方差，仍然依赖']):
    a.scatter(x,y,s=100,color=COLORS[0]);a.axvline(0,color='#aaa',lw=.7);a.axhline(y.mean(),color='#aaa',lw=.7);a.set(xlabel='X',ylabel='Y',xticks=x,xlim=(-1.4,1.4),ylim=(-1.4,1.4),title=title)
save('03_covariance.png')

fig,axes=plt.subplots(2,2,figsize=(9,5.1))
for a,n in zip(axes.flat,[1,4,16,64]):
    d=exact_mean_distribution([0,0,2,6],n);xx=[float(x) for x,p in d];pp=[float(p) for x,p in d]
    a.bar(xx,pp,width=min(.6,.8*2/n),color=COLORS[0]);a.axvline(2,color=COLORS[2],ls='--');a.set(xlim=(-.3,6.3),xlabel='一整批的平均损失',ylabel='精确概率质量',title=f'n = {n}，均值方差 = 6/{n}')
save('04_exact_sampling.png')

fig,ax=plt.subplots(2,1,figsize=(9,3.9))
for a,groups,title in [(ax[0],64,'64个独立抽样单元 → 64行'),(ax[1],8,'8个独立抽样单元，每个复制8次 → 64行')]:
    a.set(xlim=(-1,64),ylim=(-.8,1),yticks=[],xticks=[0,8,16,24,32,40,48,56,63],title=title,xlabel='记录行位置（从0开始）')
    for i in range(64):a.scatter(i,0,s=60,c=[COLORS[(i if groups==64 else i//8)%4]],marker='s')
    if groups==8:
        for g in range(8):
            a.plot([8*g,8*g+7],[.45,.45],color=COLORS[g%4],lw=3);a.text(8*g+3.5,.63,f'组{g+1}',ha='center',fontsize=9)
    else:a.text(31.5,.5,'每行重新抽一个编号；颜色只帮助辨认位置',ha='center',fontsize=10)
save('05_sampling_units.png')

with (HERE/'outputs/sampling_summary.csv').open() as stream:
    summ=list(csv.DictReader(stream))
fig,ax=plt.subplots(figsize=(8,3.5));ns=np.array([16,32,64,128])
ax.plot(ns,np.sqrt(6/ns),color=COLORS[0],label='IID理论 SE');ax.plot(ns,np.sqrt(48/ns),color=COLORS[2],label='复制8次理论 SE')
for mode,c in [('iid',COLORS[0]),('copied_groups',COLORS[2])]:
    rr=[r for r in summ if r['mode']==mode];ax.scatter([int(r['n']) for r in rr],[float(r['empirical_sd']) for r in rr],color=c,s=55,marker='x',label=f'{mode} 4000次批均值的SD')
ax.axhline(math.sqrt(6),ls=':',color='#666',label='单个观测的 SD = √6');ax.set(xlabel='每批记录行数 n',ylabel='损失单位',xticks=ns,ylim=(0,2.7));ax.legend(fontsize=9,ncol=2)
save('06_sd_se.png')

fig,ax=plt.subplots(figsize=(8,3.3))
r=next(r for r in summ if r['n']=='64' and r['mode']=='copied_groups')
values=[float(r['true_se']),float(r['empirical_sd']),float(r['rms_naive_se']),float(r['rms_unit_correct_se'])]
ax.bar(['理论真实SE','4000批均值的SD','逐行估计SE的RMS','逐组估计SE的RMS'],values,color=[COLORS[0],COLORS[0],COLORS[2],COLORS[3]])
for i,v in enumerate(values):ax.text(i,v+.02,f'{v:.3f}',ha='center')
ax.set(ylabel='损失单位',ylim=(0,1.04),title='每批64行、8个独立组，每组复制8次')
save('07_se_estimation.png')

fig,ax=plt.subplots(figsize=(8,3.4))
rng=random.Random(772);ns=np.arange(1,513)
for i in range(4):
    xs=[rng.choice([0,0,2,6]) for _ in ns];ax.plot(ns,np.cumsum(xs)/ns,lw=1,alpha=.75,label='IID路径' if i==0 else None)
ax.plot(ns,np.full(512,6),ls='--',color=COLORS[2],label='一次抽到6后永久复制的路径');ax.axhline(2,color='#222',ls=':',label='总体 μ=2')
ax.set(xlabel='累计记录数 n',ylabel='累计样本均值',ylim=(-.15,6.3));ax.legend(fontsize=9)
save('08_lln_paths.png')

fig,ax=plt.subplots(figsize=(8,3.3));n=100;p=.001
probs=[math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(4)];tail=1-sum(probs)
ax.bar(['0次','1次','2次','3次','至少4次'],probs+[max(0,tail)],color=COLORS[0]);ax.set(ylabel='精确二项概率',title='100个独立样本，事件概率 p=0.001',ylim=(0,1))
for i,v in enumerate(probs+[tail]):ax.text(i,v+.02,f'{v:.6f}',ha='center',fontsize=10)
save('09_rare_event.png')
