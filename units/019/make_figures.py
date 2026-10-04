"""Eight original Chinese mechanism figures, not empirical model benchmarks."""
from pathlib import Path
import math
import os
import tempfile
_cache = tempfile.TemporaryDirectory(prefix="dl019-plots-")
os.environ.setdefault("MPLCONFIGDIR", _cache.name)
os.environ.setdefault("XDG_CACHE_HOME", _cache.name)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiment import entropy, cross_entropy, kl, load_distributions
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p]
if not fonts: raise RuntimeError('Noto Sans CJK font needed for Chinese figures')
font_manager.fontManager.addfont(fonts[0])
plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=fonts[0]).get_name(),'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
B,G,R,O,K='#2466a8','#24816c','#b94258','#c88620','#3d4756'
def save(fig,name):
    fig.tight_layout(pad=1.2);fig.savefig(OUT/name,dpi=175,facecolor='white');plt.close(fig)
def grid(ax):ax.grid(alpha=.16,axis='y');ax.set_axisbelow(True)
def lin(a,b,n=401):return [a+(b-a)*i/(n-1) for i in range(n)]
D=load_distributions()

fig,ax=plt.subplots(1,2,figsize=(10.7,3.15))
qs=lin(.002,1);ax[0].plot(qs,[-math.log(q) for q in qs],c=B)
for q,c in [(.6,O),(.99,G)]:
    ax[0].scatter([q],[-math.log(q)],color=c,zorder=3)
    ax[0].annotate(f'q={q}\n损失={-math.log(q):.3f}',xy=(q,-math.log(q)),xytext=(q-.25,1.3 if q<.9 else 2.6),arrowprops={'arrowstyle':'->','color':c},color=c)
ax[0].set(xlabel='已观察真实类别的预测概率 q',ylabel='−ln q（nat）',title='越确定却越错，损失越大',ylim=(-.15,6.6));grid(ax[0])
for i,(p,c,label) in enumerate([([.6,.25,.15],O,'模型甲'),([.99,.005,.005],G,'模型乙')]):
    ax[1].bar([j+(i-.5)*.34 for j in range(3)],p,width=.32,color=c,label=label)
ax[1].set(xticks=[0,1,2],xticklabels=['A（真实）','B','C'],ylim=(0,1.12),ylabel='预测概率',title='两者都选A，损失却不相同');ax[1].legend(fontsize=9);grid(ax[1]);save(fig,'01_log_loss.png')

fig,ax=plt.subplots(1,2,figsize=(10.7,3.15))
p,q=D['main_p'],D['main_q']
for i,(vals,c,label) in enumerate([(p,B,'P 加权'),(q,O,'Q 评分')]):ax[0].bar([j+(i-.5)*.32 for j in range(3)],vals,width=.3,color=c,label=label)
ax[0].set(xticks=[0,1,2],xticklabels=list('ABC'),ylim=(0,.65),ylabel='概率质量',title='在同一状态集合上比较');ax[0].legend(fontsize=9);grid(ax[0])
for i,(vals,c,label) in enumerate([([-x*math.log2(x) for x in p],B,'−p log₂ p'),([-x*math.log2(y) for x,y in zip(p,q)],O,'−p log₂ q')]):ax[1].bar([j+(i-.5)*.32 for j in range(3)],vals,width=.3,color=c,label=label)
ax[1].set(xticks=[0,1,2],xticklabels=list('ABC'),ylim=(0,1.23),ylabel='加权贡献（bit）',title='评分不同，权重都来自P');ax[1].legend(fontsize=9);grid(ax[1]);save(fig,'02_mass_contributions.png')

fig,ax=plt.subplots(1,2,figsize=(10.7,3.0))
ax[0].barh([0],[1.5],color=B,height=.45,label='熵 H(P)');ax[0].barh([0],[.25],left=[1.5],color=G,height=.45,label='KL D(P∥Q)')
ax[0].text(.75,0,'1.5',color='white',ha='center',va='center',fontsize=14);ax[0].text(1.625,0,'0.25',color='white',ha='center',va='center',fontsize=11)
ax[0].set(xlim=(0,1.9),yticks=[],xlabel='交叉熵（bit）',title='H(P,Q) = 1.75 bit');ax[0].legend(loc='lower left',fontsize=9);ax[0].set_ylim(-.8,.65)
vals=[.5,0,-.25];ax[1].bar(range(3),vals,color=[G,K,R],width=.5);ax[1].axhline(0,c=K,lw=.8)
for j,v in enumerate(vals):ax[1].text(j,v+(.035 if v>=0 else -.07),f'{v:g}',ha='center')
ax[1].set(xticks=[0,1,2],xticklabels=list('ABC'),ylim=(-.42,.7),ylabel='KL分项（bit）',title='0.5 + 0 − 0.25 = 0.25');grid(ax[1]);save(fig,'03_decomposition.png')

fig,ax=plt.subplots(1,2,figsize=(10.7,3.05));ts=lin(.1,3.2)
ax[0].plot(ts,[math.log(t) for t in ts],c=B,label='ln t');ax[0].plot(ts,[t-1 for t in ts],c=O,label='t − 1');ax[0].scatter([1],[0],c=G,zorder=4);ax[0].set(xlabel='t > 0',ylabel='函数值',title='ln t ≤ t − 1');ax[0].legend(fontsize=9)
ax[1].plot(ts,[t-1-math.log(t) for t in ts],c=R);ax[1].scatter([1],[0],c=G);ax[1].axhline(0,c=K,lw=.7);ax[1].set(xlabel='t > 0',ylabel='g(t)',title='差值在 t=1 唯一达到0');ax[1].text(.6,.85,'递减',color=R);ax[1].text(2.1,.85,'递增',color=R)
for a in ax:grid(a)
save(fig,'04_log_inequality.png')

fig,ax=plt.subplots(1,2,figsize=(10.7,3.1))
for i,(v,c,l) in enumerate([(D['zero_p'],B,'P'),(D['zero_q'],O,'Q')]):ax[0].bar([j+(i-.5)*.32 for j in range(3)],v,width=.3,color=c,label=l)
ax[0].set(xticks=[0,1,2],xticklabels=['A','B：Q漏覆盖','C：共同为零'],ylim=(0,1.15),ylabel='概率质量',title='先检查正支持，再计算对数');ax[0].legend(fontsize=9);grid(ax[0]);ax[1].axis('off')
ax[1].text(.06,.8,'正向 D(P∥Q) = +∞',color=R,fontsize=17);ax[1].text(.06,.63,'按P加权，B的正质量碰到分母0',fontsize=11)
ax[1].text(.06,.38,'反向 D(Q∥P) = ln 2',color=G,fontsize=17);ax[1].text(.06,.21,'按Q加权，只在A计算 ln(1 / 0.5)',fontsize=11)
save(fig,'05_support.png')

fig,ax=plt.subplots(figsize=(10.7,3.05));es=list(range(1,13));eps=[10.**(-e) for e in es]
ax.plot(es,[kl([.5,.5],[1-e,e]) for e in eps],'-o',c=B,label='正向 D(P∥Qε)');ax.plot(es,[kl([1-e,e],[.5,.5]) for e in eps],'-s',c=G,label='反向 D(Qε∥P)');ax.axhline(math.log(2),c=K,ls=':',label='ln 2')
ax.set(xlabel='−log10 ε（越往右，ε越小）',ylabel='KL（nat）',title='P=(1/2,1/2)，Qε=(1−ε,ε)');ax.legend(fontsize=9,loc='upper left');grid(ax);save(fig,'06_epsilon.png')

fig,ax=plt.subplots(1,2,figsize=(10.7,3.5))
for i,(name,c,l) in enumerate([('target',K,'目标P'),('broad',B,'宽'),('left',G,'左'),('right',O,'右')]):ax[0].bar([j+(i-1.5)*.19 for j in range(3)],D[name],width=.18,color=c,label=l)
ax[0].set(xticks=[0,1,2],xticklabels=['A','B（低谷）','C'],ylim=(0,1.23),ylabel='概率质量',title='允许模型只有宽、左、右三种');ax[0].legend(fontsize=9,ncol=2);grid(ax[0])
for i,(direction,c,l) in enumerate([('forward',B,'正向'),('reverse',R,'反向')]):
    vals=[kl(D['target'],D[n]) if direction=='forward' else kl(D[n],D['target']) for n in ['broad','left','right']]
    xs=[j+(i-.5)*.34 for j in range(3)];ax[1].bar(xs,vals,width=.32,color=c,label=l)
    for x,v in zip(xs,vals):ax[1].text(x,v+.03,f'{v:.3f}',ha='center',fontsize=9)
ax[1].set(xticks=[0,1,2],xticklabels=['宽','左','右'],ylim=(0,1.65),ylabel='KL（nat）',title='正向选宽；反向左右并列');ax[1].legend(fontsize=9);grid(ax[1]);save(fig,'07_directional.png')

fig,ax=plt.subplots(1,2,figsize=(10.7,3.05));no=.999**100
ax[0].bar([0,1],[no,1-no],color=[O,B],width=.48)
for j,v in enumerate([no,1-no]):ax[0].text(j,v+.03,f'{100*v:.2f}%',ha='center')
ax[0].set(xticks=[0,1],xticklabels=['完全没见到','至少见到一次'],ylim=(0,1.09),ylabel='批次概率',title='罕见状态概率0.001，独立抽100次');grid(ax[0]);ax[1].axis('off')
ax[1].text(.08,.81,'模型 Q=(1,0)',fontsize=16,color=K);ax[1].text(.08,.57,'若批次没有罕见状态：经验损失 = 0',fontsize=11,color=G);ax[1].text(.08,.33,'对真实 P=(0.999,0.001)：总体损失 = +∞',fontsize=11,color=R);ax[1].text(.08,.1,'有限批次可能完全遮住支持集错误',fontsize=12,color=K)
save(fig,'08_unseen.png')
print('8 original figures generated')
