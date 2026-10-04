"""Nine original mechanism figures; plotting dependencies are optional for learners."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiment import bernoulli_likelihood, bernoulli_log_likelihood, gaussian_density
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
fonts = [p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p]
if not fonts:
    raise RuntimeError('Noto Sans CJK font required to build Chinese figures')
font_manager.fontManager.addfont(fonts[0])
plt.rcParams.update({'font.family': [font_manager.FontProperties(fname=fonts[0]).get_name(), 'DejaVu Sans'],
                     'font.size':11, 'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})
B,G,R,O = '#2466a8','#24816c','#b94258','#c88620'
def save(fig,name):
    fig.tight_layout(pad=1.25)
    fig.savefig(OUT/name,dpi=175,facecolor='white')
    plt.close(fig)
def grid(ax):
    ax.grid(alpha=.17)
def lin(a,b,n=401):
    return [a+(b-a)*i/(n-1) for i in range(n)]

fig,axes=plt.subplots(1,2,figsize=(10.7,3.35))
ax=axes[0]
ax.bar([0,1],[.25,.75],color=[B,G],width=.28)
ax.set(xticks=[0,1],ylim=(0,1.08),xlabel='数据取值 y',ylabel='概率质量',title='固定 p=3/4，沿数据轴看分布')
ax.text(0,.3,'1/4',ha='center');ax.text(1,.8,'3/4',ha='center');grid(ax)
ax=axes[1];ps=lin(0,1)
ax.plot(ps,[bernoulli_likelihood([1,1,1,0],p) for p in ps],c=R)
ax.scatter([.25,.5,.75],[3/256,1/16,27/256],c=[B,O,G],zorder=3)
ax.set(xlabel='参数 p',ylabel='L(p; 1,1,1,0)',title='固定观测，沿参数轴看似然');grid(ax)
save(fig,'01_mass_likelihood.png')

fig,ax=plt.subplots(figsize=(10.7,3));ax.axis('off')
for i,(text,color) in enumerate([('模型参数\np = k/M',B),('均匀抽编号\nJ ∈ {0,…,M−1}',G),('记录二值结果\ny = 1 当 J < k',O),('固定观测序列\n比较各候选 p',R)]):
    x=.115+i*.25
    ax.text(x,.57,text,ha='center',va='center',fontsize=12,color=color,
            bbox={'boxstyle':'round,pad=.65','facecolor':'#f7fafc','edgecolor':color})
    if i<3: ax.annotate('',xy=(x+.16,.57),xytext=(x+.085,.57),arrowprops={'arrowstyle':'->','color':'#64748b','lw':1.5})
ax.text(.5,.12,'生成时先定参数；计算似然时先固定已经看到的数据。',ha='center',color='#435064')
save(fig,'02_generator.png')

fig,axes=plt.subplots(1,2,figsize=(10.7,3.2))
ps=lin(.001,.999)
for values,label,c in [([0]*4,'全0',B),([1,1,1,0],'3个1与1个0',G),([1]*4,'全1',R)]:
    axes[0].plot(lin(0,1),[bernoulli_likelihood(values,p) for p in lin(0,1)],label=label,c=c)
    axes[1].plot(ps,[bernoulli_log_likelihood(values,p) for p in ps],label=label,c=c)
axes[0].set(xlabel='p',ylabel='似然',title='端点是否允许取决于观测')
axes[1].set(xlabel='p（图中不含端点）',ylabel='对数似然',ylim=(-22,1),title='不可能观测的端点值为 −∞')
for ax in axes:ax.legend(fontsize=9);grid(ax)
save(fig,'03_boundaries.png')

fig,axes=plt.subplots(1,3,figsize=(10.7,3.15))
for ax,vals,title in zip(axes,[[.1,.9],[.9,.1],[.5,.5]],['仅归一化似然','明确先验','由先验与似然算后验']):
    ax.bar([0,1],vals,color=[B,R],width=.55);ax.set(xticks=[0,1],xticklabels=['p=1/4','p=3/4'],ylim=(0,1.1),title=title)
    for i,v in enumerate(vals):ax.text(i,v+.035,f'{v:.1f}',ha='center')
    grid(ax)
axes[0].set_ylabel('候选参数上的权重')
save(fig,'04_posterior.png')

fig,axes=plt.subplots(1,2,figsize=(10.7,3.25))
ax=axes[0];ax.bar([0,1],[.25,.75],width=.24,color=[B,G]);ax.set(xticks=[0,1],ylim=(0,1.1),xlabel='离散 y',ylabel='点的质量',title='质量：事件概率是选中柱高之和');ax.text(.5,.94,'P(Y∈{0,1}) = 1',ha='center')
ax=axes[1];xs=lin(-.2,.5);ys=[4 if 0<=x<=.25 else 0 for x in xs];ax.plot(xs,ys,c=R);ax.fill_between([.05,.15],[4,4],color=G,alpha=.3);ax.set(xlabel='连续 x（秒）',ylabel='密度（每秒）',ylim=(0,5),title='密度：概率是高度 × 宽度');ax.text(.10,1.6,'0.4',ha='center');ax.text(.18,4.35,'高度 4 > 1',ha='center')
for ax in axes:grid(ax)
save(fig,'05_mass_density.png')

fig,axes=plt.subplots(1,2,figsize=(10.7,3.4))
xs=lin(-3,3,801)
for mu,c in [(-1,B),(0,G),(1,R)]:axes[0].plot(xs,[gaussian_density(x,mu,.5) for x in xs],label=f'μ={mu}, σ=0.5',c=c)
axes[0].set(xlabel='x',ylabel='密度',title='位置参数平移曲线');axes[0].legend(fontsize=9)
xs=lin(-1,1,801)
for sigma,c in [(.1,R),(.25,G),(.5,B)]:axes[1].plot(xs,[gaussian_density(x,0,sigma) for x in xs],label=f'μ=0, σ={sigma}',c=c)
axes[1].axhline(1,color='#64748b',ls=':',lw=1)
axes[1].set(xlabel='x',ylabel='密度',title='尺度变小：更窄、更高、总面积仍为1');axes[1].legend(fontsize=9)
for ax in axes:grid(ax)
save(fig,'06_gaussian_parameters.png')

fig,axes=plt.subplots(1,2,figsize=(10.7,3.25))
for ax,n in zip(axes,[4,16]):
    xs=lin(-2,2);ax.plot(xs,[gaussian_density(x) for x in xs],c=B)
    h=4/n;mids=[-2+(i+.5)*h for i in range(n)];vals=[gaussian_density(x) for x in mids]
    ax.bar(mids,vals,width=h,alpha=.25,color=G,edgecolor=G)
    area=sum(vals)*h
    ax.set(xlabel='x',ylabel='密度',title=f'区间 [−2,2]，{n}个中点矩形：{area:.6f}');grid(ax)
save(fig,'07_quadrature.png')

fig,axes=plt.subplots(1,2,figsize=(10.7,3.2))
for ax,n in zip(axes,[20,2000]):
    xs=lin(-.12,.12,801);ax.plot(xs,[gaussian_density(x,0,.01) for x in xs],c=B)
    h=2/n;mids=[-1+(i+.5)*h for i in range(n)];vals=[gaussian_density(x,0,.01) for x in mids]
    ax.scatter(mids,vals,c=R,s=18 if n==20 else 4,zorder=3)
    ax.set(xlim=(-.12,.12),ylim=(0,43),xlabel='x（放大峰附近）',ylabel='密度',title=f'[−1,1]内{n}个中点，面积 {sum(vals)*h:.6f}');grid(ax)
save(fig,'08_missed_peak.png')

fig,axes=plt.subplots(1,2,figsize=(10.7,3.25))
xs=lin(-1,3)
axes[0].plot(xs,[gaussian_density(x,1,.5) for x in xs],c=B)
axes[0].set(xlabel='数据 x，固定 μ=1、σ=0.5',ylabel='f(x; μ,σ)',title='这一条沿 x 轴的曲线积分为1')
mus=lin(-1,3)
axes[1].plot(mus,[gaussian_density(1,mu,.5)**2 for mu in mus],c=R)
axes[1].set(xlabel='参数 μ，固定数据 (1,1)、σ=0.5',ylabel='L(μ; 1,1)',title='这一条沿 μ 轴的曲线无需积分为1')
for ax in axes:grid(ax)
save(fig,'09_density_likelihood.png')
