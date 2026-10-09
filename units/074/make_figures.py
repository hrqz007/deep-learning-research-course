"""由真实outputs和明示数学示例重建原创彩色图，不下载素材。"""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-dl-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
font=FontProperties(fname=os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
C=['#2563eb','#e76f51','#10a380','#9b5de5']
R=json.loads((ROOT/'outputs/results.json').read_text())
F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(name):
    plt.tight_layout();plt.savefig(F/name,dpi=170,bbox_inches='tight');plt.close()
def flow(labels,name):
    fig,ax=plt.subplots(figsize=(10,2.3));ax.axis('off')
    for i,s in enumerate(labels):
        xx=.12+i*.255
        ax.text(xx,.52,s,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.65',fc=['#dbeafe','#d1fae5','#ffedd5','#ede9fe'][i],ec=C[i]))
        if i<3:ax.annotate('',xy=(xx+.17,.52),xytext=(xx+.08,.52),arrowprops=dict(arrowstyle='->',color='#475569',lw=2))
    save(name)

A=np.load(ROOT/'outputs/trials.npz')
flow(['固定总体与假设类\n抽样前写清H','IID抽取训练集\n每条损失在[0,1]','每个假设集中\n并集界覆盖全部H','数据内选ERM\n同时保证仍有效'],'01_uniform.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.4));ax[0].plot(A['x'],A['prob'],'o-',color=C[0]);ax[0].set(xlabel='输入格点x',ylabel='P(Y=1|x)',ylim=(0,1));ax[1].plot(A['thresholds'],A['true_risks'],'o-',color=C[1]);ax[1].set(xlabel='预定阈值',ylabel='精确总体风险');save('02_population.png')
fig,ax=plt.subplots(figsize=(8,3.6));ns=np.array([r['n'] for r in R['rows']]);eps=[r['epsilon'] for r in R['rows']]
ax.plot(ns,eps,'o-',label='一致界半径',color=C[1]);ax.plot(ns,[r['mean_uniform_gap'] for r in R['rows']],'o-',label='400次平均实际最大偏差',color=C[0]);ax.set(xscale='log',xlabel='样本量n',ylabel='绝对风险偏差');ax.legend();save('03_bound.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.5))
for k,n in enumerate([20,100]):
    gap=A[f'n{n}'][:,0];ax[k].hist(gap,bins=18,color=C[0],alpha=.8);ax[k].axvline(R['rows'][k]['epsilon'],color=C[1],label='delta=0.05界');ax[k].set(title=f'n={n}, 400次抽样',xlabel='max |经验风险-总体风险|',ylabel='次数');ax[k].legend()
save('04_trials.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for field,label,c in [('mean_train','阈值ERM训练风险',C[0]),('mean_true','阈值ERM总体风险',C[1]),('memory_mean_train','记忆表训练风险',C[2]),('memory_mean_true','记忆表总体风险',C[3])]:ax[0].plot(ns,[r[field] for r in R['rows']],'o-',label=label,color=c)
ax[0].set(xscale='log',xlabel='n',ylabel='400次平均风险');ax[0].legend(fontsize=8)
for field,label,c in [('wrong_singleton_radius','错误地代入H=1',C[2]),('epsilon','预定23阈值',C[0]),('full_table_radius','全部2^21标签表',C[1])]:ax[1].plot(ns,[r[field] for r in R['rows']],'o-',label=label,color=c)
ax[1].set(xscale='log',xlabel='n',ylabel='半径；左曲线无保证');ax[1].legend(fontsize=8);save('05_selection.png')
