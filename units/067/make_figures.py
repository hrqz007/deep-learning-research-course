"""Original rank-semantic diagrams; communication plot is a labelled model."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(R/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path=Path(os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if font_path.exists():font_manager.fontManager.addfont(str(font_path));font_name=font_manager.FontProperties(fname=str(font_path)).get_name()
else:font_name='Noto Sans CJK SC'
plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'font.size':10,'figure.dpi':160})
F=R/'figures';F.mkdir(exist_ok=True);r=json.loads((R/'outputs/results.json').read_text());C=['#2563a6','#df8432','#23836c','#934d98']
def save(f,n):f.tight_layout();f.savefig(F/n,bbox_inches='tight');plt.close(f)
f,ax=plt.subplots(figsize=(10,3.4));ax.axis('off');ax.set(xlim=(-.5,11),ylim=(-.4,3.5))
for i,n in enumerate([20,30,78]):
 y=2.5-i
 for x,label in [(0,f'rank {i}\n{n}个样本'),(2.5,'同一初始参数\n本地前向反向'),(5.5,'加权归约\n得到同一全局梯度'),(8.5,'各副本同样\n更新一次')]:
  ax.text(x,y,label,ha='center',va='center',fontsize=9,bbox={'boxstyle':'round,pad=.45','fc':'#f3f6fa','ec':C[i]})
 for x1,x2 in [(1,1.5),(3.5,4.4),(6.6,7.6)]:ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops={'arrowstyle':'->','color':C[i]})
ax.text(5,3.25,'概念时序；本课用单进程数值模拟，不是实测多进程图',ha='center');save(f,'01_ranks.png')
f,axes=plt.subplots(1,2,figsize=(9,3.4));axes[0].bar(['rank0','rank1','rank2'],[20/128,30/128,78/128],color=C[:3]);axes[0].axhline(1/3,color='black',ls='--',label='错误等rank权重');axes[0].set(ylabel='在全局样本均值中的权重');axes[0].legend(fontsize=8)
keys=['weighted','rank_mean','double_divide'];vals=[r['gradient_checks'][k]['relative_error'] for k in keys];axes[1].bar(keys,vals,color=C[:3]);axes[1].set(ylabel='相对梯度误差',title='不等本地样本数');save(f,'02_weighting.png')
f,axes=plt.subplots(2,1,figsize=(9,3.8));
for ax,key in zip(axes,['False','True']):
 ax.axis('off');ax.set(xlim=(0,12),ylim=(-.7,2.7))
 for rank,idxs in enumerate(r['sampler'][key]):
  ax.text(0,2-rank,f'rank {rank}',va='center')
  for j,idx in enumerate(idxs):ax.text(2+j*2.2,2-rank,str(idx),ha='center',va='center',bbox={'boxstyle':'round,pad=.3','fc':'#ffe9d5' if idx in [0,1] and key=='False' else '#e5eff8','ec':'#999'})
 ax.set_title('N=10 world=3 drop_last='+key+'（橙色ID在填充模式中重复）',fontsize=10)
save(f,'03_sampler.png')
f,ax=plt.subplots(figsize=(8,3.5));row=r['runs'][0];
for mode,c in zip(['full','weighted','rank_mean','duplicated_rank'],C):ax.plot([h['step'] for h in row['history']],[h[mode]['test_loss'] for h in row['history']],label=mode,color=c,ls='--' if mode=='weighted' else '-')
ax.set(xlabel='优化器更新次数',ylabel='独立测试交叉熵',title='种子11 全批与正确加权曲线重合');ax.legend();save(f,'04_learning.png')
f,ax=plt.subplots(figsize=(8,3.4));modes=['full','weighted','rank_mean','duplicated_rank']
for i,mode in enumerate(modes):ax.scatter([i]*3,[z['history'][-1][mode]['test_accuracy'] for z in r['runs']],color=C[i],s=50)
ax.set(xticks=range(4),xticklabels=modes,ylabel='最终测试准确率',ylim=(.5,1),title='三个初始化种子；没有网络通信');save(f,'05_results.png')
f,ax=plt.subplots(figsize=(8,3.3));rows=r['communication_model'];ax.plot([z['world'] for z in rows],[z['seconds']*1000 for z in rows],'o-',color=C[0]);ax.set(xlabel='world size',ylabel='理想环形all-reduce估计（ms）',title='100 MiB 梯度 / 12.5 GB/s / 每段5微秒：纯公式示例');ax.grid(alpha=.2);save(f,'06_communication.png')
