"""Original diagrams for one fixed, verified SGD report."""
from pathlib import Path
import json,hashlib,itertools
ROOT=Path(__file__).resolve().parent
EXPECTED={'data/config.json': '3f31d1b740f66a4224b3be2ec56f451d3181e5ccd7a17b6749c5114a2ec51c53', 'data/order.json': 'f90d8e73c6d8c7db11ce5e39f58ff17c6c1fe55d9b9e42d78ce9b011b65553df', 'data/samples.csv': '9b2e5c6d0ac24372075c01f817150430c171d1995363051ac5a861496e67f402', 'outputs/budget-comparison.csv': '472e9fc8df08153e706105811b6d5420a26b69729c9483eb314a612a4141b7ac', 'outputs/first-batch-trace.json': '7895b1484318e25472b914e6ee9384fc42882c591f8e849fd6c2e3628e16c287', 'outputs/measured-timing.json': '59a46c8bc19c1af72d364397e9d8c1b5d4ba219d5456a963c22398e2120a3f34', 'outputs/network-sampling.json': '3351de626b2af616d279c66f02d80b790fb27633ddb267fde2ac63889e819c3e', 'outputs/sampling-exact.json': 'bbb35f893431c17acefdaad8866c561b58f161613ee4f461e6ad70e1778bfd3c', 'outputs/scalar-theory.json': 'c4d44ebfdf31e4293382f3bf10eb70b2248ab10d2f3a7501c41594eed54b2586', 'outputs/summary.json': 'bea11b597a2528abf55a40eb69ce2a8da8f695313c930871a49897b530a89f68', 'outputs/trajectories.json': '47acd15135a2211f8e4de9ef87908c2efe0941f39dcc494f7233d5c3be9d1dcd'}
for name,digest in EXPECTED.items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Fixed report changed: '+name)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
import numpy as np
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
C=['#166897','#d07725','#288a65','#92539b'];out=ROOT/'figures';out.mkdir(exist_ok=True)
def save(f,name):f.savefig(out/name,bbox_inches='tight',facecolor='white');plt.close(f)
s=json.loads((ROOT/'outputs/sampling-exact.json').read_text());runs=json.loads((ROOT/'outputs/trajectories.json').read_text());summary=json.loads((ROOT/'outputs/summary.json').read_text());timing=json.loads((ROOT/'outputs/measured-timing.json').read_text());theory=json.loads((ROOT/'outputs/scalar-theory.json').read_text())
f,ax=plt.subplots(figsize=(8.5,4));g=np.array(s['fixed_gradients'])
for i,z in enumerate(g):ax.arrow(0,0,z[0],z[1],color=C[i],width=.025,length_includes_head=True,head_width=.14);ax.text(z[0]*1.05,z[1]*1.08,f'g{i+1}={tuple(int(v) for v in z)}',ha='center')
ax.scatter([0],[0],color='#18202a',zorder=5);ax.set_xlim(-3.9,3.9);ax.set_ylim(-2.8,2.8);ax.axhline(0,color='#ccc',lw=.6);ax.axvline(0,color='#ccc',lw=.6);ax.set_xlabel('梯度坐标1');ax.set_ylabel('梯度坐标2');ax.set_title('固定同一参数点：单样本梯度不必指向总体梯度');ax.text(0,.4,'总体均值 = (0,0)',ha='center');save(f,'01_vectors.png')
f,ax=plt.subplots(figsize=(8.5,3.7));b=np.arange(1,5)
ax.plot(b,5/b,'o-',label='独立有放回：5/B',color=C[0]);ax.plot(b,5*(4-b)/(3*b),'s-',label='均匀不放回：5(4-B)/(3B)',color=C[2]);ax.plot(b,[5]*4,'^--',label='完全重复同一样本：5',color=C[1]);ax.set_xticks(b);ax.set_xlabel('批量大小 B');ax.set_ylabel('第1梯度坐标的方差');ax.set_title('N=4 的精确有限总体枚举');ax.legend(fontsize=9);ax.grid(alpha=.2);save(f,'02_variance.png')
f,axs=plt.subplots(1,2,figsize=(9,3.4))
for ax,replace,title,col in zip(axs,[True,False],['有放回 16 种有序结果','不放回 6 个等概率子集'],C):
 choices=list(itertools.product(range(4),repeat=2) if replace else itertools.combinations(range(4),2));values=[float(g[list(ids),0].mean()) for ids in choices];v,c=np.unique(values,return_counts=True);ax.bar(v,c/len(choices),color=col,width=.65);ax.set_xlabel('B=2 的平均梯度坐标1');ax.set_ylabel('概率');ax.set_title(title);ax.set_ylim(0,.4)
f.tight_layout();save(f,'03_distributions.png')
f,ax=plt.subplots(figsize=(9,3));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3)
for x,txt,col in [(0,'目标 {-1,+1}\n初始 θ=0\n学习率 1/2',C[0]),(3.5,'先看到 +1\n梯度 -1\n更新 θ=1/2',C[1]),(7,'剩余只能是 -1\n下一梯度 3/2\n此处全梯度却是1/2',C[2])]:ax.text(x+1.4,1.55,txt,ha='center',va='center',bbox={'boxstyle':'round,pad=.8','fc':col+'22','ec':'none'})
for x in [2.8,6.3]:ax.annotate('',xy=(x+.7,1.5),xytext=(x,1.5),arrowprops={'arrowstyle':'->','lw':1.8})
ax.set_title('不放回训练中的条件偏差：参数已经依赖历史',loc='left');save(f,'04_conditional.png')
f,axs=plt.subplots(1,2,figsize=(9,3.7))
for b,col in zip([1,4,16],C):
 rows=[r for r in theory if r['batch_size']==b and r['learning_rate']==.5];axs[0].plot([r['step'] for r in rows],[r['expected_excess_loss'] for r in rows],label=f'B={b}',color=col)
for lr,col in zip([.1,1.,2.,2.1],C):
 rows=[r for r in theory if r['batch_size']==4 and r['learning_rate']==lr];axs[1].plot([r['step'] for r in rows],[r['expected_excess_loss'] for r in rows],label=f'η={lr}',color=col)
for a in axs:a.set_xlabel('优化器更新步数');a.set_ylabel('精确期望超额损失');a.legend();a.grid(alpha=.2)
axs[0].set_title('固定η=0.5：噪声平台');axs[1].set_title('固定B=4：大步长可不稳定');axs[1].set_yscale('log');f.tight_layout();save(f,'05_noise_floor.png')
f,axs=plt.subplots(1,2,figsize=(9,3.7))
for b,col in zip([1,4,16,32],C):
 r=runs[f'equal_examples_b{b}']['history'];axs[0].plot([z['examples'] for z in r],[z['full_training_loss'] for z in r],label=f'B={b}',color=col)
 r=runs[f'equal_steps_b{b}']['history'];axs[1].plot([z['step'] for z in r],[z['full_training_loss'] for z in r],label=f'B={b}',color=col)
for a in axs:a.set_ylabel('同一完整训练目标 F');a.legend();a.grid(alpha=.2)
axs[0].set_xlabel('已呈现的训练样本次数');axs[0].set_title('相同1280次样本预算，η=0.02');axs[1].set_xlabel('优化器更新步数');axs[1].set_title('相同40步，样本次数不同');f.tight_layout();save(f,'06_budgets.png')
f,ax=plt.subplots(figsize=(8.5,3.8))
for b,col in zip([1,4,16,32],C):
 r=runs[f'linear_scaled_examples_b{b}']['history'];ax.plot([z['examples'] for z in r],[z['full_training_loss'] for z in r],label=f'B={b}, η={.02*b/4:g}',color=col)
ax.set_title('改变学习率比较协议后，轨迹接近但不完全相同');ax.set_xlabel('训练样本呈现次数');ax.set_ylabel('完整训练目标 F');ax.legend();ax.grid(alpha=.2);save(f,'07_scaling.png')
f,axs=plt.subplots(1,2,figsize=(9,3.6));labels=[str(r['batch_size']) for r in timing['rows']];med=[r['median_seconds'] for r in timing['rows']]
lo=[r['median_seconds']-r['minimum_seconds'] for r in timing['rows']];hi=[r['maximum_seconds']-r['median_seconds'] for r in timing['rows']]
axs[0].bar(labels,med,color=C,yerr=[lo,hi],capsize=4);axs[0].set_ylabel('完整训练调用耗时 秒');axs[0].set_title('5次实测：柱为中位数，线为范围')
axs[1].bar(labels,[r['examples_per_second'] for r in timing['rows']],color=C);axs[1].set_ylabel('1280 / 中位秒数');axs[1].set_title('本实现的样本呈现吞吐')
for ax in axs:ax.set_xlabel('批量大小 B')
f.tight_layout();save(f,'08_timings.png')
f,ax=plt.subplots(figsize=(8.5,3.7));ns=json.loads((ROOT/'outputs/network-sampling.json').read_text());ax.plot([r['batch_size'] for r in ns['cases']],[r['covariance_trace'] for r in ns['cases']],'o-',color=C[0],label='全部子集枚举');ax.plot([r['batch_size'] for r in ns['cases']],[r['theory_covariance_trace'] for r in ns['cases']],'x--',color=C[1],label='有限总体公式');ax.set_xticks([1,2,4,8]);ax.set_xlabel('从固定8行抽取的批量 B');ax.set_ylabel('13维梯度协方差的迹');ax.set_title('真实小网络、固定同一θ：不放回方差也符合公式');ax.legend();ax.grid(alpha=.2);save(f,'09_network_variance.png')
print('9 figures rebuilt after fixed report guards')
