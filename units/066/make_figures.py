"""Original precision diagrams and real experiment plots."""
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
f,ax=plt.subplots(figsize=(9,3));start=np.zeros(3)
for j,(label,vals) in enumerate([('符号位',[1,1,1]),('指数位',[8,5,8]),('小数位',[23,10,7])]):
 ax.barh(['FP32','FP16','BF16'],vals,left=start,color=C[j],label=label)
 for y,(v,s) in enumerate(zip(vals,start)):ax.text(s+v/2,y,str(v),ha='center',va='center',color='white')
 start+=vals
ax.set(xlabel='位数（隐含的前导1不计入小数位）',xlim=(0,34));ax.legend(loc='upper right');save(f,'01_bits.png')
f,ax=plt.subplots(figsize=(9,2.8));ax.axis('off');labels=['FP32 参数与输入','autocast 按算子选型','计算损失','反向得 FP32 参数梯度','FP32 参数更新']
for i,s in enumerate(labels):
 ax.text(i*2,1,s,ha='center',va='center',bbox={'boxstyle':'round,pad=.6','fc':['#e4eef9','#fff0db','#e8f3ed','#e4eef9','#e4eef9'][i],'ec':C[i%4]},fontsize=9)
 if i<4:ax.annotate('',xy=(i*2+1.1,1),xytext=(i*2+.85,1),arrowprops={'arrowstyle':'->'})
ax.set(xlim=(-1.1,9.1),ylim=(0,2));ax.text(4,.25,'本课实际路径：CPU BF16 autocast；不把整个模型 .half()',ha='center');save(f,'02_path.png')
f,ax=plt.subplots(figsize=(8,3.5));keys=['bf16','accumulate','checkpoint','wrong_equal'];v=[r['comparisons'][k]['gradient_relative_error'] for k in keys];ax.bar(keys,[max(z,1e-10) for z in v],color=C);ax.set(yscale='log',ylabel='相对梯度误差',title='同一参数点对照 FP32 全批（零值画在 1e-10 并注明）');ax.text(2,2e-10,'精确 0',ha='center');save(f,'03_gradient.png')
f,axes=plt.subplots(1,2,figsize=(9,3.4));a=r['saved_tensors'];axes[0].bar(['完整保存','重计算'],[a[k]['saved_nonparameter_unique_storage_bytes']/1024 for k in ['full','checkpoint']],color=C[:2]);axes[0].set(ylabel='前向保留的非参数存储（KiB）',title='不是进程峰值或显存峰值');axes[1].bar(['完整保存','重计算'],[r['timing'][k]['median_seconds']*1000 for k in ['full','checkpoint']],color=C[:2]);axes[1].set(ylabel='前向与反向中位耗时（ms）',title='CPU单线程 独立计时');save(f,'04_tradeoff.png')
f,ax=plt.subplots(figsize=(8,3.5));names=['Dropout保留随机状态','Dropout不保留随机状态','BatchNorm微批累积'];vals=[r['dropout']['preserve_rng_relative_error'],r['dropout']['no_preserve_relative_error'],r['batchnorm_accumulation_relative_error']];bars=ax.barh(names,vals,color=[C[2],C[1],C[3]]);ax.bar_label(bars,fmt='%.3f');ax.set(xlabel='相对梯度误差（各自对照对应全批/非重算网络）',xlim=(0,.7));save(f,'05_counterexamples.png')
f,axes=plt.subplots(1,2,figsize=(10,3.5));modes=['fp32','bf16','accumulate','checkpoint']
for mode,c in zip(modes,C):
 row=next(z for z in r['runs'] if z['seed']==11 and z['mode']==mode);axes[0].plot([h['step'] for h in row['history']],[h['test_mse_fp32_inference'] for h in row['history']],label=mode,color=c,ls='--' if mode in ['accumulate','checkpoint'] else '-')
 vals=[z['final_test_mse'] for z in r['runs'] if z['mode']==mode];axes[1].scatter([modes.index(mode)]*len(vals),vals,color=c)
axes[0].set(xlabel='更新步数',ylabel='测试 MSE（统一FP32推理）',title='种子11 曲线可能重合');axes[0].legend(fontsize=8);axes[1].set(xticks=range(4),xticklabels=modes,ylabel='最终测试 MSE',title='每个点是一个初始化种子');save(f,'06_training.png')
