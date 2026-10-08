"""Rebuild original figures from retained experimental evidence."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(R/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
fp=Path(os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if fp.exists():font_manager.fontManager.addfont(str(fp));fn=font_manager.FontProperties(fname=str(fp)).get_name()
else:fn='Noto Sans CJK SC'
plt.rcParams.update({'font.family':fn,'axes.unicode_minus':False,'font.size':10,'figure.dpi':170})
F=R/'figures';F.mkdir(exist_ok=True);r=json.loads((R/'outputs/results.json').read_text());C=['#2665a8','#d8892c','#24816f','#98589a']
def save(f,n):f.tight_layout();f.savefig(F/n,bbox_inches='tight');plt.close(f)
def flow(labels,n,title):
 f,ax=plt.subplots(figsize=(10,1.8));ax.axis('off');ax.set(xlim=(-.6,len(labels)*2.7-.5),ylim=(-.35,.9))
 for i,label in enumerate(labels):
  ax.text(i*2.7,.25,label,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=.65','fc':'#edf4fa','ec':C[i%4]})
  if i:ax.annotate('',xy=(i*2.7-1.1,.25),xytext=((i-1)*2.7+1.05,.25),arrowprops={'arrowstyle':'->','color':C[i%4]})
 ax.set_title(title);save(f,n)

flow(['原始请求\n形状与有限值','训练集统计量\n标准化','eval模型\n固定批次导出','类别与状态\n拒绝非法输入'],'01_contract.png','推理契约的四个边界（概念图）')
f,ax=plt.subplots(figsize=(8,3.5));ax.bar(['正确封装','跳过标准化'],[r['accuracy'],r['wrong_preprocessing_accuracy']],color=C[:2]);ax.set(ylim=(0,1.05),ylabel='测试准确率',title='同一模型与384个测试样本：只有输入处理改变');save(f,'02_preprocessing.png')
f,ax=plt.subplots(figsize=(8,3.5));
for k,c in zip(['eager_1','export_1','eager_32','export_32'],C):
 v=np.sort(r['timings'][k]['samples_us']);ax.plot(v,np.arange(1,len(v)+1)/len(v),label=k,color=c)
ax.set(xlabel='模型调用时间（微秒）',ylabel='经验累计比例',title='CPU单线程稳态120次调用，不含请求解析与网络');ax.legend();save(f,'03_latency.png')
f,axes=plt.subplots(1,2,figsize=(9,3.5));keys=list(r['timings'])
axes[0].bar(keys,[r['timings'][k]['p95_us'] for k in keys],color=C);axes[0].set(ylabel='p95模型调用时间（微秒）');axes[0].tick_params(axis='x',rotation=20)
axes[1].bar(keys,[r['timings'][k]['throughput_examples_per_second'] for k in keys],color=C);axes[1].set(ylabel='闭环批调用样本/秒');axes[1].tick_params(axis='x',rotation=20);save(f,'04_batch.png')
f,axes=plt.subplots(1,2,figsize=(9,3.5))
for i,b in enumerate(['1','32']):
 axes[0].scatter([i]*5,r['cold'][b]['load_plus_first_call_ms'],color=C[i])
 axes[1].scatter([i]*3,[v['spawn_to_exit_ms'] for v in r['fresh_process'][b]['observations']],color=C[i])
for ax in axes:ax.set(xticks=[0,1],xticklabels=['batch1','batch32'],ylabel='毫秒')
axes[0].set_title('暖进程：重新加载加首调（5次）',fontsize=10)
axes[1].set_title('新进程：启动至退出含首调（3次）',fontsize=10)
save(f,'05_load.png')
f,ax=plt.subplots(figsize=(8,3.5));q=r['queue_simulation'];ax.plot(q['arrival_ms'],q['response_ms'],'o-',color=C[0],label='响应时间');ax.axhline(q['service_ms'],color=C[1],ls='--',label='恒定服务时间');ax.set(xlabel='请求到达时刻（毫秒）',ylabel='时间（毫秒）',title='排队公式模拟：0.2ms到达间隔，0.5ms处理一次');ax.legend();save(f,'06_queue.png')
