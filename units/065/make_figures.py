"""Original conceptual diagrams and plots from measured results; no stock images."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
R=Path(__file__).resolve().parent;F=R/'figures';F.mkdir(exist_ok=True)
from matplotlib import font_manager
font_path=Path(os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    font_name=font_manager.FontProperties(fname=str(font_path)).get_name()
else:font_name='Noto Sans CJK SC'
plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'font.size':10,'figure.dpi':160})
C=['#2563a6','#df8432','#23836c','#934d98','#627180']
def save(f,n):f.tight_layout();f.savefig(F/n,bbox_inches='tight');plt.close(f)
r=json.loads((R/'outputs/results.json').read_text())
f,ax=plt.subplots(figsize=(9,2.6));ax.set(xlim=(0,10),ylim=(-.8,2),yticks=[0,1],yticklabels=['GPU 队列示意','CPU 主线程示意'],xlabel='时间（概念示意，无实测比例）')
ax.broken_barh([(0,.8),(.8,.8),(1.6,.8)],(.8,.4),facecolors=C[:3]);ax.broken_barh([(.2,2),(2.2,2),(4.2,1.8)],(-.2,.4),facecolors=C[:3]);ax.axvline(2.4,color='#c0392b',ls='--');ax.axvline(6,color='#111',ls='--')
ax.text(2.5,1.4,'未同步：只量提交',color='#c0392b');ax.text(6.1,.6,'同步后：工作完成');ax.text(6.1,1.3,'蓝 前向 / 橙 反向 / 绿 更新');save(f,'01_async.png')
a=r['memory_after_training'];f,ax=plt.subplots(figsize=(8,3.4));labels=['参数','梯度','Adam 状态'];vals=[a[k]/1024 for k in ['parameter_bytes','gradient_bytes','optimizer_tensor_bytes']];bars=ax.bar(labels,vals,color=C[:3]);ax.bar_label(bars,fmt='%.2f KiB');ax.set(ylabel='逻辑张量载荷（KiB）',ylim=(0,max(vals)*1.3),title='实际计数不含激活、缓存与临时工作区');save(f,'02_memory.png')
f,axes=plt.subplots(1,2,figsize=(9,3.4));rows=r['measurements'];axes[0].boxplot([[t*1000 for t in z['seconds_per_step_blocks']] for z in rows],tick_labels=['向量化','逐行循环']);axes[0].set(ylabel='毫秒 / step',title='相同批量 64 与计算目标');axes[1].bar(['向量化','逐行循环'],[z['samples_per_second'] for z in rows],color=C[:2]);axes[1].set(ylabel='样本 / 秒',title='独立于 profiler 的稳态测量');save(f,'03_timing.png')
f,ax=plt.subplots(figsize=(8,3.3));rows=r['batch_sweep'];ax.plot([z['batch'] for z in rows],[z['samples_per_second'] for z in rows],'o-',color=C[0]);ax.set(xlabel='批量',ylabel='样本 / 秒',title='CPU 单线程 同一结构同一更新算法');ax.grid(alpha=.2);save(f,'04_batch.png')
f,ax=plt.subplots(figsize=(8,3.8));rows=r['profile'][:8];ax.barh([z['operator'] for z in rows][::-1],[z['self_cpu_us']/1000 for z in rows][::-1],color=C[0]);ax.set(xlabel='self CPU 时间（ms，5个被观测step）',title='含分析器开销，仅用于定位');save(f,'05_profile.png')
f,axes=plt.subplots(1,2,figsize=(9,3.1));h=r['history'];axes[0].plot([v['epoch'] for v in h],[v['train_loss'] for v in h],color=C[0]);axes[1].plot([v['epoch'] for v in h],[v['test_accuracy'] for v in h],color=C[2]);axes[0].set(xlabel='epoch',ylabel='训练交叉熵');axes[1].set(xlabel='epoch',ylabel='独立测试准确率',ylim=(.5,1));save(f,'06_learning.png')
