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

flow(['数据与许可\n实体拆分','代码与配置\n环境版本','训练状态\n权重与随机数','结果与限制\n模型卡'],'01_assets.png','每个结果都要能追溯上游输入（概念图）')
f,ax=plt.subplots(figsize=(8,3.5));
for key,c,style in [('continuous',C[0],'-'),('resumed',C[2],'--'),('weights_only',C[1],':')]:
 h=r['histories'][key];ax.plot([a['epoch'] for a in h],[a['train_loss'] for a in h],label=key,color=c,ls=style,lw=2)
ax.axvline(12,color='gray',alpha=.5);ax.set(xlabel='epoch',ylabel='训练平均交叉熵',title='中断点后：完整恢复重合，权重重启偏离');ax.legend();save(f,'02_resume.png')
f,axes=plt.subplots(1,2,figsize=(9,3.5));keys=list(r['metrics']);axes[0].bar(keys,[r['metrics'][k]['accuracy'] for k in keys],color=C[:3]);axes[0].set(ylim=(.85,1),ylabel='测试准确率（截断纵轴）');axes[0].tick_params(axis='x',rotation=15)
axes[1].bar(['完整恢复','仅权重'],[r['resume_max_parameter_gap'],r['weights_only_max_parameter_gap']],color=[C[2],C[1]]);axes[1].set(ylabel='与连续训练的最大参数差');save(f,'03_results.png')
flow(['资产原始字节','SHA256摘要','受信清单比较','一致或报错'],'04_integrity.png','完整性检查只比较字节，不证明数据合法或结论正确')
f,ax=plt.subplots(figsize=(8,3.4));ax.bar(['原始样本','附加重复','总记录'],[384,4,388],color=C[:3]);ax.set(ylabel='条数',title='不同记录ID仍可能重复：本课检测4条完全相同特征');save(f,'05_duplicates.png')
f,ax=plt.subplots(figsize=(9,3.6));ax.axis('off');labels=['推理权重','续训状态','实验包'];items=['参数与缓冲区','+ 优化器、随机状态、进度','+ 数据、代码、配置、许可、环境'];
for i,(a,b) in enumerate(zip(labels,items)):
 ax.text(.02,.8-i*.3,a,transform=ax.transAxes,color=C[i],fontsize=13,weight='bold');ax.text(.25,.8-i*.3,b,transform=ax.transAxes,fontsize=11)
ax.set_title('保存层级对应不同承诺（概念图）');save(f,'06_levels.png')
