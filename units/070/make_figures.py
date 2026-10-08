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

flow(['论文主张\n适用任务','可观测量\n指标定义','控制变量\n预算与拆分','复现实验\n差异与边界'],'01_claims.png','主张到证据的映射（概念图）')
f,axes=plt.subplots(2,2,figsize=(9,5));
for ax,mode,c in zip(axes.flat,['baseline','dropout','weight_decay','both'],C):
 row=next(z for z in r['runs'] if z['seed']==11 and z['mode']==mode);h=row['history'];ax.plot([a['epoch'] for a in h],[a['train_nll'] for a in h],color=c,label='训练');ax.plot([a['epoch'] for a in h],[a['validation_nll'] for a in h],color=c,ls='--',label='验证');ax.axvline(row['selected_epoch'],color='gray',lw=1);ax.set(title=mode,xlabel='更新次数',ylabel='NLL');ax.legend(fontsize=8)
save(f,'02_curves.png')
f,ax=plt.subplots(figsize=(8,3.5));modes=['baseline','dropout','weight_decay','both']
for i,mode in enumerate(modes):ax.scatter([i]*5,[a['test']['nll'] for a in r['runs'] if a['mode']==mode],color=C[i],s=40)
ax.set(xticks=range(4),xticklabels=modes,ylabel='验证选择checkpoint的测试NLL',title='五个初始化种子；同一数据拆分');save(f,'03_runs.png')
f,ax=plt.subplots(figsize=(8,3.5));p=r['paired_dropout_minus_baseline'];ax.scatter([11,23,37,51,67],p['values'],color=C[0]);ax.axhline(0,color='gray');ax.axhline(p['mean'],color=C[1],ls='--',label='均值');ax.set(xlabel='配对初始化种子',ylabel='dropout - baseline 测试NLL',title='负值有利于dropout；仅此数据与设置');ax.legend();save(f,'04_paired.png')
f,ax=plt.subplots(figsize=(8,3.4));v=r['factorial_interaction'];ax.bar([str(x) for x in [11,23,37,51,67]],v['values'],color=C[2]);ax.axhline(0,color='gray');ax.set(xlabel='初始化种子',ylabel='L11 - L10 - L01 + L00',title='正则化交互：零才符合简单可加效应');save(f,'05_interaction.png')
f,ax=plt.subplots(figsize=(8,3.4));
for i,mode in enumerate(modes):ax.scatter([i]*5,[a['selected_epoch'] for a in r['runs'] if a['mode']==mode],color=C[i])
ax.set(xticks=range(4),xticklabels=modes,ylabel='验证集选中的epoch',ylim=(0,165),title='最大预算相同，不意味着选中模型训练步数相同');save(f,'06_selection.png')
