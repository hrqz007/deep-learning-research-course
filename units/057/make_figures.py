"""Regenerate only figures from recorded results; no training or metric changes."""
from pathlib import Path
import argparse,json,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT=Path(os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if not FONT.is_file():raise RuntimeError('Install Noto Sans CJK or set DL_CJK_FONT to a readable CJK font')
fp=FontProperties(fname=str(FONT));plt.rcParams.update({'font.family':fp.get_name(),'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.dpi':160})
from matplotlib import font_manager
font_manager.fontManager.addfont(str(FONT))
COLORS=['#2873a3','#de8737','#4b9573','#9c67a6','#d65559']
def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
def main(results,output):
    source=Path(results);r=json.loads(source.read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8.5,2.9));ax.axis('off')
    for i,(title,text,color) in enumerate([('训练40组','160文档\n仅训练梯度',COLORS[0]),('验证12组','48文档\n固定0/40/120步',COLORS[1]),('测试12组','48文档\n固定120步一次',COLORS[2])]):
        ax.add_patch(plt.Rectangle((i*3,0),2.6,1.7,facecolor=color,alpha=.14));ax.text(i*3+1.3,1.23,title,ha='center',fontsize=14);ax.text(i*3+1.3,.55,text,ha='center',va='center')
    ax.set(xlim=(-.1,8.7),ylim=(-.1,2));ax.set_title('先按64个规则前缀分组，再把每组4个物体变体放在同一侧');save(fig,out,'01_group_split.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.3))
    for i,run in enumerate(r['runs']):
        steps=[t['step'] for t in run['validation_trace']]
        axs[0].plot(steps,[t['validation']['nll'] for t in run['validation_trace']],marker='o',label=str(run['seed']),color=COLORS[i])
        axs[1].plot(steps,[t['validation']['rule_accuracy'] for t in run['validation_trace']],marker='o',label=str(run['seed']),color=COLORS[i])
    axs[0].set(xlabel='训练步',ylabel='验证每token NLL');axs[1].set(xlabel='训练步',ylabel='验证规则准确率',ylim=(-.02,.3));axs[0].legend(title='初始化种子');save(fig,out,'02_metric_gap.png')
    fig,ax=plt.subplots(figsize=(8.5,3.3));x=np.arange(7)
    for i,run in enumerate(r['runs']):ax.plot(x,run['test']['position_nll'],marker='o',label=str(run['seed']),color=COLORS[i])
    ax.set(xticks=x,xticklabels=['颜色1','动物','动作','规则颜色2','物体','句号','EOS'],ylabel='测试每位置NLL');ax.legend();save(fig,out,'03_position_loss.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.2));names=[str(x['seed']) for x in r['runs']]
    vals=np.array([x['test']['task_success_rate'] for x in r['runs']]);cis=np.array([x['test']['wilson95_task_success'] for x in r['runs']]);axs[0].errorbar(names,vals,yerr=np.vstack([vals-cis[:,0],cis[:,1]-vals]),fmt='o',capsize=5,color=COLORS[0]);axs[0].set(ylabel='测试任务成功率',ylim=(0,1),title='每种子12组 Wilson近似95%区间')
    bottom=np.zeros(3)
    for j,key in enumerate(['success','wrong_rule','format_or_stop_failure']):
        values=[x['test']['error_counts'][key] for x in r['runs']];axs[1].bar(names,values,bottom=bottom,label={'success':'成功','wrong_rule':'规则错误','format_or_stop_failure':'格式或终止错误'}[key],color=COLORS[j]);bottom+=values
    axs[1].set(ylabel='互斥错误分类 组数',ylim=(0,16));axs[1].legend(fontsize=9,ncol=2);save(fig,out,'04_uncertainty_errors.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/results.json');p.add_argument('--output',default='figures');a=p.parse_args();main(a.results,a.output)
