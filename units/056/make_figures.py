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
    fig,ax=plt.subplots(figsize=(8.5,3.2));x=np.arange(4)
    for i,(name,key) in enumerate([('原分布','T1'),('温度0.5','T0.5'),('top-k2','k2'),('top-p0.6','p0.6')]):ax.bar(x+(i-1.5)*.18,r['hand_calculation'][key],width=.18,label=name,color=COLORS[i])
    ax.set(xticks=x,xticklabels=['A','B','C','D'],ylabel='重归一化概率',ylim=(0,.65));ax.legend(ncol=2);save(fig,out,'01_probabilities.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.2))
    for ax,cached in zip(axs,[False,True]):
        matrix=np.zeros((7,7))
        for i in range(7):matrix[i,:i+1]=.3;matrix[i,i if cached else 0:i+1]=1
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1);ax.set(xticks=range(7),xticklabels=range(1,8),yticks=range(7),yticklabels=range(1,8),xlabel='输入位置',ylabel='解码步骤',title='缓存：只投影新位置' if cached else '无缓存：重算全部前缀')
    fig.suptitle('深蓝=本步计算QKV/FFN；浅蓝=此前K/V复用',y=1.04);save(fig,out,'02_cache_timeline.png')
    lengths=np.arange(1,129);fig,axs=plt.subplots(1,2,figsize=(9,3.2))
    axs[0].plot(lengths,np.cumsum(lengths**2),label='全前缀 attention 分数',color=COLORS[0]);axs[0].plot(lengths,np.cumsum(lengths),label='单步缓存 attention 分数',color=COLORS[1]);axs[0].set(xlabel='总输入长度T',ylabel='累计分数元素数 每层每头');axs[0].legend(fontsize=9)
    axs[1].plot(lengths,2*2*1*lengths*16*4/1024,color=COLORS[2]);axs[1].set(xlabel='缓存长度T',ylabel='K和V纯张量载荷 KiB',title='2层 B1 D16 float32 理论延长');save(fig,out,'03_cost_memory.png')
    s=r['summary'];fig,axs=plt.subplots(1,2,figsize=(9,3.2));names=[x['config'] for x in s]
    axs[0].bar(names,[x['mean_length'] for x in s],color=COLORS);axs[0].set(ylabel='新增token均值 含EOS',ylim=(0,7));axs[0].tick_params(axis='x',rotation=25)
    axs[1].bar(names,[x['eos_fraction'] for x in s],color=COLORS);axs[1].set(ylabel='EOS终止比例',ylim=(0,1.1));axs[1].tick_params(axis='x',rotation=25);save(fig,out,'04_outputs.png')
    fig,ax=plt.subplots(figsize=(7.5,3.1));a=r['fixed_workload_seconds'];ax.boxplot([np.array(a[k])*1000 for k in ['full','cache']],tick_labels=['全前缀重算','KV缓存'],showmeans=True);ax.set(ylabel='固定7步墙钟时间 ms',title='20次交替测量；3次warm-up；CPU单线程');save(fig,out,'05_timing.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/results.json');p.add_argument('--output',default='figures');a=p.parse_args();main(a.results,a.output)
