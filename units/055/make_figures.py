"""Seven original diagrams from the complete predeclared run matrix."""
from pathlib import Path
import os,json,argparse
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'tmp/mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'tmp/cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent
COLORS={16:'#397d9c',32:'#c17a43'}
def configure():
    f=Path(os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    if not f.exists():raise FileNotFoundError('Install Noto Sans CJK or set DL_CJK_FONT')
    font_manager.fontManager.addfont(str(f));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(f)).get_name(),'font.size':12,'axes.unicode_minus':False,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,p):fig.savefig(p,bbox_inches='tight',facecolor='white');plt.close(fig)
def group(runs,d,pool):return [r for r in runs if r['config']['d']==d and r['config']['pool']==pool]
def main(results=ROOT/'outputs/results.json',output=ROOT/'figures'):
    configure();obj=json.loads(Path(results).read_text());runs=obj['runs'];out=Path(output);out.mkdir(exist_ok=True,parents=True)
    fig,ax=plt.subplots(figsize=(12,3.4));ax.axis('off');ax.set(xlim=(0,12),ylim=(0,3))
    labels=['输入 B × 7\n[BOS, 六个正文词元]','词嵌入 + 位置\nB × 7 × D','两层因果块\n+ 末端 LN','词表线性头\nB × 7 × 20','有效目标 NLL\n7B 个预测对']
    for i,t in enumerate(labels):
        ax.add_patch(FancyBboxPatch((i*2.4+.08,1.3),2.08,1.1,boxstyle='round,pad=.05',fc='#edf3f7',ec='#397d9c'));ax.text(i*2.4+1.12,1.85,t,ha='center',va='center',fontsize=11)
        if i<4:ax.annotate('',(i*2.4+2.42,1.85),(i*2.4+2.2,1.85),arrowprops={'arrowstyle':'->'})
    ax.text(6,.55,'标签 = [六个正文词元, EOS]；移位一次；无padding；不读取测试集loss',ha='center')
    save(fig,out/'01_training_flow.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4));labels=['词嵌入','位置表','两层块','末端LN','输出头'];vals=[]
    for d in [16,32]:vals.append([20*d,7*d,2*(8*d*d+11*d),2*d,20*d+20])
    bottom=np.zeros(2)
    for j,label in enumerate(labels):v=np.array(vals)[:,j];axs[0].bar(['D=16','D=32'],v,bottom=bottom,label=label);bottom+=v
    axs[0].set(ylabel='参数个数',title='从模块逐项计数');axs[0].legend(fontsize=9)
    axs[1].bar(['训练池32\n唯一目标位置','训练池128\n唯一目标位置','每个run\n累计有效token'],[224,896,8960],color=['#397d9c','#397d9c','#c17a43']);axs[1].set(ylabel='计数',title='重复见到不等于新增独立内容')
    fig.tight_layout();save(fig,out/'02_counts.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.2))
    for ax,pool in zip(axs,[32,128]):
        for d in [16,32]:
            g=group(runs,d,pool);x=np.array([t['seen_tokens'] for t in g[0]['trace']]);a=np.array([[t['nll'] for t in r['trace']] for r in g]);mean=a.mean(0);sd=a.std(0,ddof=1)
            ax.plot(x,mean,'o-',label=f'D={d}',color=COLORS[d]);ax.fill_between(x,mean-sd,mean+sd,color=COLORS[d],alpha=.16)
        ax.set(title=f'训练池 {pool} 个唯一文档',xlabel='累计有效训练token（含重复）',ylabel='验证NLL（nat/token）');ax.legend()
    fig.suptitle('3种子均值；阴影为样本标准差，不是置信区间',y=1.02);fig.tight_layout();save(fig,out/'03_learning.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,pool in zip(axs,[32,128]):
        for offset,key,label,color in [(-.18,'final_train','训练','#397d9c'),(.18,'final_validation','验证','#c17a43')]:
            vals=[np.mean([r[key]['nll'] for r in group(runs,d,pool)]) for d in [16,32]]
            ax.bar(np.arange(2)+offset,vals,width=.34,label=label,color=color)
        ax.set(xticks=[0,1],xticklabels=['D=16','D=32'],ylabel='平均NLL',title=f'80步，训练池{pool}');ax.legend()
    fig.tight_layout();save(fig,out/'04_gap.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.2))
    for d in [16,32]:
        g=group(runs,d,128);steps=np.array([t['step'] for t in g[0]['trace']]);compute=3*g[0]['forward_matmul_flops_per_step']*steps/1e6
        means=np.mean([[t['nll'] for t in r['trace']] for r in g],axis=0)
        axs[0].plot(compute,means,'o-',c=COLORS[d],label=f'D={d}')
    axs[0].set(xlabel='估算训练矩阵乘法 MFLOPs',ylabel='验证NLL',title='训练池128；横轴不是实测时间');axs[0].legend()
    vals=[]
    for d,step in [(16,74),(32,20)]:vals.append(np.mean([next(t['nll'] for t in r['trace'] if t['step']==step) for r in group(runs,d,128)]))
    axs[1].bar(['D=16 / 74步','D=32 / 20步'],vals,color=[COLORS[16],COLORS[32]]);axs[1].set(ylabel='验证NLL',title='约2.4亿矩阵FLOPs，预算差约0.43%');axs[1].set_ylim(0,1.65)
    for i,v in enumerate(vals):axs[1].text(i,v+.03,f'{v:.4f}',ha='center')
    fig.tight_layout();save(fig,out/'05_compute.png')
    fig,ax=plt.subplots(figsize=(12,4.2))
    for d in [16,32]:
        for pool,style in [(32,'--'),(128,'-')]:
            vals=np.mean([r['final_validation']['position_nll'] for r in group(runs,d,pool)],axis=0)
            ax.plot(vals,marker='o',ls=style,c=COLORS[d],label=f'D={d}, 文档={pool}')
    ax.set(xticks=range(7),xticklabels=['颜色1','动物','动作','颜色2\n规则依赖','物体','句号','EOS'],ylabel='验证NLL',title='按预测位置拆解；低总体loss不等于学会依赖规则');ax.legend(ncol=2,fontsize=10)
    fig.tight_layout();save(fig,out/'06_positions.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4));groups=[(16,32),(16,128),(32,32),(32,128)];labs=[f'D{d}\n文档{p}' for d,p in groups]
    throughput=[[r['steady_tokens_per_second']/1000 for r in group(runs,d,p)] for d,p in groups]
    means=np.mean(throughput,axis=1);lo=means-np.min(throughput,axis=1);hi=np.max(throughput,axis=1)-means
    axs[0].bar(labs,means,yerr=np.array([lo,hi]),capsize=4,color=['#397d9c']*2+['#c17a43']*2);axs[0].set(ylabel='千有效token / 秒',title='排除前5步；误差线是3次最小到最大')
    rss=[np.mean([r['peak_worker_rss_bytes']/2**20 for r in group(runs,d,p)]) for d,p in groups]
    axs[1].bar(labs,rss,color='#8297a4');axs[1].set(ylabel='MiB',title='独立worker峰值RSS：含解释器与验证')
    fig.tight_layout();save(fig,out/'07_measurement.png');print('Rendered 7 figures');return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default=str(ROOT/'outputs/results.json'));p.add_argument('--output',default=str(ROOT/'figures'));a=p.parse_args();main(a.results,a.output)
