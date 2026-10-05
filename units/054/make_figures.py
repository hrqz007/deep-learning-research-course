"""Original information-boundary diagrams and data-backed objective diagnostics."""
from pathlib import Path
import os,json,argparse
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'tmp/mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'tmp/cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from generate_data import VOCAB
ROOT=Path(__file__).resolve().parent

def configure():
    f=Path(os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    if not f.exists():raise FileNotFoundError('Install Noto Sans CJK or set DL_CJK_FONT')
    font_manager.fontManager.addfont(str(f));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(f)).get_name(),'font.size':12,'axes.unicode_minus':False,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,p):fig.savefig(p,bbox_inches='tight',facecolor='white');plt.close(fig)
def mask(ax,a,title):
    ax.imshow(a,cmap='Blues',vmin=0,vmax=1);ax.set(title=title,xlabel='被读取的 key',ylabel='发起读取的 query');ax.set_xticks(range(len(a)));ax.set_yticks(range(len(a)))
def main(results=ROOT/'outputs/results.json',output=ROOT/'figures'):
    configure();r=json.loads(Path(results).read_text());out=Path(output);out.mkdir(exist_ok=True,parents=True)
    fig,axs=plt.subplots(1,3,figsize=(13,4.4));encdec=np.zeros((7,7));encdec[:3,:3]=1;encdec[3:,:3]=1;encdec[3:,3:]=np.tril(np.ones((4,4)))
    for ax,a,title in zip(axs,[np.tril(np.ones((6,6))),np.ones((6,6)),encdec],['自回归：当前输入及其前缀','MLM：损坏后的双向输入','编码解码：源0-2，目标3-6']):mask(ax,a,title)
    axs[2].axhline(2.5,c='orange');axs[2].axvline(2.5,c='orange');fig.suptitle('蓝色允许读取；MLM仍须把被预测原词元从输入替换',y=1.03);fig.tight_layout();save(fig,out/'01_objectives.png')
    fig,axs=plt.subplots(2,1,figsize=(11,3.4));texts=[['<bos>','蓝','猫','追','红','球'],['蓝','猫','追','红','球','<eos>']]
    for ax,title,rows in [(axs[0],'AR：同一列的输入预测下一词元',texts),(axs[1],'MLM：只有选中位置计算loss',[['<bos>','<mask>','猫','<mask>','红','<mask>','<eos>'],['忽略','蓝','忽略','追','忽略','球','忽略']])]:
        ax.axis('off');tab=ax.table(cellText=rows,rowLabels=['输入','目标'],cellLoc='center',loc='center');tab.auto_set_font_size(False);tab.set_fontsize(13);tab.scale(1,1.7);ax.set_title(title)
    fig.tight_layout();save(fig,out/'02_targets.png')
    lengths=[3,2];seg=np.repeat([0,1],lengths);good=(seg[:,None]==seg[None,:])&np.tril(np.ones((5,5),dtype=bool));fig,axs=plt.subplots(1,2,figsize=(10,4))
    mask(axs[0],good,'独立文档packing：分块因果');mask(axs[1],np.tril(np.ones((5,5))),'错误对照：全局因果仍跨文档')
    for ax in axs:ax.axhline(2.5,c='orange');ax.axvline(2.5,c='orange')
    fig.tight_layout();save(fig,out/'03_packing.png')
    fig,axs=plt.subplots(1,2,figsize=(11,3.8));h=r['hand_nll'];probs=np.array(h[0]['probabilities'])[0,:2];grads=np.array(h[0]['gradient'])[0]
    for j,label in enumerate(['类0','类1','类2']):axs[0].bar(np.arange(2)+(.23*j-.23),probs[:,j],width=.23,label=label)
    axs[0].set(xticks=[0,1],xlabel='有效位置',ylabel='概率',title='手算softmax：目标为类0和类1');axs[0].legend()
    im=axs[1].imshow(grads,cmap='coolwarm',vmin=-.25,vmax=.25);axs[1].set(xticks=[0,1,2],yticks=[0,1,2],xlabel='类别',ylabel='位置',title='按2个有效token平均的梯度')
    for (i,j),v in np.ndenumerate(grads):axs[1].text(j,i,f'{v:.3f}',ha='center',va='center')
    fig.tight_layout();save(fig,out/'04_nll.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));axs[0].bar(['2个token\n每个0.1','6个token\n每个2.0'],[.1,2.],color=['#467f98','#d18a49']);axs[0].set(ylabel='每token平均NLL',title='两组有效目标数不同')
    axs[1].bar(['正确token加权','错误均值再平均'],[1.525,1.05],color=['#467f98','#d18a49']);axs[1].set(ylabel='总体NLL',title='同一组loss，权重决定结果')
    for i,v in enumerate([1.525,1.05]):axs[1].text(i,v+.04,str(v),ha='center')
    axs[1].set_ylim(0,1.8);fig.tight_layout();save(fig,out/'05_reduction.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4));vals=[r['blocked_cross_document_change'],r['leaky_cross_document_change'],r['missing_position_reset_max_error']]
    axs[0].bar(['分块mask\n修改前文档','全局因果\n修改前文档','分块mask\n未重置位置'],vals,color=['#467f98','#d18a49','#d18a49']);axs[0].set(ylabel='最大logit差',title='保持独立样本需同时处理mask与位置')
    for i,v in enumerate(vals):axs[0].text(i,v+.025,f'{v:.4g}',ha='center')
    d=r['copy_diagnostic'];axs[1].bar(['错误未移位标签','正确下一词元标签'],[d['wrong_unshifted_nll'],d['correct_shifted_nll']],color=['#d18a49','#467f98']);axs[1].set(ylabel='平均NLL',title='固定复制规则在错误标签上看似完美')
    fig.tight_layout();save(fig,out/'06_diagnostics.png');print('Rendered 6 figures');return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default=str(ROOT/'outputs/results.json'));p.add_argument('--output',default=str(ROOT/'figures'));a=p.parse_args();main(a.results,a.output)
