"""Original architecture, numerical ledger and measured-resource figures."""
from pathlib import Path
import os,argparse,json,gzip
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'../../tmp/047-mpl'));os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'../../tmp/047-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
FONT=os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if not Path(FONT).is_file():raise FileNotFoundError('CJK font missing; set DL_CJK_FONT')
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':12,'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
COLORS=['#22679d','#b46a28','#368871'];KINDS=['cnn','vit','hybrid'];LABELS=['CNN','小ViT','混合结构']
def grid(ax,m,title,low=None,high=None):
    a=np.array(m);lo=low if low is not None else -np.abs(a).max();hi=high if high is not None else np.abs(a).max();im=ax.imshow(a,cmap='RdBu_r',vmin=lo,vmax=hi);ax.set_title(title);ax.set_xticks(range(a.shape[1]) if a.shape[1]<=4 else [0,a.shape[1]-1]);ax.set_yticks(range(a.shape[0]) if a.shape[0]<=4 else [0,a.shape[0]-1])
    if a.size<=16:
        for ij in np.ndindex(a.shape):
            rgb=im.cmap(im.norm(a[ij]))[:3];luminance=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];ax.text(ij[1],ij[0],f'{a[ij]:.4f}',ha='center',va='center',fontsize=12,color='black' if luminance>.5 else 'white')
def main(out):
    out.mkdir(parents=True,exist_ok=True);r=json.loads((ROOT/'outputs/results.json').read_text());d=json.loads((ROOT/'data/images.json').read_text());h=r['hand'];tr=json.loads(gzip.decompress((ROOT/'outputs/training_traces.json.gz').read_bytes()))
    def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5));img=np.array(d['train']['x'][0][0]);ax[0].imshow(img,cmap='gray',vmin=-.4,vmax=1.4)
    for row in range(4):
        for col in range(4):ax[0].add_patch(Rectangle((2*col-.5,2*row-.5),2,2,fill=False,edgecolor='#e38924',lw=1.5));ax[0].text(2*col,2*row,str(row*4+col),color='#ffe69a',fontsize=8)
    ax[0].set_title('8×8切为16个2×2 patch，按行优先编号');ax[0].set_xticks(range(8));ax[0].set_yticks(range(8))
    tokens=np.stack([img[a:a+2,b:b+2].ravel() for a in range(0,8,2) for b in range(0,8,2)])
    ax[1].imshow(tokens,cmap='RdBu_r',aspect='auto',vmin=-1.4,vmax=1.4);ax[1].set_title('token矩阵：16×4，尚未线性嵌入');ax[1].set(xlabel='patch内像素索引',ylabel='token位置',xticks=range(4),yticks=[0,3,7,11,15]);fig.tight_layout();save(fig,'01_patches.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));cnn=np.zeros((16,16));vit=np.ones((16,16))
    for i in range(16):
        rr,cc=divmod(i,4)
        for j in range(16):rj,cj=divmod(j,4);cnn[i,j]=abs(rr-rj)<=1 and abs(cc-cj)<=1
    for a,m,title in zip(ax,[cnn,vit],['示意局部邻接：固定邻域','全局注意力：潜在全连接']):a.imshow(m,cmap='Blues',vmin=0,vmax=1);a.set_title(title);a.set(xlabel='输入位置',ylabel='输出位置',xticks=[0,5,10,15],yticks=[0,5,10,15])
    fig.suptitle('连接是否允许 ≠ 当前贡献大小；右图并非注意力权重的因果解释');fig.tight_layout();save(fig,'02_connectivity.png')
    fig,ax=plt.subplots(figsize=(10,2.8));ax.axis('off');nodes=[('patch + position\nB × 16 × 8',.02),('LN → 2头注意力\n加回原输入',.26),('LN → FFN\n8→16→8\n加回残差',.51),('末端LN → 均值\n线性头 8→2',.77)]
    for label,x in nodes:ax.add_patch(Rectangle((x,.33),.2,.4,facecolor='#e8f0f7',edgecolor='#506f8b',lw=1.5));ax.text(x+.1,.53,label,ha='center',va='center',fontsize=12)
    for x in [.23,.48,.74]:ax.add_patch(FancyArrowPatch((x,.53),(x+.03,.53),arrowstyle='-|>',mutation_scale=12,color='#334155'))
    ax.text(.5,.05,'本实验单个pre-norm编码块，无CLS、dropout、预训练；混合结构先加3×3卷积stem',ha='center');save(fig,'03_block.png')
    fig,ax=plt.subplots(2,3,figsize=(10,5))
    for i,s in enumerate(h[0]['samples']):grid(ax[i,0],s['patches'],f'样本{i+1}：两个行patch',0,1);grid(ax[i,1],s['scores'],'QK转置：缩放因子√1',-.065,.065);grid(ax[i,2],s['attention'],'每行softmax',0,1)
    fig.tight_layout();save(fig,'04_hand_attention.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    for i in range(2):grid(ax[i],h[0]['samples'][i]['score_upstream'],f'样本{i+1}：dL/dS（每行和为0）',-.024,.024)
    fig.tight_layout();save(fig,'05_softmax_backward.png')
    fig,ax=plt.subplots(figsize=(10,3.4));a=np.array([s['contribution'] for s in h[0]['samples']]+[h[0]['gradient']]);v=np.abs(a).max();ax.imshow(a,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto');ax.set_xticks(range(9),['w0','w1','pos0','pos1','q','k','v','o','c']);ax.set_yticks(range(3),['样本1','样本2','相加']);ax.set_title('九个参数都接入目标；q与k的梯度较小但并非0')
    for ij in np.ndindex(a.shape):ax.text(ij[1],ij[0],f'{a[ij]:.4f}',ha='center',va='center',fontsize=12,color='white' if abs(a[ij])>.7*v else 'black')
    fig.tight_layout();save(fig,'06_hand_gradients.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.3));
    for i,c in enumerate(COLORS[:2]):ax[0].plot(range(3),[s['samples'][i]['loss'] for s in h],'o-',label=f'样本{i+1}',color=c)
    ax[0].plot(range(3),[s['loss'] for s in h],'s--',label='平均',color=COLORS[2]);ax[0].set(xlabel='更新次数',ylabel='half-MSE',xticks=[0,1,2]);ax[0].set_title('共享注意力参数：总损失下降，样本2变差')
    perm=r['permutation'];ax[1].bar(['无位置\n重排patch','固定位置\n只重排patch','位置与patch\n一同重排'],[perm['no_position_mean_invariance_max_abs'],perm['fixed_position_mean_change'],perm['joint_permutation_mean_invariance_max_abs']],color=COLORS);ax[1].set_ylabel('均值表示的最大绝对变化');ax[1].set_title('位置是内容以外的信息');handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False);fig.tight_layout(rect=[0,0,1,.84]);save(fig,'07_updates_positions.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));sizes=np.array([32,128]);
    for kind,label,c in zip(KINDS,LABELS,COLORS):
        ax[0].plot(sizes,[r['summary'][str(n)][kind]['accuracy'] for n in sizes],'o-',label=label,color=c);ax[1].plot(sizes,[r['summary'][str(n)][kind]['nll'] for n in sizes],'o-',label=label,color=c)
        for n in sizes:
            runs=[v for v in r['runs'] if v['kind']==kind and v['train_size']==n];ax[0].scatter([n]*3,[v['test']['accuracy'] for v in runs],color=c,s=15,alpha=.45)
    if 'orientation_reference' in r:ax[0].axhline(r['orientation_reference']['splits']['test']['accuracy'],color='#666',ls=':',label='事后邻接相关基线')
    for a in ax:a.set_xticks(sizes);a.set_xlabel('训练图数（全批次；每次200更新）')
    handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=4,frameon=False,fontsize=9)
    ax[0].set(ylabel='测试准确率',ylim=(0,1.05));ax[1].set_ylabel('测试NLL');fig.tight_layout(rect=[0,0,1,.84]);save(fig,'08_data_sensitivity.png')
    fig,ax=plt.subplots(2,3,figsize=(11,5.5))
    for row,n in enumerate([32,128]):
        for col,(kind,label) in enumerate(zip(KINDS,LABELS)):
            a=ax[row,col]
            for t in tr:
                if t['train_size']==n and t['kind']==kind:a.plot([s['step'] for s in t['states']],[s['train_nll'] for s in t['states']],label=str(t['seed']),lw=.8)
            a.set_title(f'{label}，n={n}');a.set(xlabel='更新次数',ylabel='训练NLL')
    handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False);fig.tight_layout(rect=[0,0,1,.94]);save(fig,'09_all_traces.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5));x=np.arange(3)
    for j,b in enumerate([32,128]):ax[0].bar(x+(j-.5)*.32,[a['unique_saved_storage_bytes']/1024 for a in r['resource_audit'] if a['batch']==b],.32,label=f'batch={b}')
    ax[0].set_xticks(x,LABELS);ax[0].set_ylabel('唯一反向保存storage / KiB');ax[0].set_title('CPU实测口径，包含输入和参数引用');handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
    ps=np.array([4,8,16,32]);N=(224//ps)**2;memory=12*N**2*4/(1024**2);ax[1].plot(ps,memory,'o-',color=COLORS[1]);ax[1].set_yscale('log');ax[1].set_xticks(ps);ax[1].set(xlabel='patch边长P，图像224×224',ylabel='单个稠密注意力矩阵 / MiB');ax[1].set_title('解析估计：B=1，12头，float32，无CLS');fig.tight_layout(rect=[0,0,1,.84]);save(fig,'10_resource.png')
    fig,ax=plt.subplots(1,3,figsize=(10,3.2));
    for j,(kind,label) in enumerate(zip(KINDS,LABELS)):
        run=next(z for z in r['runs'] if z['kind']==kind and z['train_size']==128 and z['seed']==4711);logits=np.array(run['test']['logits']);margins=logits[:,1]-logits[:,0];y=np.array(d['test']['y']);ax[j].hist(margins[y==0],bins=15,alpha=.6,label='竖条0');ax[j].hist(margins[y==1],bins=15,alpha=.6,label='横条1');ax[j].axvline(0,color='#333',ls='--');ax[j].set(title=label,xlabel='类1分数 - 类0分数',ylabel='图像数');
    handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False);fig.suptitle('同一个128图/seed4711协议：准确率与概率置信度不是同一指标',y=1.14);fig.tight_layout(rect=[0,0,1,.84]);save(fig,'11_margins.png')
    print(json.dumps({'figures':11,'font':FONT},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');main(p.parse_args().output)
