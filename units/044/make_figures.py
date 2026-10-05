"""Original diagrams and measured plots, all derived from retained calculations."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
# Linux TTC collections expose only the first (JP) face to Matplotlib by default.
# Extract the installed SC face into an isolated cache; never silently emit tofu.
try:
    font_manager.findfont('Noto Sans CJK SC', fallback_to_default=False)
except ValueError:
    fontfile=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not fontfile.exists():
        raise RuntimeError('请安装Noto Sans CJK SC字体，然后重新生成图；不接受缺字输出')
    from fontTools.ttLib import TTCollection
    cache=Path(tempfile.gettempdir())/'dl044-font-cache';cache.mkdir(parents=True,exist_ok=True)
    extracted=cache/'NotoSansCJKSC-Regular.ttf'
    if not extracted.exists():
        collection=TTCollection(str(fontfile))
        font=next(f for f in collection.fonts if f['name'].getDebugName(1)=='Noto Sans CJK SC')
        font.save(extracted)
    font_manager.fontManager.addfont(str(extracted))
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':['Noto Sans CJK SC','DejaVu Sans'],'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.dpi':150})
BLUE='#2864a6';ORANGE='#d57b25';GREEN='#287b64';GREY='#64748b'

def box(ax,x,y,w,h,text,color='#e9f0f8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',facecolor=color,edgecolor='#7390ad',lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(ax,a,b,color=GREY,style='->'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle=style,color=color,lw=1.5))
def save(fig,out,name):
    fig.savefig(out/name,bbox_inches='tight',facecolor='white');plt.close(fig)

def main(results=HERE/'outputs/results.json',output=HERE/'figures'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);r=json.loads(Path(results).read_text());runs=r['runs'];h=r['hand']['rounds']
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,skip in zip(axs,[False,True]):
        ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off');ax.set_title('残差块：两条输入路径' if skip else '普通块：仅残差分支对应的堆叠')
        box(ax,.2,2.4,1.2,.7,'x');box(ax,2,2.1,3.4,1.3,'Conv → BN → ReLU\nConv → BN\nF(x; θ)');box(ax,7.1,2.4,2.5,.7,'ReLU → y')
        arrow(ax,(1.45,2.75),(1.95,2.75))
        if skip:
            box(ax,6,2.4,.55,.7,'+','#fce5c9');arrow(ax,(5.45,2.75),(5.95,2.75));arrow(ax,(6.6,2.75),(7.05,2.75))
            ax.plot([.8,.8,6.3,6.3],[3.15,4.6,4.6,3.2],color=ORANGE,lw=2);arrow(ax,(6.3,3.6),(6.3,3.12),ORANGE)
            ax.text(3.5,4.8,'identity shortcut：无新增参数',ha='center',color=ORANGE)
            ax.text(5,1.05,'反向：g_x = J_F^T u + u；u = g_y ⊙ 1[F+x>0]',ha='center',fontsize=10)
        else:
            arrow(ax,(5.45,2.75),(7.05,2.75));ax.text(5,1.05,'反向：g_x = J_F^T u；u = g_y ⊙ 1[F>0]',ha='center',fontsize=10)
        ax.text(5,.3,'主实验两边的 Conv、BN、宽度与初始化逐项相同',ha='center',fontsize=9,color=GREY)
    fig.tight_layout();save(fig,out,'01_plain_residual.png')
    fig,ax=plt.subplots(figsize=(10.8,4.4));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
    box(ax,.15,2.4,2,.9,'输入 x\nN × 8 × 13 × 13');box(ax,3,3.65,4,1.15,'3×3, 8→16, stride 2, pad 1\n3×3, 16→16, stride 1, pad 1\nF: N × 16 × 7 × 7')
    box(ax,3,1.1,4,1.1,'1×1 投影 P, 8→16, stride 2\nshortcut: N × 16 × 7 × 7','#fce5c9');box(ax,8.1,2.5,.7,.7,'+');box(ax,9.5,2.25,2.2,1.2,'输出 y\nN × 16 × 7 × 7')
    for a,b in [((2.2,2.85),(3,4.2)),((2.2,2.85),(3,1.65)),((7.05,4.2),(8.4,3.22)),((7.05,1.65),(8.4,2.45)),((8.85,2.85),(9.45,2.85))]:arrow(ax,a,b)
    ax.text(6,5.6,'奇数尺寸也要实际代入输出公式：floor((13+2−3)/2)+1 = 7',ha='center')
    ax.text(6,.35,'反向输入必须相加：g_x = J_F^T g_y + P^T g_y；投影参数也接收梯度',ha='center')
    save(fig,out,'02_projection_shapes.png')
    fig,ax=plt.subplots(figsize=(11,4.3));ax.axis('off');ax.set(xlim=(0,12),ylim=(0,5))
    labels=[('x = [−1, 1]',.15,2.6,1.8),('z = wx+b\n[−0.25, 0.75]',2.6,2.5,2.1),('h = ReLU(z)\n[0, 0.75]',5.3,2.5,2.1),('F = vh+c\n[−0.1, 0.2]',8,2.5,2.1)]
    for text,x,y,w in labels:box(ax,x,y,w,1,text)
    for a,b in [((2,3.1),(2.55,3.1)),((4.75,3),(5.25,3)),((7.45,3),(7.95,3))]:arrow(ax,a,b)
    ax.text(6,4.5,'同一组旧参数 theta_0 = (0.5, 0.25, 0.4, −0.1, 0.8, 0.1)',ha='center')
    box(ax,7.8,.45,3.3,1,'s = x+F = [−1.1, 1.2]\nm = mean(s) = 0.05','#fce5c9')
    box(ax,2.7,.45,4,1,'y_hat = am+d = 0.14\n(y_hat−t)²/2 = 0.0098')
    arrow(ax,(9.05,2.45),(9.4,1.5));arrow(ax,(7.75,.95),(6.75,.95));ax.plot([1.1,1.1,9.4],[2.55,.15,.15],color=ORANGE,lw=1.5);arrow(ax,(9.4,.15),(9.4,.42),ORANGE)
    save(fig,out,'03_hand_forward.png')
    fig,ax=plt.subplots(figsize=(9.5,3.2));arr=np.array([q['gradient'] for q in h[0]['rows']]+[h[0]['gradient']]);im=ax.imshow(arr,cmap='Blues',vmin=0,vmax=.5,aspect='auto')
    for (i,j),value in np.ndenumerate(arr):ax.text(j,i,f'{value:.4f}',ha='center',va='center',color='white' if value>.32 else '#172c44')
    ax.set(xticks=range(6),xticklabels=r['hand']['parameter_order'],yticks=range(3),yticklabels=['样本 A','样本 B','批梯度 = A+B'],title='样本路径必须在共享参数处累加（已含 1/N）');fig.colorbar(im,ax=ax,label='梯度');fig.tight_layout();save(fig,out,'04_parameter_ledger.png')
    fig,axs=plt.subplots(1,2,figsize=(9.8,3.4));steps=np.arange(3)
    axs[0].plot(steps,[q['loss'] for q in h],'o-',color=BLUE);axs[0].set(xlabel='已执行同步更新次数',ylabel='平均平方损失 / 2',xticks=steps,title='每次更新后重做前向')
    for i,label in enumerate(['A，目标0','B，目标1']):axs[1].plot(steps,[q['rows'][i]['pred'] for q in h],'o-',label=label)
    axs[1].axhline(0,color=GREY,lw=.7);axs[1].axhline(1,color=GREY,lw=.7);axs[1].set(xlabel='已执行同步更新次数',ylabel='预测',xticks=steps,title='不能沿用旧激活');axs[1].legend();fig.tight_layout();save(fig,out,'05_hand_updates.png')
    X=np.load(HERE/'data/train_x.npy');Y=np.load(HERE/'data/train_y.npy');fig,axs=plt.subplots(2,6,figsize=(10.5,3.6))
    for i,ax in enumerate(axs.flat):ax.imshow(X[i,0],cmap='gray',vmin=-.7,vmax=1.7);ax.set_title(f'训练 {i}: '+('横' if Y[i]==0 else '竖'),fontsize=10);ax.axis('off')
    fig.suptitle('原创12×12合成条纹；位置、长度、幅度与噪声均改变');fig.tight_layout();save(fig,out,'06_data.png')
    fig,axs=plt.subplots(2,2,figsize=(10.8,6.8),sharex=True)
    for row,b in enumerate([2,6]):
        for col,lr in enumerate([.03,1.]):
            ax=axs[row,col]
            for skip,color in [(False,BLUE),(True,ORANGE)]:
                rr=[q for q in runs if q['blocks']==b and q['lr']==lr and q['skip']==skip]
                for j,q in enumerate(rr):ax.plot([v['epoch'] for v in q['history']],[v['validation']['loss'] for v in q['history']],color=color,alpha=.55,lw=1.1,label=('Residual' if skip else 'Plain') if j==0 else None)
            ax.set(yscale='log',ylabel='验证交叉熵（eval 状态）',title=f'{b} 块 / {1+2*b} 卷积，学习率 {lr:g}',xlabel='epoch');ax.legend(fontsize=10);ax.grid(alpha=.15)
    fig.tight_layout();save(fig,out,'07_all_learning_curves.png')
    fig,axs=plt.subplots(2,2,figsize=(10.6,6.7))
    for row,b in enumerate([2,6]):
        for col,epoch in enumerate([0,18]):
            ax=axs[row,col]
            for skip,color in [(False,BLUE),(True,ORANGE)]:
                rr=[q for q in runs if q['blocks']==b and q['lr']==.03 and q['skip']==skip]
                for j,q in enumerate(rr):
                    pr=next(v['layers'] for v in q['probes'] if v['epoch']==epoch);vv=list(pr.values())
                    ax.plot(range(len(vv)),vv,'o-',ms=3,lw=1,alpha=.6,color=color,label=('Residual' if skip else 'Plain') if j==0 else None)
            ax.set(yscale='log',xlabel='卷积次序（最后一点为分类头）',ylabel='参数梯度 RMS',title=f'{b} 块，epoch {epoch}，学习率 0.03');ax.legend(fontsize=10);ax.grid(alpha=.15)
    fig.tight_layout();save(fig,out,'08_gradient_profiles.png')
    rr=[q for q in runs if q['blocks']==6 and q['lr']==1.];fig,axs=plt.subplots(1,2,figsize=(10.6,4.4));names=[('R' if q['skip'] else 'P')+str(q['seed']) for q in rr];x=np.arange(len(rr))
    for key,label,off,color in [('original_validation_loss','原始 BN eval',-.18,BLUE),('training_only_bn_recalibrated_validation_loss','仅训练数据重估 BN',.18,ORANGE)]:
        axs[0].bar(x+off,[q['bn_peak_diagnostic'][key] for q in rr],width=.35,label=label,color=color)
    axs[0].set(yscale='log',xticks=x,xticklabels=names,ylabel='验证交叉熵',title='高学习率各运行峰值时刻：相同权重');axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=9)
    for key,label,off,color in [('fixed_train_batch_eval_loss','eval',-.18,BLUE),('same_train_batch_train_mode_loss','train',.18,ORANGE)]:
        axs[1].bar(x+off,[q['bn_peak_diagnostic'][key] for q in rr],width=.35,label=label,color=color)
    axs[1].set(yscale='log',xticks=x,xticklabels=names,ylabel='固定32条训练样本交叉熵',title='同一小批：仅切换 BN 使用的统计');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=9)
    fig.tight_layout();save(fig,out,'09_bn_instability.png')
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.8));tests={q['id']:q for q in r['tests']}
    for ax,b in zip(axs,[2,6]):
        for seed in [11,29,47]:
            vals=[100*tests[f'{name}_b{b}_lr0.03_s{seed}']['accuracy'] for name in ['plain','residual']]
            ax.plot([0,1],vals,'o-',label=f'seed {seed}')
        ax.set(xticks=[0,1],xticklabels=['Plain','Residual'],ylim=(97.5,100.15),ylabel='固定测试准确率 (%)',title=f'{b} 块：同一数据、同一初始化配对');ax.legend(fontsize=10);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,out,'10_paired_test.png')
    chosen=tests['residual_b6_lr0.03_s47'];errors=chosen['mistakes'];XX=np.load(HERE/'data/test_x.npy');YY=np.load(HERE/'data/test_y.npy');fig,axs=plt.subplots(1,len(errors),figsize=(10.5,2.5))
    for ax,idx in zip(np.atleast_1d(axs),errors):
        ax.imshow(XX[idx,0],cmap='gray',vmin=-.7,vmax=1.7);ax.set_title(f'测试 {idx}\n真:{"横" if YY[idx]==0 else "竖"} / 预测:{"横" if chosen["predictions"][idx]==0 else "竖"}',fontsize=9);ax.axis('off')
    fig.suptitle('Residual 6块 seed47 的全部5个错误；没有只挑容易解释的错误');fig.tight_layout();save(fig,out,'11_all_errors.png')
    (out/'figure_index.json').write_text(json.dumps([p.name for p in sorted(out.glob('*.png'))],ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default=str(HERE/'outputs/results.json'));p.add_argument('--output',default=str(HERE/'figures'));a=p.parse_args();main(a.results,a.output)
