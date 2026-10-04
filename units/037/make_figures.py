"""Original mechanism / measured-result figures, with pre-write provenance checks."""
from pathlib import Path
import argparse,csv,io,hashlib,json,math,os,tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"course-037-mpl"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(tempfile.gettempdir())/"course-037-cache"))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
import experiment as e
BASE=Path(__file__).resolve().parent
OUTPUT_HASHES = {'ablation.csv': '865d113108e413e1dae3403833434a030ade7294a43ccde7d984972dc99cf190', 'calculations.json': 'd386873d1be294a5a348d9a8106fcf044bac4a4133ac8ea4bd015d6112045d67', 'decision-grid.json': '688f807278eafd452f2484d9ea6aeddda70769daa843d12ab01b8130461fc4f5', 'run-details.json': '987d5681fb9359f80d8fd6b3e13f1de4780fd567465c25bfdda8eae3806547be', 'test-predictions.csv': '4629468540912f4bb9f78ecd71cf012ebe476b00b9c8b0213e591c647166675d', 'training-history.csv': 'be857fb73bbe4bae26d23d00d66afe0b6f067d2171e8545c2b5a38516e3588a6'}
CJK=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if CJK.exists():font_manager.fontManager.addfont(str(CJK));FONT=font_manager.FontProperties(fname=str(CJK)).get_name()
else:FONT='DejaVu Sans'
plt.rcParams.update({'font.family':[FONT,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
C=['#365d9b','#dc8134','#168f83','#a85680','#637a3b','#ba4c4c','#7866ad']
LABELS=['固定预算','早停','dropout','权重衰减','合法增强','错误增强','标签平滑']
def main(source=None,destination=None):
    src=Path(source) if source else BASE/'outputs';dest=Path(destination) if destination else BASE/'figures'
    # Full bytes are verified before making the destination or a single figure.
    for name,digest in OUTPUT_HASHES.items():
        if hashlib.sha256((src/name).read_bytes()).hexdigest()!=digest:raise ValueError('unreviewed figure source: '+name)
    data=e.read_data();calc=json.loads((src/'calculations.json').read_text());details=json.loads((src/'run-details.json').read_text());grid=json.loads((src/'decision-grid.json').read_text())
    rows=list(csv.DictReader(io.StringIO((src/'ablation.csv').read_text())));hist=list(csv.DictReader(io.StringIO((src/'training-history.csv').read_text())))
    if details['protocol']['epochs']!=160 or len(details['runs'])!=21:raise ValueError('figures require full default protocol')
    dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.savefig(dest/name,bbox_inches='tight',facecolor='white');plt.close(fig)
    # 1: explicitly defined quadratic illustration, not an empirical neural result.
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5),layout='constrained')
    xx,yy=np.meshgrid(np.linspace(-.4,2.4,180),np.linspace(-1.5,.7,180));loss=.5*((xx-2)**2+.2*(yy+1)**2)
    axs[0].contour(xx,yy,loss,levels=[.05,.1,.2,.5,1,2],colors=C[0],alpha=.65)
    axs[0].contour(xx,yy,.5*(xx**2+yy**2),levels=[.1,.5,1,2],colors=C[1],linestyles='--',alpha=.55)
    lam=np.array([0,.1,1,10]);w1=2/(1+lam);w2=-.2/(.2+lam)
    axs[0].plot(w1,w2,'o-',color=C[2]);
    for a,b,l in zip(w1,w2,lam):axs[0].annotate('λ='+str(l),(a,b),xytext=(4,7),textcoords='offset points',fontsize=8)
    axs[0].set(xlabel='参数 w1',ylabel='参数 w2',title='数据等高线与向零收缩的偏好')
    axs[1].plot(lam,np.sqrt(w1*w1+w2*w2),'o-',label='参数范数',color=C[2]);axs[1].plot(lam,.5*((w1-2)**2+.2*(w2+1)**2),'s-',label='纯数据损失',color=C[0])
    axs[1].set(xscale='symlog',xlabel='λ（含零）',title='更小范数也可能付出拟合代价');axs[1].legend(frameon=False)
    save(fig,'01_weight_penalty.png')
    # 2: exact finite masks for the first sample.
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.3),layout='constrained');masks=list(__import__('itertools').product((0,1),repeat=2));h=np.array([1.2,.8]);xs=np.arange(4)
    for j in range(2):axs[0].bar(xs+(-.18 if j==0 else .18),[m[j]*h[j]*2 for m in masks],width=.35,label=f'隐藏单元 {j+1}',color=C[j])
    axs[0].set(xticks=xs,xticklabels=[str(x) for x in masks],ylabel='mask × h / q',xlabel='两个独立 Bernoulli 掩码',title='每个掩码概率 1/4，q=1/2');axs[0].legend(frameon=False)
    p=np.array([np.sum(np.array(m)*h*2*np.array([.6,-.5]))+.1 for m in masks]);axs[1].bar(xs,p,color=C[0]);axs[1].axhline(.42,color=C[1],ls='--',label='均值 = 无 dropout 输出 .42')
    axs[1].set(xticks=xs,xticklabels=[str(x) for x in masks],ylabel='仿射输出 P',title='保持的是条件均值');axs[1].legend(frameon=False,fontsize=8)
    save(fig,'02_dropout_masks.png')
    # 3: complete data/penalty gradient decomposition.
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5),gridspec_kw={'width_ratios':[.95,1.3]},layout='constrained')
    ax=axs[0];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    nodes=[(.17,.87,'XW+b\n仿射'),(.75,.87,'ReLU\n门控'),(.75,.48,'M/q\n随机门控'),(.17,.48,'输出 P\n半平方平均'),(.17,.1,'数据梯度\n＋ λSθ'),(.75,.1,'全部参数\n同时更新')]
    for x,y,t in nodes:ax.text(x,y,t,ha='center',va='center',bbox={'boxstyle':'round,pad=.4','facecolor':'#f2f5f9','edgecolor':'#bac8da'},fontsize=9)
    for a,b in zip(nodes,nodes[1:]):ax.add_patch(FancyArrowPatch((a[0],a[1]),(b[0],b[1]),arrowstyle='-|>',mutation_scale=10,color=C[0],shrinkA=28,shrinkB=28))
    r=calc['hand']['initial'];ix=np.arange(9);axs[1].bar(ix-.18,r['data_gradient'],.36,label='数据路径总和',color=C[0]);axs[1].bar(ix+.18,r['penalty_gradient'],.36,label='惩罚路径',color=C[1]);axs[1].axhline(0,color='#999',lw=.6)
    axs[1].set(xticks=ix,xticklabels=e.PARAM_NAMES,ylabel='当前固定掩码下的梯度',title='九个参数全部保留');axs[1].tick_params(axis='x',rotation=45);axs[1].legend(frameon=False,fontsize=8)
    save(fig,'03_trace_gradients.png')
    # 4: nonlinear expectations are not preserved.
    en=calc['enumeration'];fig,axs=plt.subplots(1,2,figsize=(9.2,3.3),layout='constrained')
    axs[0].bar(['关闭 dropout','枚举16种掩码的平均'],[en['no_dropout']['data_loss'],en['mean_data_loss']],color=[C[0],C[1]])
    for i,v in enumerate([en['no_dropout']['data_loss'],en['mean_data_loss']]):axs[0].text(i,v+.006,f'{v:.4f}',ha='center')
    axs[0].set(ylabel='两样本平均半平方损失',ylim=(0,.29),title='线性输出均值相同，平方损失不同')
    ix=np.arange(2);a=1/(1+np.exp(-np.array(en['mean_prediction'])));b=en['mean_sigmoid_prediction']
    axs[1].bar(ix-.18,a,.36,label='sigmoid(E[P])',color=C[0]);axs[1].bar(ix+.18,b,.36,label='E[sigmoid(P)]',color=C[1]);axs[1].set(xticks=ix,xticklabels=['样本1','样本2'],ylim=(.35,.66),ylabel='附加 sigmoid 后的数值',title='非线性预测也不保持');axs[1].legend(frameon=False,fontsize=8)
    save(fig,'04_expectation_limits.png')
    # 5: dropout modes, observed outputs (the fixed seed only illustrates).
    fig,axs=plt.subplots(1,4,figsize=(10,2.9),layout='constrained')
    for ax,r in zip(axs,calc['mode_probe']['dropout_modes']):
        im=ax.imshow(r['first'],vmin=0,vmax=2,cmap='Blues',aspect='equal')
        for i in range(2):
            for j in range(6):ax.text(j,i,str(int(r['first'][i][j])),ha='center',va='center',fontsize=9,color='white' if r['first'][i][j]==2 else '#333')
        ax.set(xticks=[],yticks=[],title=('train' if r['training'] else 'eval')+'\n'+('记录梯度' if r['grad_enabled'] else 'no_grad'))
    fig.suptitle('同一全1输入：模式控制 dropout；no_grad 控制运算图',fontsize=12);save(fig,'05_train_eval.png')
    # 6: all early-stop seeds, displayed against the matched baseline path.
    fig,axs=plt.subplots(1,3,figsize=(10,3.3),layout='constrained')
    for ax,seed in zip(axs,e.SEEDS):
        r=[a for a in hist if a['config']=='baseline' and int(a['seed'])==seed];d=next(a for a in details['runs'] if a['config']=='early_stop' and a['seed']==seed)
        ep=[int(a['epoch']) for a in r];ax.plot(ep,[float(a['train_ce']) for a in r],color=C[0],label='训练 CE');ax.plot(ep,[float(a['validation_ce']) for a in r],color=C[1],label='验证 CE')
        ax.axvline(d['selected_epoch'],color=C[2],label='恢复点');ax.axvline(d['epochs_run'],color=C[3],ls='--',label='停止点')
        ax.set(xlabel='epoch',ylabel='关闭 dropout 的硬标签 CE',title=f'seed {seed}：选{d["selected_epoch"]} / 停{d["epochs_run"]}')
    axs[0].legend(frameon=False,fontsize=8);save(fig,'06_early_stopping.png')
    # 7: label preservation / wrong semantic mirror, no real-world claims.
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.4),layout='constrained')
    for ax,kind in zip(axs,['valid','invalid']):
        ax.axvspan(-1,0,color=C[0],alpha=.08);ax.axvspan(0,1,color=C[1],alpha=.08);ax.axvline(0,color='#777',ls=':')
        starts=np.array([[.45,.6],[-.7,-.4],[.25,-.65]])
        ends=starts.copy();ends[:,1 if kind=='valid' else 0]*=-1
        for a,b in zip(starts,ends):ax.scatter(*a,s=55,color=C[int(a[0]>0)]);ax.scatter(*b,s=55,facecolors='none',edgecolors=C[int(a[0]>0)]);ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':'#566477'})
        ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='信号坐标 x1',ylabel='干扰坐标 x2',title='x2 翻转：多数类不变' if kind=='valid' else 'x1 翻转但不改标签：多数类反转')
    save(fig,'07_augmentation_semantics.png')
    # 8: smoothing changes target and optimal probability, not merely scale.
    z=np.linspace(-4,7,400);pr=1/(1+np.exp(-z));fig,axs=plt.subplots(1,2,figsize=(9.2,3.4),layout='constrained')
    for target,color,label in [(1,C[0],'硬标签 y=1'),(.95,C[1],'平滑目标 0.95')]:
        loss=np.logaddexp(0,z)-target*z;axs[0].plot(z,loss,color=color,label=label);axs[1].plot(z,pr-target,color=color,label=label)
    axs[0].set(xlabel='单个二分类 logit z',ylabel='交叉熵',title='α=.1，向两个类别均匀分配');axs[0].legend(frameon=False)
    axs[1].axhline(0,color='#999',lw=.6);axs[1].axvline(math.log(19),color=C[2],ls=':',label='log(19) ≈ 2.944')
    axs[1].set(xlabel='logit z',ylabel='局部导数 sigmoid(z) - target',title='过度自信时平滑梯度可以反向');axs[1].legend(frameon=False,fontsize=8);save(fig,'08_label_smoothing.png')
    # 9: no best-seed selection, no confidence-interval styling.
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
    for i,config in enumerate(e.CONFIGS):
        sub=[a for a in rows if a['config']==config];val=[float(a['test_ce']) for a in sub];axs[0].scatter(np.full(3,i)+[-.12,0,.12],val,color=C[i],s=38)
        axs[0].plot([i-.22,i+.22],[np.mean(val)]*2,color=C[i],lw=2)
        axs[1].scatter([float(a['updates']) for a in sub],np.full(3,i)+[-.12,0,.12],color=C[i],s=34)
    axs[0].set(xticks=range(7),xticklabels=LABELS,ylabel='固定测试集硬标签 CE',title='每点一次预定运行；短线仅为三次均值');axs[0].tick_params(axis='x',rotation=30)
    axs[1].set(yticks=range(7),yticklabels=LABELS,xlabel='实际 optimizer 更新数',title='相同上限不等于相同消耗');axs[1].invert_yaxis();save(fig,'09_ablation_and_budget.png')
    # 10-12: decision surfaces, shared scale, designated seed, complete arms.
    for num,configs in [(10,['baseline','dropout','augment_valid']),(11,['early_stop','decay','label_smooth']),(12,['augment_invalid'])]:
        n=len(configs);fig,axs=plt.subplots(1,n,figsize=(9.6 if n>1 else 5.2,3.4),layout='constrained',squeeze=False)
        for ax,config in zip(axs[0],configs):
            im=ax.imshow(grid['probabilities'][config],origin='lower',extent=(-1,1,-1,1),vmin=0,vmax=1,cmap='coolwarm',aspect='equal')
            ax.contour(grid['axis'],grid['axis'],grid['probabilities'][config],levels=[.5],colors='black',linewidths=.9)
            x=data['train']['x'];y=data['train']['y'];flipped=y!=data['train']['clean_y']
            ax.scatter(x[:,0],x[:,1],c=y,cmap='coolwarm',vmin=0,vmax=1,edgecolors='white',s=14,lw=.5)
            ax.scatter(x[flipped,0],x[flipped,1],facecolors='none',edgecolors='#202020',s=48,lw=.9)
            ax.set(xlabel='x1',ylabel='x2',title=LABELS[list(e.CONFIGS).index(config)]+'  seed370')
        fig.colorbar(im,ax=list(axs[0]),label='预测 P(Y=1|x)',shrink=.85);save(fig,f'{num:02d}_decision_surfaces.png')
    # 13: theoretical calibration boundary and measured Brier deltas (same test rows).
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.3),layout='constrained');eta=np.linspace(0,1,100)
    axs[0].plot(eta,eta,color='#888',ls='--',label='真实条件概率 η');axs[0].plot(eta,.9*eta+.05,color=C[6],label='无限制平滑目标最优值 .9η+.05')
    axs[0].set(xlabel='η = P(Y=1|X=x)',ylabel='预测概率',title='即使精确优化，也未必校准');axs[0].legend(frameon=False,fontsize=8)
    for seed in e.SEEDS:
        a=next(x for x in rows if x['config']=='baseline' and int(x['seed'])==seed);b=next(x for x in rows if x['config']=='label_smooth' and int(x['seed'])==seed)
        axs[1].plot([0,1],[float(a['test_brier']),float(b['test_brier'])],'o-',label=f'seed {seed}',alpha=.85)
    axs[1].set(xticks=[0,1],xticklabels=['固定预算基线','标签平滑'],ylabel='测试 Brier 均值',title='本次配对观察也不是校准保证');axs[1].legend(frameon=False,fontsize=8);save(fig,'13_calibration_boundary.png')
    return {'figures':13,'source_files':len(OUTPUT_HASHES)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args();print(json.dumps(main(a.source,a.destination)))
